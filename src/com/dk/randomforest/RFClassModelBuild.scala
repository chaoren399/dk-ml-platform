package com.dk.randomforest

import com.dakuai.util.DKUtil
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, BinaryClassificationMetrics}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by Administrator on 2016/3/23 0023.
 */
object RFClassModelBuild {
  def main(args: Array[String]) {
    if (args.length != 4) {
      print("-------------args error--------------")
      System.exit(0)
    }

    val inputPath = args(0)
    val modelPath = args(1)
    val dataType = args(2)
    val numClasses = args(3).toInt

    val conf = new SparkConf()
    conf.setAppName("RFClassModelBuild")

    val sc = new SparkContext(conf)

    val data = DKUtil.forBuildData(sc, dataType, inputPath)

    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    //分类
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 10 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 6
    val maxBins = 100

    val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    val predictionAndLabels = testData.map {
      case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
    }

    if (numClasses == 2) {
      val metrics = new BinaryClassificationMetrics(predictionAndLabels)
      val auPR = metrics.areaUnderPR()
      val auROC = metrics.areaUnderROC()

      println("Area under ROC = " + auROC)
      println("Area under PR = " + auPR)
    }
    else if (numClasses > 2) {
      val metrics = new MulticlassMetrics(predictionAndLabels)
      val precision = metrics.precision
      val recall = metrics.recall

      println("precision = " + precision)
      println("recall = " + recall)
    }
    // Save and load model
    model.save(sc, modelPath)
  }
}
