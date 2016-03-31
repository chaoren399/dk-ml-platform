package com.dk.randomforest

import com.dk.util.DKUtil
import org.apache.spark.mllib.evaluation.{RegressionMetrics, MulticlassMetrics}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by Administrator on 2016/3/23 0023.
 */
object RFRegresModelBuild {
  def main(args: Array[String]) {
    if (args.length != 3) {
      print("-------------args error--------------")
      System.exit(0)
    }

    val inputPath = args(0)
    val modelPath = args(1)
    val dataType = args(2)


    val conf = new SparkConf()
    conf.setAppName("RFRegresModelBuild")

    val sc = new SparkContext(conf)

    val data = DKUtil.forBuildData(sc, dataType, inputPath).cache()

    val splits = data.randomSplit(Array(0.8, 0.2))
    val (trainingData, testData) = (splits(0), splits(1))

    //回归
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 10 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "variance"
    val maxDepth = 6
    val maxBins = 100
    val model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    val predictionAndLabels = testData.map {
      case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
    }

    val metrics = new RegressionMetrics(predictionAndLabels)
    val mae = metrics.meanAbsoluteError
    val mse = metrics.meanSquaredError
    val rmse = metrics.rootMeanSquaredError
    val r2 = metrics.r2

    println("mean absolute error = " + mae)
    println("mean squared error = " + mse)
    println("root mean squared error = " + rmse)
    println("coefficient of determination = " + r2)
    // Save and load model
    model.save(sc, modelPath)
  }
}
