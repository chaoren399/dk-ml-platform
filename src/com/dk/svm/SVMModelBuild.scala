package com.dk.svm

import com.dakuai.util.DKUtil
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, BinaryClassificationMetrics}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by Administrator on 2016/3/25 0025.
 */
object SVMModelBuild {
  def main(args: Array[String]) {
    if (args.length != 3) {
      print("-------------args error--------------")
      System.exit(0)
    }

    val inputPath = args(0)
    val modelPath = args(1)
    val dataType = args(2)

    val conf = new SparkConf()
    conf.setAppName("SVMModelBuild")

    val sc = new SparkContext(conf)

    val data = DKUtil.forBuildData(sc, dataType, inputPath)
    val splits = data.randomSplit(Array(0.8, 0.2), 11l)
    val trainData = splits(0)
    val testData = splits(1)

    val numIterations = 100
    val model = SVMWithSGD.train(trainData, numIterations)

    val predictionAndLabels = testData.map {
      case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
    }

    //仅支持二分类
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    val auPR = metrics.areaUnderPR()
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC)
    println("Area under PR = " + auPR)

    model.save(sc, modelPath)

  }
}
