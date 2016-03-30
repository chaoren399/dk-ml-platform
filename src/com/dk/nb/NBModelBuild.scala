package com.dk.nb

import com.dk.util.DKUtil
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by Administrator on 2016/3/25 0025.
 */
object NBModelBuild {
  def main(args: Array[String]) {
    if (args.length != 3) {
      print("-------------args error--------------")
      System.exit(0)
    }

    val inputPath = args(0)
    val modelPath = args(1)
    val dataType = args(2)

    val conf = new SparkConf()
    conf.setAppName("NBModelBuild")

    val sc = new SparkContext(conf)

    val data = DKUtil.forBuildData(sc, dataType, inputPath)
    val splits = data.randomSplit(Array(0.8, 0.2), 11l)
    val trainData = splits(0)
    val testData = splits(1)

    val model = NaiveBayes.train(trainData)

    val predictionAndLabels = testData.map {
      case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
    }

    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision
    val recall = metrics.recall

    println("precision = " + precision)
    println("recall = " + recall)

    model.save(sc, modelPath)
  }

}
