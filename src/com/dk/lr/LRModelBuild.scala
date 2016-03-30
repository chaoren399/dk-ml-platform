package com.dk.lr

import com.dk.util.DKUtil
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by Administrator on 2016/3/22 0022.
 */
object LRModelBuild {
  def main(args: Array[String]) {
    if (args.length != 4) {
      print("-------------args error--------------")
      System.exit(0)
    }

    val inputPath = args(0)
    val modelPath = args(1)
    val dataType = args(2)
    val numClass = args(3).toInt

    val conf = new SparkConf()
    conf.setAppName("LRModelBuild")

    val sc = new SparkContext(conf)

    val data = DKUtil.forBuildData(sc, dataType, inputPath)

    //    if (dataType == "LibSVM") {
    //        data= MLUtils.loadLibSVMFile(sc, inputPath)
    //      }
    //    else if (dataType == "LabeledPoints") {
    //        val text = sc.textFile(inputPath)
    //        data=text.map { line =>
    //          val parts = line.split(",")
    //          LabeledPoint(parts(0).toDouble, Vectors.dense(parts.tail.map(java.lang.Double.parseDouble)))
    //        }
    //      }


    // val data = MLUtils.loadLabeledPoints(sc,inputPath)


    val splits = data.randomSplit(Array(0.8, 0.2), 11l)
    val trainData = splits(0)
    val testData = splits(1)

    val model = new LogisticRegressionWithLBFGS().setNumClasses(numClass).run(trainData)

    val predictionAndLabels = testData.map {
      case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
    }

    if (numClass == 2) {
      val metrics = new BinaryClassificationMetrics(predictionAndLabels)
      val auPR = metrics.areaUnderPR()
      val auROC = metrics.areaUnderROC()

      println("Area under ROC = " + auROC)
      println("Area under PR = " + auPR)
    }
    else if (numClass > 2) {
      val metrics = new MulticlassMetrics(predictionAndLabels)
      val precision = metrics.precision
      val recall = metrics.recall

      println("precision = " + precision)
      println("recall = " + recall)
    }

    model.save(sc, modelPath)
  }
}
