package com.dk.lr

import com.dakuai.util.DKUtil
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by Administrator on 2016/3/22 0022.
 */
object LRModelPredict {
  def main(args: Array[String]) {
    if (args.length != 4) {
      print("-------------args error--------------")
      System.exit(0)
    }

    val inputPath = args(0)
    val modelPath = args(1)
    val outputPath = args(2)
    val dataType = args(3)

    val conf = new SparkConf()
    conf.setAppName("LRModelPredict")

    val sc = new SparkContext(conf)
    val model = LogisticRegressionModel.load(sc,modelPath)
    val numFeatures=model.numFeatures

    val data=DKUtil.forPredictData(sc,dataType,numFeatures,inputPath)

    val predictionAndFeatures=data.map{
      features=>
        val prediction = model.predict(features)
        println(features+"\n---------->"+prediction)
        (prediction,features)
    }

    predictionAndFeatures.map(x=>Array(x._2,x._1).mkString("--")).saveAsTextFile(outputPath)
  }
}
