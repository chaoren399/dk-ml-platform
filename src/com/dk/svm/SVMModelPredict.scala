package com.dk.svm

import com.dakuai.util.DKUtil
import org.apache.spark.mllib.classification.{SVMModel, LogisticRegressionModel}
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by Administrator on 2016/3/25 0025.
 */
object SVMModelPredict {
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
    conf.setAppName("SVMModelPredict")

    val sc = new SparkContext(conf)
    val model = SVMModel.load(sc,modelPath)

    val data=DKUtil.forPredictData(sc,dataType,inputPath)

    val predictionAndFeatures=data.map{
      features=>
        val prediction = model.predict(features)
        println(features+"\n---------->"+prediction)
        (prediction,features)
    }

    predictionAndFeatures.map(x=>Array(x._2,x._1).mkString("--")).saveAsTextFile(outputPath)
  }
}
