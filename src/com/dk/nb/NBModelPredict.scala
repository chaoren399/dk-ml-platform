package com.dk.nb

import com.dk.util.DKUtil
import org.apache.spark.mllib.classification. NaiveBayesModel
import org.apache.spark.{SparkContext, SparkConf}


/**
 * Created by Administrator on 2016/3/25 0025.
 */
object NBModelPredict {
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
    conf.setAppName("NBModelPredict")

    val sc = new SparkContext(conf)
    val model = NaiveBayesModel.load(sc, modelPath)

    val numFeatures = DKUtil.getNumFeatures(sc,modelPath)
    //println("numFeatures = " + numFeatures)

    val data = DKUtil.forPredictData(sc, dataType, numFeatures, inputPath)

    val predictionAndFeatures = data.map {
      features =>
        val prediction = model.predict(features)
        println(features + "\n---------->" + prediction)
        (prediction, features)
    }

    predictionAndFeatures.map(x=>Array(x._2,x._1).mkString("--")).saveAsTextFile(outputPath)
  }

}
