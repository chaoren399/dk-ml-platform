package com.dk.randomforest

import com.dakuai.util.DKUtil
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by Administrator on 2016/3/24 0024.
 */
object RFModelPredict {
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
    conf.setAppName("RFModelPredict")

    val sc = new SparkContext(conf)
    val model = RandomForestModel.load(sc, modelPath)

    println("toDebugString:\n" + model.toDebugString)

    val data = DKUtil.forPredictData(sc, dataType, inputPath)//无numfeature

    val predictionAndFeatures=data.map{
      features=>
        val prediction = model.predict(features)
        println(features+"\n---------->"+prediction)
        (prediction,features)
    }

    predictionAndFeatures.map(x=>Array(x._2,x._1).mkString("--")).saveAsTextFile(outputPath)
  }
}
