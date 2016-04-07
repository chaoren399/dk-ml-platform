package com.dk.util

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.linalg.Vectors
/**
 * Created by Administrator on 2016/4/6 0006.
 */
object Normaliz {
  def main(args: Array[String]) {
    if (args.length != 3) {
      print("-------------args error--------------")
      System.exit(0)
    }

    val inputPath = args(0)
    val outputPath = args(1)
    val label = args(2)

    val conf = new SparkConf()
    conf.setAppName("Normalizer")
    val sc = new SparkContext(conf)
    val nm = new Normalizer()
    if (label=="nolabeled"){
      val data = DKUtil.forPredictData(sc,"LabeledPoints",inputPath)

      nm.transform(data).map(x=>x.toArray.mkString(",")).saveAsTextFile(outputPath)
    }
    else if (label=="labeled"){
      val data=DKUtil.forBuildData(sc,"LabeledPoints",inputPath)

      data.map(x => Array(x.label, nm.transform(x.features).toArray.mkString(",")).mkString(",")).saveAsTextFile(outputPath)
    }


  }
}
