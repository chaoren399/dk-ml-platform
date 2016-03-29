package com.dk.als

import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.{SparkContext, SparkConf}

import scala.collection.mutable

/**
 * Created by Administrator on 2016/3/25 0025.
 */
object RMProducts {
  def main(args: Array[String]) {
    if (args.length != 3) {
      print("-------------args error--------------")
      System.exit(0)
    }

    val inputPath = args(0)
    val modelPath = args(1)
    val outputPath = args(2)

    val conf = new SparkConf()
    conf.setAppName("RMProducts")

    val sc = new SparkContext(conf)
    val model = MatrixFactorizationModel.load(sc, modelPath)

    val data = sc.textFile(inputPath).distinct().collect()

    val rawData = new mutable.HashMap[String, String]()
    data.foreach(
      user => {
        // 依次为用户推荐商品
        val rs = model.recommendProducts(user.toInt, 10)
        var value = ""
        var key = 0
        // 拼接推荐结果
        rs.foreach(r => {
          key = r.product
          value = value + r.user + ":" + r.rating + ","
        })
        println("用户--推荐的商品：分值, ······")
        println(key+"--"+value)
        rawData.put(key.toString, value.substring(0, value.length - 1))
      }
    )

    val rs = sc.parallelize(rawData.toSeq).map(x=>Array(x._1,x._2).mkString("--"))
    rs.saveAsTextFile(outputPath)
  }
}
