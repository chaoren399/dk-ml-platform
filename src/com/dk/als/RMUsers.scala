package com.dk.als

import breeze.linalg.convert
import org.apache.spark.mllib.recommendation.{Rating, MatrixFactorizationModel}
import org.apache.spark.{SparkContext, SparkConf}

import scala.collection.mutable

/**
 * Created by Administrator on 2016/3/25 0025.
 */
object RMUsers {
  def main(args: Array[String]) {
    if (args.length != 3) {
      print("-------------args error--------------")
      System.exit(0)
    }

    val inputPath = args(0)
    val modelPath = args(1)
    val outputPath = args(2)

    val conf = new SparkConf()
    conf.setAppName("ALSRMUsers")

    val sc = new SparkContext(conf)
    val model = MatrixFactorizationModel.load(sc, modelPath)

    val data = sc.textFile(inputPath).distinct().map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#"))).collect()

    val rawData = new mutable.HashMap[String, String]()
    data.foreach(
      product => {
        // 依次为商品推荐用户
        val rs = model.recommendUsers(product.toInt, 10)
        var value = ""
        var key = 0
        // 拼接推荐结果
        rs.foreach(r => {
          key = r.product
          value = value + r.user + ":" + r.rating + ","
        })
        println("商品--推荐的用户：分值, ······")
        println(key+"--"+value)
        rawData.put(key.toString, value.substring(0, value.length - 1))
      }
    )

    val rs = sc.parallelize(rawData.toSeq).map(x=>Array(x._1,x._2).mkString("--"))
    rs.saveAsTextFile(outputPath)
  }
}
