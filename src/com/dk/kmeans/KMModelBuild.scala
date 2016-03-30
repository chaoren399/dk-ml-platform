package com.dk.kmeans

import com.dk.util.DKUtil
import com.dk.util.MM.Cluster
import org.apache.hadoop.fs.Path
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}
import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods._
import org.json4s.JsonDSL._

/**
 * Created by Administrator on 2016/3/24 0024.
 */
object KMModelBuild {
  def main(args: Array[String]) {
    if (args.length != 4) {
      print("-------------args error--------------")
      System.exit(0)
    }

    val inputPath = args(0)
    val modelPath = args(1)
    val dataType  = args(2)
    val numClusters = args(3).toInt

    val conf = new SparkConf()
    conf.setAppName("RFModelBuild")

    val sc = new SparkContext(conf)

    val data = DKUtil.forPredictData(sc,dataType,inputPath)

    val numIterations = 20
    val clusters = KMeans.train(data, numClusters, numIterations)

    val WSSSE = clusters.computeCost(data)
    println("Within Set Sum of Squared Errors = " + WSSSE)
    println("Cluster Number:" + clusters.clusterCenters.length)

    var clusterIndex:Int = 0
    println("Cluster Centers Information Overview:")
    clusters.clusterCenters.foreach(
      x => {

        println("Center Point of Cluster " + clusterIndex + ":")

        println(x)
        clusterIndex += 1
      })

    data.collect().foreach(testDataLine => {
      val predictedClusterIndex:Int = clusters.predict(testDataLine)

      println("The data " + testDataLine.toString + " belongs to cluster " +
        predictedClusterIndex)
    })
    // Save model
    val sqlContext = new SQLContext(sc)
    val thisFormatVersion = "1.0"
    val thisClassName = "org.apache.spark.mllib.clustering.KMeansModel"

    import sqlContext.implicits._

    implicit val formats = DefaultFormats//json隐式参数

    val metadata = compact(render(("class" -> thisClassName) ~ ("version" -> thisFormatVersion) ~ ("k" -> clusters.k)))
    sc.parallelize(Seq(metadata), 1).saveAsTextFile(new Path(modelPath, "metadata").toUri.toString)

    val dataRDD = sc.parallelize(clusters.clusterCenters.zipWithIndex,1).map { case (point, id) =>
      Cluster(id, point)
    }.toDF()
    dataRDD.saveAsParquetFile(new Path(modelPath, "data").toUri.toString)

  }

}
