package com.dk.kmeans

import com.dk.util.MM.Cluster
import com.dk.util.DKUtil
import org.apache.hadoop.fs.Path
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}
import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods._

/**
 * Created by Administrator on 2016/3/24 0024.
 */
object KMModelPredict {
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
    conf.setAppName("KMModelPredict")

    val sc = new SparkContext(conf)

    val thisFormatVersion = "1.0"
    val thisClassName = "org.apache.spark.mllib.clustering.KMeansModel"
    val sqlContext = new SQLContext(sc)

    implicit val formats = DefaultFormats

    val metadata = parse(sc.textFile(new Path(modelPath, "metadata").toUri.toString).first())
    val className = (metadata \ "class").extract[String]
    val formatVersion = (metadata \ "version").extract[String]
    val k = (metadata \ "k").extract[Int]
    assert(className == thisClassName)
    assert(formatVersion == thisFormatVersion)

    val centroids = sqlContext.parquetFile(new Path(modelPath, "data").toUri.toString)
    //Loader.checkSchema[Cluster](centroids.schema)
    val localCentroids = centroids.map(Cluster.apply).collect()
    assert(k == localCentroids.size)

    val model = new KMeansModel(localCentroids.sortBy(_.id).map(_.point))

    val data = DKUtil.forPredictData(sc, dataType, inputPath)

    //    data.collect().foreach(testDataLine => {
    //      val predictedClusterIndex: Int = model.predict(testDataLine)
    //
    //      println("The data " + testDataLine.toString + " belongs to cluster " +
    //        predictedClusterIndex)
    //    })

    val dataAndPre = data.map(testDataLine => {
      val predictedClusterIndex: Int = model.predict(testDataLine)

      (testDataLine.toString, predictedClusterIndex)
    })
    dataAndPre.map(x => Array(x._1, x._2).mkString("--")).saveAsTextFile(outputPath)
  }
}
