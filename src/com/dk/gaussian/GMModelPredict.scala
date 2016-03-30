package com.dk.gaussian

import com.dk.util.DKUtil
import org.apache.hadoop.fs.Path
import org.apache.spark.mllib.clustering.GaussianMixtureModel
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vector}
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkContext, SparkConf}
import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods._

/**
 * Created by Administrator on 2016/3/24 0024.
 */
object GMModelPredict {
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
    conf.setAppName("GMModelPredict")

    val sc = new SparkContext(conf)

    val sqlContext = new SQLContext(sc)

    val dataFrame = sqlContext.parquetFile(new Path(modelPath, "data").toUri.toString)
    //    val dataArray = dataFrame.select("weight", "mu", "sigma").collect()
    //
    //    val (weights, gaussians) = dataArray.map {
    //      case Row(weight: Double, mu: Vector, sigma: Matrix) =>
    //        (weight, new MultivariateGaussian(mu, sigma))
    //    }.unzip

    val dataArray = dataFrame.select("weight", "mu", "numRows", "numCols", "sigma").collect()

    val (weights, gaussians) = dataArray.map {
      case Row(weight: Double, mu: Vector, numRows: Int, numCols: Int, sigma: String) =>
        (weight, new MultivariateGaussian(mu, Matrices.dense(numRows, numCols, sigma.split(",").map(x => x.toDouble))))
    }.unzip


    //    val dataArray = dataFrame.select("weight", "gaussians").collect()
    //
    //    val (weights, gaussians) = dataArray.map {
    //      case Row(weight: Double, gaussians:MultivariateGaussian) =>
    //        (weight, gaussians)
    //    }.unzip

    val model = new GaussianMixtureModel(weights.toArray, gaussians.toArray)
    println("----------")
    println(model.k)
    for (i <- 0 until model.k) {
      println("weight=%f\nmu=%s\nsigma=\n%s\n" format
        (model.weights(i), model.gaussians(i).mu, model.gaussians(i).sigma))
    }

    val data = DKUtil.forPredictData(sc, dataType, inputPath)

//    data.foreach(testDataLine => {
//      val predictedClusterIndex: Int = model.predict(testDataLine)
//
//      println("The data " + testDataLine.toString + " belongs to cluster " +
//        predictedClusterIndex)
//    })
    val predictedClusterIndex = model.predict(data)

    predictedClusterIndex.zip(data).foreach(println(_))
    predictedClusterIndex.zip(data).map(x=>Array(x._2,x._1).mkString("--")).saveAsTextFile(outputPath)
  }
}
