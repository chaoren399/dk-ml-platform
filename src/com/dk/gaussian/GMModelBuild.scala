package com.dk.gaussian

import com.dk.util.DKUtil
import com.dk.util.MM.{Gau, Data}
import org.apache.hadoop.fs.Path
import org.apache.spark.mllib.clustering.GaussianMixture
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.{SparkContext, SparkConf}
import org.json4s.jackson.JsonMethods._
import org.json4s.JsonDSL._
import org.apache.spark.sql.SQLContext

/**
 * Created by Administrator on 2016/3/24 0024.
 */
object GMModelBuild {
  def main(args: Array[String]) {
    if (args.length != 4) {
      print("-------------args error--------------")
      System.exit(0)
    }

    val inputPath = args(0)
    val modelPath = args(1)
    val dataType = args(2)
    val numClusters = args(3).toInt

    val conf = new SparkConf()
    conf.setAppName("GMModelBuild")

    val sc = new SparkContext(conf)

    val data = DKUtil.forPredictData(sc, dataType, inputPath)
    val gmm = new GaussianMixture().setK(numClusters).run(data)

    // output parameters of max-likelihood model
    for (i <- 0 until gmm.k) {
      println("weight=%f\nmu=%s\nsigma=\n%s\n" format
        (gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma))
    }


    // Save and load model
    //    gmm.save(sc, modelPath)
    //val sameModel = GaussianMixtureModel.load(sc, modelPath)
    val sqlContext = new SQLContext(sc)
    val formatVersionV1_0 = "1.0"
    val classNameV1_0 = "org.apache.spark.mllib.clustering.GaussianMixtureModel"

    import sqlContext.implicits._
    // Create JSON metadata.
    val metadata = compact(render
      (("class" -> classNameV1_0) ~ ("version" -> formatVersionV1_0) ~ ("k" -> gmm.weights.length)))
    sc.parallelize(Seq(metadata), 1).saveAsTextFile(new Path(modelPath, "metadata").toUri.toString)

    // Create Parquet data.
    val dataArray = Array.tabulate(gmm.weights.length) { i =>
      Data(gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma.numRows, gmm.gaussians(i).sigma.numCols, gmm.gaussians(i).sigma.toArray.mkString(","))
    }

    //    val dataArray = Array.tabulate(gmm.weights.length) { i =>
    //      Gau(gmm.weights(i), gmm.gaussians(i))
    //    }
    val dataRDD = sc.parallelize(dataArray, 1).toDF()
    dataRDD.saveAsParquetFile(new Path(modelPath, "data").toUri.toString)
  }

}
