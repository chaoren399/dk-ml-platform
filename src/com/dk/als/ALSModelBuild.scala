package com.dk.als

import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by Administrator on 2016/3/25 0025.
 */
object ALSModelBuild {
  def main(args: Array[String]) {
    if (args.length != 4) {
      print("-------------args error--------------")
      System.exit(0)
    }

    val inputPath = args(0)
    val modelPath = args(1)
    val rank = args(2).toInt
    val numIterations = args(3).toInt

    val conf = new SparkConf()
    conf.setAppName("ALSModelBuild")

    val sc = new SparkContext(conf)

    val data = sc.textFile(inputPath)
    val ratings = data.map(_.split(',') match { case Array(user, item, rate) =>
      Rating(user.toInt, item.toInt, rate.toDouble)
    })

    val model = ALS.train(ratings, rank, numIterations, 0.01)

    val usersProducts = ratings.map { case Rating(user, product, rate) =>
      (user, product)
    }
    val predictions =
      model.predict(usersProducts).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }
    val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictions)

    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()
    println("Mean Squared Error = " + MSE)

    val predictedAndTrue = ratesAndPreds.map { case ((user, product), (actual, predicted)) => (actual, predicted) }
    val regressionMetrics = new RegressionMetrics(predictedAndTrue)
    println("Mean Squared Error = " + regressionMetrics.meanSquaredError)
    println("Root Mean Squared Error = " + regressionMetrics.rootMeanSquaredError)


//    ratesAndPreds.sortByKey().repartition(1).sortBy(_._1).map({
//      case ((user, product), (rate, pred)) => (user + "," + product + "," + rate + "," + pred)
//    }).saveAsTextFile("/tmp/result")

    model.save(sc, modelPath)

  }

}
