package com.dk.fpgrowth

import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by Administrator on 2016/3/25 0025.
 */
object FPGrowthModel {
  def main(args: Array[String]) {
    if (args.length != 3) {
      print("-------------args error--------------")
      System.exit(0)
    }

    val inputPath = args(0)
    val outputPath = args(1)
    val minSupport = args(2).toDouble//defalut:0.3

    val conf = new SparkConf()
    conf.setAppName("FPGrowthModelBuild")

    val sc = new SparkContext(conf)

    val data = sc.textFile(inputPath).map(_.split(" ")).cache()

    val model = new FPGrowth().setMinSupport(minSupport).run(data)

    println(s"Number of frequent itemsets: ${model.freqItemsets.count()}")

    model.freqItemsets.foreach { itemset =>
      println(itemset.items.mkString("[", ",", "]") + ": " + itemset.freq)
    }
    //model.freqItemsets.saveAsTextFile(outputPath)
    model.freqItemsets
      .map { itemset => (itemset.items.mkString("[", ",", "]"), itemset.freq) }
      .sortBy(_._2,false,1).map(x=>Array(x._1,x._2).mkString(": ")).saveAsTextFile(outputPath)

  }
}
