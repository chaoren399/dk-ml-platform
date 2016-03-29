package com.dk.pca

import java.io.File

import com.dakuai.util.DKUtil
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrix, Vectors, Vector}
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import breeze.linalg.DenseMatrix
import breeze.linalg.csvwrite

/**
 * Created by Administrator on 2016/3/24 0024.
 */
object PCAModel {
  def main(args: Array[String]) {
    if (args.length != 4) {
      print("-------------args error--------------")
      System.exit(0)
    }

    val inputPath = args(0)
    val outputPath = args(1)
    val dataType = args(2)
    val k = args(3).toInt//主成分数目

    val conf = new SparkConf()
    conf.setAppName("PCACompute")

    val sc = new SparkContext(conf)

    val data = DKUtil.forPredictData(sc,dataType,inputPath)

    val matrix = new RowMatrix(data)
    val pc = matrix.computePrincipalComponents(k)

    val col=pc.numCols
    val row=pc.numRows

    println("pc.numRows = "+row)
    println("pc.numCols = "+col)
    println("pc.toString = "+pc.toString())

    val pcm = new DenseMatrix(row,col,pc.toArray)
    csvwrite(new File(outputPath),pcm)//写出到本地，目录必须存在


  }
}
