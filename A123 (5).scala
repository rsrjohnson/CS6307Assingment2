// Databricks notebook source
//import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StopWordsRemover
import spark.implicits._
import scala.math._


// COMMAND ----------

// add the original document for the search return
val movies =sc.textFile("/FileStore/tables/plot_summaries.txt")
  .map(_.split("	"))
  .map(c => (c(0),c(1)))
  .toDF("id","description")


//separating id and plot, removing non alpha numeric characters except apostrophe, converting to lower case, wrapping all into a DF 
var movie_plots=sc.textFile("/FileStore/tables/plot_summaries.txt").map(line => line.split("\t")).map(x=>(x(0),x(1).replaceAll("[\\W&&[^']]"," ").split(" ").filter(word=>word.length>0).map(_.toLowerCase))).toDF("id","plot")


//Stop Words Remover
val remover = new StopWordsRemover()
  .setInputCol("plot")
  .setOutputCol("plot_filtered")

movie_plots=remover.transform(movie_plots).drop("plot") //removing stopwords and dropping not processed plots

val N=movie_plots.count //42306 films

//display(movie_plots)



// COMMAND ----------

// COMMAND ----------

val flattened = movie_plots.withColumn("token",explode($"plot_filtered")).drop("plot_filtered")

//finding term i frequencies by document j
val TF = flattened.groupBy("id", "token").count().toDF("id","token","tf_ij")


TF.show
//display(TF.filter($"token".contains("haymitch")))

// COMMAND ----------

//finding number of documents of every term
val DF = flattened.distinct().groupBy("token").count().toDF("token","df")

DF.show

// COMMAND ----------

//display(DF.filter($"token".contains("haym")))

// COMMAND ----------

//Euclidean Norm 
def norm(v:Seq[Double]):Double={
    
  sqrt((v).map { case (x) => pow(x, 2) }.sum)
} 

val normUDF = udf(norm _)


// COMMAND ----------

val logresult = udf((df:Long) =>log(42306/df))

val newdf = DF.withColumn("idf",logresult(col("df")))

//newdf.show

//join TF and newdf, calculate the tf-idf
val tf_idfDF = TF
      .join(newdf, Seq("token"))
      .withColumn("tf_idf", col("tf_ij") * col("idf")).select("id","token","tf_idf")


val tf_idfvectors=tf_idfDF.groupBy("id").agg(collect_list("tf_idf").as("tf_idf_vector"))
// create TF vector
val tf_vectors = TF.groupBy("id").agg(collect_list("tf_ij").as("tf_vector"))
// find the tf_ij norm of each document
val tf_norm = tf_vectors.withColumn("norm",normUDF(col("tf_vector")))
//finding the norm of each document
// tf_idfvectors.withColumn("norm",normUDF(col("tf_idf_vector"))).show

// COMMAND ----------

// COMMAND ----------

val terms = sc.textFile("/FileStore/tables/user_terms.txt").map(_.toLowerCase).map(x=>x.split(" "))

val flat_terms = terms.toDF("query").withColumn("token",explode($"query"))

//preparing the terms for cosine similarity
val queries_weights = flat_terms.groupBy("query", "token").count().toDF("query","token","weight")

// create queries_vector
val tf_vectors = TF.groupBy("id").agg(collect_list("tf_ij").as("tf_vector"))
val queries_vectors =queries_weights.groupBy("query").agg(collect_list("weight").as("queries_vector")) 
//queries_weights.show(30)

//finding norms of the queries
// val queries_norms=queries_weights.groupBy("query").agg(normUDF(collect_list(col("weight").cast("double"))).as("norm"))
val queries_norms = queries_vectors.withColumn("qnorm",normUDF(col("queries_vector")))
//display(queries_vectors)
display(queries_norms)


// COMMAND ----------

//join movies and tf_idfDF
val movies_tf_idfDF = movies
      .join(tf_idfDF, Seq("id"))
      .select("id","description","token","tf_idf")


def query_results(search:Array[String], datafr : org.apache.spark.sql.DataFrame):org.apache.spark.sql.DataFrame={
   
  if(search.length==1){
    //single term query
    val qDF = datafr.filter($"token"===search(0))   
    return qDF.orderBy(desc("tf_idf")).limit(10)    
  }
  else{
    //cosine similarity answer to be implemented
    return datafr
  }
  
}

// COMMAND ----------

// search one term
val answers=terms.collect().map(x=>query_results(x,movies_tf_idfDF))
display(answers(0))


// COMMAND ----------

//val movie_meta=sc.textFile("/FileStore/tables/movie_metadata.tsv")

val newColumns = Seq("Wikipedia movie ID","Freebase movie ID", "Movie name","Movie release date","Movie box office revenue","Movie runtime","Movie languages","Movie countries","Movie genres")


val movie_meta=spark.read.option("sep", "\t").option("inferSchema","true").csv("/FileStore/tables/movie_metadata.tsv").toDF(newColumns:_*)



display(movie_meta)

// COMMAND ----------

//answer to query number 8(index 7)
display(answers(7).join(movie_meta,answers(7).col("id") === movie_meta.col("Wikipedia movie ID")).select("id","Movie name","tf_idf").orderBy(desc("tf_idf")))


// COMMAND ----------

display(tf_idfDF)

// COMMAND ----------

display(movies_tf_idfDF)

// COMMAND ----------


def query_results12(search:Array[String], 
                    movies_tf_idfDF : org.apache.spark.sql.DataFrame, 
                    TF : org.apache.spark.sql.DataFrame,
                    tf_norm : org.apache.spark.sql.DataFrame,
                    queries_norms : org.apache.spark.sql.DataFrame
                   ):org.apache.spark.sql.DataFrame={
   
  if(search.length==1){
    //single term query
    val qDF = movies_tf_idfDF.filter($"token"===search(0))   
    return qDF.orderBy(desc("tf_idf")).limit(10)    
  }
  else{
    
    //cosine similarity answer to be implemented

    val tf_query_token = TF.where($"token" isin (search:_*))
    

    val tf_query_dot = tf_query_token.groupBy("id").agg(sum("tf_ij").as("dotproduct"))
    val query_norm = queries_norms.filter($"query"===search)

    val tf_query_norm = tf_query_dot
        .join(tf_norm, Seq("id"))
        .join(query_norm)
 
   val tf_query_cossim = tf_query_norm.withColumn("cossim",($"dotproduct")/(($"norm")*($"qnorm")))
   
    
    val topten = tf_query_cossim.orderBy(desc("cossim")).limit(10)
    val toptenmovies = movies_tf_idfDF.join(topten, Seq("id"))
           .drop("token","tf_idf","dotproduct","tf_vector","norm","queries_vector","qnorm").distinct()
        //return tf_query_dot.filter($"id"==="12596771")
    //return tf_query_token.filter($"id"==="12596771")
   // return tf_query_norm.filter($"id"==="12596771")
    return toptenmovies.orderBy(desc("cossim")).limit(10)    
  }
  
}


// COMMAND ----------

val answers=terms.collect().map(x=>query_results12(x,movies_tf_idfDF,TF,tf_norm, queries_norms))
display(answers(0))

// COMMAND ----------


