// Databricks notebook source
//Part 1

//Packages and libraries
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.ner.NerConverter
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import com.johnsnowlabs.nlp.SparkNLP

//Preparing the annotation methods
val document = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
    .setInputCols(Array("document"))
    .setOutputCol("sentence")

val token = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val normalizer = new Normalizer()
  .setInputCols("token")
  .setOutputCol("normal")

val embeddings = WordEmbeddingsModel.pretrained()
   .setOutputCol("embeddings")

val ner = NerDLModel.pretrained()
  .setInputCols("document", "token", "embeddings")
  .setOutputCol("ner")

val nerConverter = new NerConverter()
  .setInputCols("document", "token", "ner")
  .setOutputCol("entities")

val finisher = new Finisher()
  .setInputCols("ner", "entities")
  .setIncludeMetadata(true)
  .setOutputAsArray(true)
  .setCleanAnnotations(false)
  .setAnnotationSplitSymbol("@")
  .setValueSplitSymbol("#")


//Setting up pipeline
val pipeline = new Pipeline().setStages(
  Array(
 document, 
    sentence,
    token, 
    normalizer,
    embeddings,
    ner, 
    nerConverter,
    finisher
  )
)

// COMMAND ----------

val book_orig = sc.textFile("/FileStore/tables/sherlock_holmes.txt").filter(row => !row.isEmpty) //ignoring empty lines

val bookDF=book_orig.toDS.toDF( "text") //preparing data frame to be transformed

val result = pipeline.fit(bookDF).transform(bookDF) //pipeline is already pre-trained, we only need to fit

val named_entities=result.select($"finished_entities").as[String].rdd // getting named entities and converting to rdd

// COMMAND ----------

val words=named_entities.flatMap(line => line.split(",")).map(_.replaceAll("[\\W&&[^']]", " ")).map(_.trim).filter(x=> x.length>0) //joining each line, removing selected non alphanumeric characters, trimming and removing empty characters

//val words=named_entities.flatMap(line => line.split(",")).map(_.replaceAll("\\W", " ")).map(_.trim).filter(x=> x.length>0) 

val wordsCount=words.map(x => (x,1)).reduceByKey((x,y) => x+y) //finding words counts

// COMMAND ----------

wordsCount.sortBy(-_._2).collect() //reduce operation, displaying word count in descending order

// COMMAND ----------

//Part 2

//import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StopWordsRemover
import spark.implicits._
import scala.math._



//separating id and plot, removing non alpha numeric characters except apostrophe, converting to lower case, wrapping all into a DF 
var movie_plots=sc.textFile("/FileStore/tables/plot_summaries.txt").map(line => line.split("\t")).map(x=>(x(0),x(1).replaceAll("[\\W&&[^']]"," ").split(" ").filter(word=>word.length>0).map(_.toLowerCase))).toDF("id","plot")


//Stop Words Remover
val remover = new StopWordsRemover()
  .setInputCol("plot")
  .setOutputCol("plot_filtered")

movie_plots=remover.transform(movie_plots).drop("plot") //removing stopwords and dropping not processed plots

val N=movie_plots.count //42306 films

display(movie_plots)



// COMMAND ----------

val flattened = movie_plots.withColumn("token",explode($"plot_filtered")).drop("plot_filtered")

//finding term i frequencies by document j
val TF = flattened.groupBy("id", "token").count().toDF("id","token","tf_ij")

TF.show
//display(TF.filter($"token".contains("haymitch")))

//finding number of documents of every term
val DF = flattened.distinct().groupBy("token").count().toDF("token","df")

DF.show


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



// COMMAND ----------

val tf_idfvectors=tf_idfDF.groupBy("id").agg(collect_list("tf_idf").as("tf_idf_vector"))

//finding the norm of each document
tf_idfvectors.withColumn("norm",normUDF(col("tf_idf_vector"))).show

// COMMAND ----------

val terms = sc.textFile("/FileStore/tables/user_terms.txt").map(_.toLowerCase).map(x=>x.split(" "))

val flat_terms = terms.toDF("query").withColumn("token",explode($"query"))

//preparing the terms for cosine similarity
val queries_weights = flat_terms.groupBy("query", "token").count().toDF("query","token","weight")

//queries_weights.show(30)

//finding norms of the queries
val queries_norms=queries_weights.groupBy("query").agg(normUDF(collect_list(col("weight").cast("double"))).as("norm"))

display(queries_norms)




// COMMAND ----------

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

val answers=terms.collect().map(x=>query_results(x,tf_idfDF))

display(answers(7))

// COMMAND ----------

//val movie_meta=sc.textFile("/FileStore/tables/movie_metadata.tsv")

val newColumns = Seq("Wikipedia movie ID","Freebase movie ID", "Movie name","Movie release date","Movie box office revenue","Movie runtime","Movie languages","Movie countries","Movie genres")


val movie_meta=spark.read.option("sep", "\t").option("inferSchema","true").csv("/FileStore/tables/movie_metadata.tsv").toDF(newColumns:_*)



display(movie_meta)

// COMMAND ----------

//answer to query number 8(index 7)
display(answers(7).join(movie_meta,answers(7).col("id") === movie_meta.col("Wikipedia movie ID")).select("id","Movie name","tf_idf").orderBy(desc("tf_idf")))



// COMMAND ----------


