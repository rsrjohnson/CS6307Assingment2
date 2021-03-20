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


//separating id and plot, removing non alpha numeric characters except apostrophe, converting to lower case, wrapping all into a DF 
var movie_plots=sc.textFile("/FileStore/tables/plot_summaries.txt").map(line => line.split("\t")).map(x=>(x(0),x(1).split("[\\W&&[^']]").filter(word=>word.length>0).map(_.toLowerCase))).toDF("id","plot")



//Stop Words Remover
val remover = new StopWordsRemover()
  .setInputCol("plot")
  .setOutputCol("plot_filtered")

movie_plots=remover.transform(movie_plots).drop("plot") //removing stopwords and dropping not processed plots


display(movie_plots)



// COMMAND ----------


