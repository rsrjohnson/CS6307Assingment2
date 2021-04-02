// Databricks notebook source
//CS6307_Assingment 2
//Team Members:Randy Suarez Rodes rxs179030 and Ping Chen pxc190026

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
  .setInputCols("document", "normal", "embeddings")
  .setOutputCol("ner")

val nerConverter = new NerConverter()
  .setInputCols("document", "normal", "ner")
  .setOutputCol("entities")

val finisher = new Finisher()
  .setInputCols("ner", "entities")
  .setIncludeMetadata(true)
  .setOutputAsArray(true)
  .setCleanAnnotations(false)
  .setAnnotationSplitSymbol("@")
  .setValueSplitSymbol("#")


//Setting up the pipeline
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

//Preprocessing data, removing special characters and replacing non alphanumeric characters with space
val book_processed=book_orig.map(line=>line.replaceAll("'`¨“”", "").replaceAll("\\W+", " ").trim)

val bookDF=book_processed.toDS.toDF( "text") //preparing data frame to be transformed

val result = pipeline.fit(bookDF).transform(bookDF) //pipeline is already pre-trained, we only need to fit pur data frame

val named_entities=result.select($"finished_entities").as[String].rdd //getting named entities and converting to rdd



// COMMAND ----------

//joining each line, replacing non alphanumeric characters with space, trimming and removing empty and single characters
val words=named_entities.flatMap(line => line.split(",")).map(_.replaceAll("\\W+", " ")).map(_.replaceAll("_","")).map(_.trim).filter(x=> x.length>1)

val wordsCount=words.map(x => (x,1)).reduceByKey((x,y) => x+y) //finding words counts

// COMMAND ----------

wordsCount.sortBy(-_._2).collect() //reduce operation, displaying word count in descending order

// COMMAND ----------

//Part 2

import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StopWordsRemover
import spark.implicits._
import scala.math._

//separating id and plot, changing non alpha numeric characters (except apostrophe) to " ", replacing apostrophe by "",  converting to lower case, wrapping all into a DF 
var movie_plots=sc.textFile("/FileStore/tables/plot_summaries.txt").map(line => line.split("\t")).map(x=>(x(0),x(1).replaceAll("[\\W&&[^']]"," ").replaceAll("'","").split(" ").filter(word=>word.length>1).map(_.toLowerCase))).toDF("id","plot")


//Stop Words Remover
val remover = new StopWordsRemover()
  .setInputCol("plot")
  .setOutputCol("plot_filtered")

movie_plots=remover.transform(movie_plots).drop("plot") //removing stopwords and dropping not processed plots

val N=movie_plots.count //42306 films


// COMMAND ----------

//Function to create a dictionary of word counts for each document
def arr_to_dict(words:Seq[String]):scala.collection.immutable.Map[String,Int]={
  
  return words.groupBy(identity).mapValues(_.length)
   
}

val arr_to_dictUDF = udf(arr_to_dict _)

//Dictionary of each word count per document to be used for cosine similarity 
val words_dict=movie_plots.withColumn("words_dict",arr_to_dictUDF(col("plot_filtered"))).drop("plot_filtered")

// COMMAND ----------

val flattened = movie_plots.withColumn("token",explode($"plot_filtered")).drop("plot_filtered")

//finding term i frequencies in document j
val TF = flattened.groupBy("id", "token").count().toDF("id","token","tf_ij")

//finding number of documents of every term
val DF = flattened.distinct().groupBy("token").count().toDF("token","df")

// COMMAND ----------

//Function to find a term idf
val idfFunc = udf((df:Long) =>log(N/df))

val newdf = DF.withColumn("idf",idfFunc(col("df")))

//Join TF and newdf to calculate the tf-idf
val tf_idfDF = TF
      .join(newdf, Seq("token"))
      .withColumn("tf_idf", col("tf_ij") * col("idf")).select("id","token","tf_idf")

// COMMAND ----------

//Fucntion to calculate Euclidean Norm for arrays
def norm(v:Seq[Double]):Double={
    
  sqrt((v).map { case (x) => pow(x, 2) }.sum)
} 

val normUDF = udf(norm _)

//Fucntion to calculate Euclidean Norm  for dictionary of counts
def normD(v:Map[String,Int] ):Double={
    
  sqrt((v.values).map { case (x) => pow(x, 2) }.sum)
} 

val normDUDF = udf(normD _)



// COMMAND ----------

//Finding the norms of each document
val tf_vectors=TF.groupBy("id").agg(collect_list(col("tf_ij").cast("double")).as("tf_vector"))
val document_norms = tf_vectors.withColumn("norm",normUDF(col("tf_vector"))).drop("tf_vector")

//Dataframe of norms
val normDF=words_dict.join(document_norms,Seq("id"))

// COMMAND ----------

//Original user's terms, stop words will be removed

//comedies
//action films
//most popular
//adventure films
//musicals
//dramas
//romantic films
//Horror
//Mystery
//Thriller
//Western
//Fantasy
//fiction
//action films with action stars
//funny movie with action scenes

var terms = sc.textFile("/FileStore/tables/user_terms.txt").map(_.toLowerCase).map(x=>x.split(" ")).toDF("query")

//removing stop words from queries
val remover2 = new StopWordsRemover()
  .setInputCol("query")
  .setOutputCol("query_filtered")

terms=remover2.transform(terms).drop("query") //removing stopwords and dropping not processed queries

// COMMAND ----------

//Function to find the dot product of two dictionaries of words
def dot_prod(term_dic:Map[String,Int],doc_dict:Map[String,Int]):Double=
{
  var result=0
  //only if the term is present, we find the product of the term frequency in the query times the term frequency in the document 
  for ((token,weight) <- term_dic){
    if(doc_dict.contains(token)){
      result=result+doc_dict(token)*weight
    }
  
  }
  return result
}

val dot_prodUDF = udf(dot_prod _)

// COMMAND ----------

//Function to find the top 10 movies based on the user's terms
def query_results(search:Array[String],  tf_idfDF: org.apache.spark.sql.DataFrame,normDF:org.apache.spark.sql.DataFrame):org.apache.spark.sql.DataFrame={
   
  //Creating a dictionary of word counts from the query
  val query_dict=search.groupBy(identity).mapValues(_.length)
  
  //Single term queries
  if(query_dict.size==1){
    
      val qDF = tf_idfDF.filter($"token"===query_dict.keys.toList(0) )   
    
    //returning tf_idf
    return qDF.orderBy(desc("tf_idf")).limit(10)    
  }
  else{ 
    
     //Norm of the query for cosine similarity
    val query_norm=sqrt((query_dict.values).map { case (x) => pow(x, 2) }.sum)
    
    //Adding the query dictionary to a column of the documents data frame    
    val ext_Norm=normDF.withColumn("query_dict",typedLit(query_dict))      
    
    //Finding the cosine similarity between the query and every document    
    val qDF=ext_Norm.withColumn("cosine_sim",dot_prodUDF('query_dict,'words_dict)/('norm*query_norm)).select("id","cosine_sim")
    
    return qDF.orderBy(desc("cosine_sim")).limit(10)  
  }
  
}

// COMMAND ----------

//Answers to the user's queries, later we use this answers to look for the movie name
val answers=terms.select($"query_filtered").as[Array[String]].rdd.collect.map(x=>query_results(x,tf_idfDF,normDF))

// COMMAND ----------

//Columns' names for the movie_metadata dataframe
val newColumns = Seq("Wikipedia movie ID","Freebase movie ID", "Movie name","Movie release date","Movie box office revenue","Movie runtime","Movie languages","Movie countries","Movie genres")

//Movie_metadata dataframe
val movie_meta=spark.read.option("sep", "\t").option("inferSchema","true").csv("/FileStore/tables/movie_metadata.tsv").toDF(newColumns:_*)

// COMMAND ----------

//Final answers to all the queries

var k=0
//We join the top documents id of each query with the movie id
for(query<-terms.select("query_filtered").as[Array[String]].collect){
  println("Query: " + query.mkString(", "))
  if(query.length==1)
    answers(k).join(movie_meta,answers(k).col("id") === movie_meta.col("Wikipedia movie ID")).select("id","Movie name","tf_idf").orderBy(desc("tf_idf")).show(false)
  else
    answers(k).join(movie_meta,answers(k).col("id") === movie_meta.col("Wikipedia movie ID")).select("id","Movie name","cosine_sim").orderBy(desc("cosine_sim")).show(false)
  
  k=k+1
}

