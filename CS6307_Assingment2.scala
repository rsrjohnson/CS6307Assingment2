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

//val final_words=words.map(_.toLowerCase).map(_.replaceAll("'",""))
//val words=named_entities.flatMap(line => line.split(",")).map(_.replaceAll("\\W", " ")).map(_.trim).filter(x=> x.length>0) 

val wordsCount=words.map(x => (x,1)).reduceByKey((x,y) => x+y) //finding words counts

// COMMAND ----------

wordsCount.sortBy(-_._2).collect() //reduce operation, displaying word count in descending order

// COMMAND ----------

|//Part 2

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

def arr_to_dict(words:Seq[String]):scala.collection.immutable.Map[String,Int]={
  
  return words.groupBy(identity).mapValues(_.length)
   
}
val arr_to_dictUDF = udf(arr_to_dict _)

//Dictionary of each word count per document to be used for cosine similarity 
val words_dict=movie_plots.withColumn("words_dict",arr_to_dictUDF(col("plot_filtered"))).drop("plot_filtered")

display(words_dict)

// COMMAND ----------

val flattened = movie_plots.withColumn("token",explode($"plot_filtered")).drop("plot_filtered")

//val lltokens = flattened.groupBy("token")

//finding term i frequencies by document j
val TF = flattened.groupBy("id", "token").count().toDF("id","token","tf_ij")

TF.show
//display(TF.filter($"token".contains("haymitch")))

//finding number of documents of every term
val DF = flattened.distinct().groupBy("token").count().toDF("token","df")

DF.show


//display(DF.filter($"token".contains("haym")))

// COMMAND ----------

val idfFunc = udf((df:Long) =>log(42306/df))

val newdf = DF.withColumn("idf",idfFunc(col("df")))

newdf.show(false)

//join TF and newdf, calculate the tf-idf
val tf_idfDF = TF
      .join(newdf, Seq("token"))
      .withColumn("tf_idf", col("tf_ij") * col("idf")).select("id","token","tf_idf")


tf_idfDF.show(false)


// COMMAND ----------

//Euclidean Norm 
def norm(v:Seq[Double]):Double={
    
  sqrt((v).map { case (x) => pow(x, 2) }.sum)
} 

val normUDF = udf(norm _)



//Euclidean Norm 
def normD(v:Map[String,Int] ):Double={
    
  sqrt((v.values).map { case (x) => pow(x, 2) }.sum)
} 

val normDUDF = udf(normD _)



// COMMAND ----------

//val tf_idfvectors=tf_idfDF.groupBy("id").agg(collect_list("tf_idf").as("tf_idf_vector"))

val tf_vectors=TF.groupBy("id").agg(collect_list(col("tf_ij").cast("double")).as("tf_vector"))
//finding the norm of each document
//tf_idfvectors.withColumn("norm",normUDF(col("tf_idf_vector"))).show

val document_norms = tf_vectors.withColumn("norm",normUDF(col("tf_vector"))).drop("tf_vector")

//document_norms.show(false)


val normDF=words_dict.join(document_norms,Seq("id"))

normDF.show

// COMMAND ----------

//val terms = sc.textFile("/FileStore/tables/user_terms.txt").map(_.toLowerCase).map(x=>x.split(" "))

var terms = sc.textFile("/FileStore/tables/user_terms.txt").map(_.toLowerCase).map(x=>x.split(" ")).toDF("query")

val remover2 = new StopWordsRemover()
  .setInputCol("query")
  .setOutputCol("query_filtered")

//removing stop words from queries
terms=remover2.transform(terms).drop("query") //removing stopwords and dropping not processed queries




terms.show(false)


// COMMAND ----------

terms=terms.withColumn("words_dict",arr_to_dictUDF(col("query_filtered")))



terms=terms.withColumn("norm",normDUDF(col("words_dict")))


// COMMAND ----------

def dot_prod(term_dic:Map[String,Int],doc_dict:Map[String,Int]):Double=
{
  var result=0
  
  for ((token,weight) <- term_dic){
    if(doc_dict.contains(token)){
      result=result+doc_dict(token)*weight
    }
  
  }
  return result
}

val dot_prodUDF = udf(dot_prod _)



// COMMAND ----------

def query_results(search:Array[String],  tf_idfDF: org.apache.spark.sql.DataFrame,normDF:org.apache.spark.sql.DataFrame):org.apache.spark.sql.DataFrame={
   
  val query_dict=search.groupBy(identity).mapValues(_.length)
  
  if(query_dict.size==1){
    
    //single term query
    val qDF = tf_idfDF.filter($"token"===query_dict.keys.toList(0) )   
    
    return qDF.orderBy(desc("tf_idf")).limit(10)    
  }
  else{    
      
  
    val ext_Norm=normDF.withColumn("query_dict",typedLit(query_dict))
    
    
    
    val query_norm=sqrt((query_dict.values).map { case (x) => pow(x, 2) }.sum)
    
    val qDF=ext_Norm.withColumn("cosine_sim",dot_prodUDF('query_dict,'words_dict)/('norm*query_norm)).select("id","cosine_sim")
    
    return qDF.orderBy(desc("cosine_sim")).limit(10)  
  }
  
}

// COMMAND ----------

//answer to the user terms, we use this answers to look for the movie name
val answers=terms.select($"query_filtered").as[Array[String]].rdd.collect.map(x=>query_results(x,tf_idfDF,normDF))

// COMMAND ----------

val flat_terms = terms.withColumn("token",explode($"query_filtered"))

//preparing the terms for cosine similarity
val queries_weights = flat_terms.groupBy("query_filtered", "token").count().toDF("query","token","weight")

queries_weights.show(30,false)

//finding norms of the queries
val queries_norms=queries_weights.groupBy("query").agg(normUDF(collect_list(col("weight").cast("double"))).as("norm"))

queries_norms.show(false)

// COMMAND ----------

//val movie_meta=sc.textFile("/FileStore/tables/movie_metadata.tsv")

val newColumns = Seq("Wikipedia movie ID","Freebase movie ID", "Movie name","Movie release date","Movie box office revenue","Movie runtime","Movie languages","Movie countries","Movie genres")


val movie_meta=spark.read.option("sep", "\t").option("inferSchema","true").csv("/FileStore/tables/movie_metadata.tsv").toDF(newColumns:_*)



display(movie_meta)

// COMMAND ----------

var k=0
//answer to all the queries
for(query<-terms.select("query_filtered").as[Array[String]].collect){
  println("Query: " + query.mkString(", "))
  if(query.length==1)
    answers(k).join(movie_meta,answers(k).col("id") === movie_meta.col("Wikipedia movie ID")).select("id","Movie name","tf_idf").orderBy(desc("tf_idf")).show(false)
  else
    answers(k).join(movie_meta,answers(k).col("id") === movie_meta.col("Wikipedia movie ID")).select("id","Movie name","cosine_sim").orderBy(desc("cosine_sim")).show(false)
  
  k=k+1
}

