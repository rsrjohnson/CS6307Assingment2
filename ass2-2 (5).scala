
 // import the text to DF
 val movies =sc.textFile("/FileStore/tables/plot_summaries.txt")
           .map(_.split("	"))
           .map(c => (c(0),c(1)))
           .toDF("movieID","description")
 val terms = spark.read.option("header","false").csv("/FileStore/tables/user_terms.txt")
 movies.count()  // the total number of movieID is 42306=N, used for computer tf-idf weight

// COMMAND ----------
//  tokenize the description to words
import org.apache.spark.sql.functions.explode
import org.apache.spark.ml.feature.Tokenizer

val tokenizer = new Tokenizer().setInputCol("description").setOutputCol("Words")
val wordsData = tokenizer.transform(movies)


// COMMAND ----------

val flattened = wordsData.withColumn("token",explode($"Words")) // explode the df by token
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.sql.functions.length

val dropstop = flattened.select(flattened.col("*")).where(length(flattened.col("token")) >5 ) // remove the stop words by length <= 4

val dropstopnew = dropstop.withColumn("newtoken", regexp_replace(dropstop("token"), """[\p{Punct}&&[^.]]""", "")) // remove the punctuation

// COMMAND ----------

// calculate the TF
val TF = dropstopnew.groupBy("movieID", "newtoken").count().as("tf").toDF("movieID","newtoken","tf")

// COMMAND ----------

TF.orderBy($"tf".desc).show()

// COMMAND ----------

// calculate the DF
val dropdup = dropstopnew.distinct()  // keep only one distinct movieID-token 
val DF = dropdup.groupBy("newtoken").count().as("df").toDF("newtoken","df")

// COMMAND ----------

import org.apache.spark.sql.functions.countDistinct
import sqlContext.implicits._
import org.apache.spark.sql.functions._
// calculate the DF
val DF = dropstopnew.select("movieID","newtoken")
        .groupBy("newtoken")
        .agg(countDistinct("movieID")).as("df").toDF("newtoken","df")
DF.show()

// COMMAND ----------

TF.show()

// calculate idf
import scala.math._
import org.apache.spark.sql.functions.{col, udf}
val calidf = (df: Long) => {
  log10(42306)/log10(df)
}
spark.udf.register("calidf", calidf)
val calcIdfUdf = udf { df: Long => calidf(df) }
val newdf = DF.withColumn("idf", calcIdfUdf(col("df")))


// COMMAND ----------

newdf.show()

// COMMAND ----------

// join the TF and newdf, calculate the tf-idf

val tf_idf = TF
      .join(newdf, Seq("newtoken"), "left")
      .withColumn("tf_idf", col("tf") * col("idf"))

// COMMAND ----------

tf_idf.show()

// COMMAND ----------

tf_idf.orderBy($"tf_idf".desc).show()

// COMMAND ----------

