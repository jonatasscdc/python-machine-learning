from pyspark.ml.recommendation
import ALS 
from pyspark.sql import SparkSession 
from pyspark.ml.evaluation 
import RegressionEvaluator 

#Create a SparkSession
spark = SparkSession.builder.appName("RecommendationSystem").getOrCreate()

#Load the Data 
data = spark.read.csv("https://raw.githubusercontent.com/caserec/Datasets-for-Recommender-Systems/master/ml-100k/u.data", 
                     header=False, inferSchema=True, sep='\t')
data = data.selectExpr("_c0 as userId", "_c1 as movieId", "_c2 as rating", "_c3 as timestamp")

#Split the data into training and test sets
(training, test) = data.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
model = als.fit(training)

#Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)

#Generate top 10 movie recommendation for each user 
userRecs - model.recommendForAllUsers(10)

#Stop the SparkSession
spark.stop()

