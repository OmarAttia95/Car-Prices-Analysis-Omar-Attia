import logging
import os
import shutil
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to create or get SparkSession
def get_spark_session():
    spark = SparkSession.builder \
        .appName("CarPrices-Prediction-Analysis") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    return spark


def extract_and_cleanse_data():
    """
    Extracts and cleanses the car prices dataset from a CSV file,
    and writes the processed data to Parquet format for further processing.
    """
    try:
        logger.info(f"{datetime.utcnow()}: Starting data extraction and cleansing...")
        
        spark = get_spark_session()
        
        # Load the raw dataset from CSV
        schema = "year INT, make STRING, model STRING, trim STRING, body STRING, transmission STRING, " \
                 "vin STRING, state STRING, condition INT, odometer INT, color STRING, interior STRING, " \
                 "seller STRING, mmr INT, sellingprice INT, saledate STRING"
        car_prices_df = spark.read.csv("/home/omarattia95/airflow/worksets/car_prices.csv", 
                                       header=True, schema=schema)
        
        # Drop rows with missing values
        car_prices_df = car_prices_df.dropna()
        
        # Write cleaned data to Parquet format for further use
        cleaned_data_path = '/home/omarattia95/airflow/worksets/car_prices_data.parquet'
        car_prices_df.write.mode('overwrite').parquet(cleaned_data_path)
        
        logger.info(f"{datetime.utcnow()}: Data extraction and cleansing complete.")
        
        return cleaned_data_path
    
    except Exception as e:
        logger.error(f"Error in extract_and_cleanse_data: {str(e)}")
        raise

def preprocess_data(cleaned_data_path: str):
    """
    Preprocesses the extracted car prices data from Parquet,
    including assembling feature vectors and splitting into training and testing datasets.
    """
    try:
        logger.info(f"{datetime.utcnow()}: Starting data preprocessing...")
        
        spark = get_spark_session()
        
        # Load the cleaned dataset from Parquet
        car_prices_df = spark.read.parquet(cleaned_data_path)
        
        # Example preprocessing steps (modify as per ML requirements)
        assembler = VectorAssembler(
            inputCols=["year", "condition", "odometer"],
            outputCol="features"
        )
        feature_df = assembler.transform(car_prices_df)
        
        # Split data into training and testing sets
        train_df, test_df = feature_df.randomSplit([0.8, 0.2], seed=42)
        
        # Write preprocessed datasets to Parquet format for further use
        train_data_path = '/home/omarattia95/airflow/worksets/car_prices_train.parquet'
        test_data_path = '/home/omarattia95/airflow/worksets/car_prices_test.parquet'
        train_df.write.mode('overwrite').parquet(train_data_path)
        test_df.write.mode('overwrite').parquet(test_data_path)
        
        logger.info(f"{datetime.utcnow()}: Data preprocessing complete.")
        
        return train_data_path, test_data_path
    
    except Exception as e:
        logger.error(f"Error in preprocess_data: {str(e)}")
        raise

def train_machine_learning_model(train_data_path: str) -> str:
    """
    Trains a machine learning model using the preprocessed training data
    to predict car prices based on selected features,
    and saves the trained model for future predictions.
    """
    try:
        logger.info(f"{datetime.utcnow()}: Starting model training...")
        
        spark = get_spark_session()
        
        # Load the preprocessed training dataset from Parquet
        train_df = spark.read.parquet(train_data_path)
        
        # Define and train your preferred ML model (e.g., RandomForestRegressor)
        rf = RandomForestRegressor(featuresCol="features", labelCol="sellingprice", numTrees=10)
        rf_model = rf.fit(train_df)
        
        # Save the trained model
        model_path = '/home/omarattia95/airflow/worksets/car_prices_rf_model'
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        rf_model.save(model_path)
        
        logger.info(f"{datetime.utcnow()}: Model training complete.")
        
        return model_path
    
    except Exception as e:
        logger.error(f"Error in train_machine_learning_model: {str(e)}")
        raise

def evaluate_machine_learning_model(test_data_path: str, model_path: str):
    try:
        logger.info(f"{datetime.utcnow()}: Starting model evaluation...")
        
        spark = get_spark_session()
        
        # Load the preprocessed testing dataset from Parquet
        test_df = spark.read.parquet(test_data_path)
        
        # Load the trained Random Forest regression model
        from pyspark.ml.regression import RandomForestRegressionModel
        rf_model = RandomForestRegressionModel.load(model_path)
        
        # Make predictions using the trained model
        predictions = rf_model.transform(test_df)
        
        # Print or log the schema of predictions DataFrame for debugging
        predictions.printSchema()
        
        # Verify the presence of required columns
        required_columns = ['sellingprice', 'prediction']
        for col in required_columns:
            if col not in predictions.columns:
                raise ValueError(f"Column '{col}' does not exist in predictions DataFrame.")
        
        # Evaluate predictions using RMSE metric
        evaluator = RegressionEvaluator(labelCol="sellingprice", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        
        # Log the RMSE metric for model evaluation
        logger.info(f"{datetime.utcnow()}: Root Mean Squared Error (RMSE): {rmse}")
        logger.info(f"{datetime.utcnow()}: Model evaluation complete.")
        
    except Exception as e:
        logger.error(f"Error in evaluate_machine_learning_model: {str(e)}")
        raise

# Define the default arguments for the DAG
default_args = {
    'owner': 'Omar Attia',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Instantiate the DAG
dag = DAG(
    dag_id='CarPrices-Prediction-Analysis',
    default_args=default_args,
    description='A DAG for predicting car prices based on historical data',
    schedule_interval='@daily',
    start_date=datetime(2024, 6, 15),  # Adjust as per your requirement
    catchup=False,
)

# Define the tasks using PythonOperator
extract_and_cleanse_task = PythonOperator(
    task_id='extract_and_cleanse_data',
    python_callable=extract_and_cleanse_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    op_args=['{{ task_instance.xcom_pull(task_ids="extract_and_cleanse_data") }}'],
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_machine_learning_model',
    python_callable=train_machine_learning_model,
    op_args=['{{ task_instance.xcom_pull(task_ids="preprocess_data")[0] }}'],
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id='evaluate_machine_learning_model',
    python_callable=evaluate_machine_learning_model,
    op_args=['{{ task_instance.xcom_pull(task_ids="preprocess_data")[1] }}',
             '{{ task_instance.xcom_pull(task_ids="train_machine_learning_model") }}'],
    dag=dag,
)

# Set task dependencies
extract_and_cleanse_task >> preprocess_task >> train_task >> evaluate_task
