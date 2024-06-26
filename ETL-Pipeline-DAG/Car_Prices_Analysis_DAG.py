import logging
import os
import shutil
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to create or get SparkSession
def get_spark_session():
    """
    Establishes or retrieves the SparkSession for data processing.
    """
    spark = SparkSession.builder \
        .appName("CarPrices-Prediction-Analysis") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    return spark

# Function to extract and cleanse data
def extract_and_cleanse_data():
    """
    Extracts car prices data from a CSV file, performs cleansing,
    and saves the processed data to Parquet format.
    """
    try:
        logger.info(f"{datetime.utcnow()}: Initiating data extraction and cleansing...")
        
        spark = get_spark_session()
        
        # Replace with your actual CSV file path or utilize Airflow Variable
        csv_file_path = Variable.get("car_prices_csv_path")
        schema = "year INT, make STRING, model STRING, trim STRING, body STRING, transmission STRING, " \
                 "vin STRING, state STRING, condition INT, odometer INT, color STRING, interior STRING, " \
                 "seller STRING, mmr INT, sellingprice INT, saledate STRING"
        
        car_prices_df = spark.read.csv(csv_file_path, header=True, schema=schema)
        
        # Remove rows with missing values
        car_prices_df = car_prices_df.dropna()
        
        # Save cleansed data to Parquet format
        cleaned_data_path = '/tmp/airflow/worksets/car_prices_data.parquet'
        car_prices_df.write.mode('overwrite').parquet(cleaned_data_path)
        
        logger.info(f"{datetime.utcnow()}: Data extraction and cleansing complete.")
        
        return cleaned_data_path
    
    except Exception as e:
        logger.error(f"Error in extract_and_cleanse_data: {str(e)}")
        raise

# Function to preprocess data
def preprocess_data(cleaned_data_path: str):
    """
    Preprocesses car prices data from Parquet format,
    including feature vector assembly and splitting into training/testing sets.
    """
    try:
        logger.info(f"{datetime.utcnow()}: Initiating data preprocessing...")
        
        spark = get_spark_session()
        
        # Load cleaned data from Parquet
        car_prices_df = spark.read.parquet(cleaned_data_path)
        
        # Example preprocessing steps (customize as per ML requirements)
        assembler = VectorAssembler(
            inputCols=["year", "condition", "odometer"],
            outputCol="features"
        )
        feature_df = assembler.transform(car_prices_df)
        
        # Split data into training and testing sets
        train_df, test_df = feature_df.randomSplit([0.8, 0.2], seed=42)
        
        # Save preprocessed datasets to Parquet format
        train_data_path = '/tmp/airflow/worksets/car_prices_train.parquet'
        test_data_path = '/tmp/airflow/worksets/car_prices_test.parquet'
        train_df.write.mode('overwrite').parquet(train_data_path)
        test_df.write.mode('overwrite').parquet(test_data_path)
        
        logger.info(f"{datetime.utcnow()}: Data preprocessing complete.")
        
        return train_data_path, test_data_path
    
    except Exception as e:
        logger.error(f"Error in preprocess_data: {str(e)}")
        raise

# Function to train machine learning model
def train_machine_learning_model(train_data_path: str) -> str:
    """
    Trains a machine learning model using preprocessed training data,
    predicts car prices, and saves the trained model.
    """
    try:
        logger.info(f"{datetime.utcnow()}: Initiating model training...")
        
        spark = get_spark_session()
        
        # Load preprocessed training dataset
        train_df = spark.read.parquet(train_data_path)
        
        # Define and train the ML model (e.g., RandomForestRegressor)
        rf = RandomForestRegressor(featuresCol="features", labelCol="sellingprice", numTrees=10)
        rf_model = rf.fit(train_df)
        
        # Save the trained model
        model_path = '/tmp/airflow/worksets/car_prices_rf_model'
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        rf_model.save(model_path)
        
        logger.info(f"{datetime.utcnow()}: Model training complete.")
        
        return model_path
    
    except Exception as e:
        logger.error(f"Error in train_machine_learning_model: {str(e)}")
        raise

# Function to evaluate machine learning model
def evaluate_machine_learning_model(test_data_path: str, model_path: str):
    try:
        logger.info(f"{datetime.utcnow()}: Initiating model evaluation...")
        
        spark = get_spark_session()
        
        # Load preprocessed testing dataset
        test_df = spark.read.parquet(test_data_path)
        
        # Load trained Random Forest regression model
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

# Function to generate Matplotlib plot
def generate_plot(test_data_path: str):
    try:
        logger.info(f"{datetime.utcnow()}: Generating Matplotlib plot...")
        
        # Load data into Pandas DataFrame (limit to 50 rows for clean plot)
        spark = get_spark_session()
        test_df = spark.read.parquet(test_data_path)
        pandas_df = test_df.limit(50).toPandas()
        
        # Example plot (customize for specific data visualization needs)
        plt.figure(figsize=(10, 6))
        plt.scatter(pandas_df['sellingprice'], pandas_df['prediction'], alpha=0.5)
        plt.title('Predicted vs Actual Selling Price')
        plt.xlabel('Actual Selling Price')
        plt.ylabel('Predicted Selling Price')
        plt.grid(True)
        
        # Get output path from Airflow Variable (or use a default if not set)
        output_path = Variable.get("plot_output_path", default_var="/tmp/airflow/plots/")
        
        # Save the plot
        plot_file = f"{output_path}/prediction_plot.png"
        plt.savefig(plot_file)
        
        logger.info(f"{datetime.utcnow()}: Matplotlib plot saved at {plot_file}")
        
    except Exception as e:
        logger.error(f"Error in generate_plot: {str(e)}")
        raise

# Define default arguments for the DAG
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

# Define tasks using PythonOperator
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

plot_task = PythonOperator(
    task_id='generate_prediction_plot',
    python_callable=generate_plot,
    op_args=['{{ task_instance.xcom_pull(task_ids="preprocess_data")[1] }}'],
    dag=dag,
)

# Set task dependencies
extract_and_cleanse_task >> preprocess_task >> train_task >> evaluate_task >> plot_task
