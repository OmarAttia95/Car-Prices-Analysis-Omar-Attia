import logging
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
    spark = SparkSession.builder \
        .appName("CarPrices-Prediction-Analysis") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    return spark

# Function to limit rows and create matplotlib plot
def generate_plot(test_data_path: str):
    try:
        logger.info(f"{datetime.utcnow()}: Generating Matplotlib plot...")
        
        # Load data into Pandas DataFrame (limit to 50 rows)
        spark = get_spark_session()
        test_df = spark.read.parquet(test_data_path)
        pandas_df = test_df.limit(50).toPandas()
        
        # Example plot (modify as per your data visualization needs)
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

# New task for Matplotlib plot
plot_task = PythonOperator(
    task_id='generate_prediction_plot',
    python_callable=generate_plot,
    op_args=['{{ task_instance.xcom_pull(task_ids="preprocess_data")[1] }}'],
    dag=dag,
)

# Set task dependencies
extract_and_cleanse_task >> preprocess_task >> train_task >> evaluate_task >> plot_task
