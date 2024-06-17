Car Prices Prediction Analysis with Apache Airflow and Apache Spark
This repository contains an Apache Airflow DAG (Directed Acyclic Graph) designed for predicting car prices using historical data. The pipeline integrates Apache Spark for data processing and machine learning model training, and Matplotlib for generating predictive analytics visualizations.

Features:
Data Extraction and Cleansing: Extracts car prices dataset from CSV, cleanses missing values, and stores processed data in Parquet format.

Data Preprocessing: Preprocesses the cleaned dataset, including feature engineering and splitting into training and testing datasets.

Machine Learning Model Training: Trains a RandomForestRegressor model to predict car prices based on selected features.

Model Evaluation: Evaluates the trained model's performance using RMSE (Root Mean Squared Error) metric on the test dataset.

Visualization: Generates a scatter plot using Matplotlib to visualize predicted vs actual selling prices of cars.

Technologies Used:
Apache Airflow
Apache Spark
Matplotlib
Python (PySpark)
How to Use:
Clone the Repository: git clone https://github.com/yourusername/your-repo.git

Install Dependencies: Ensure you have Apache Airflow, Apache Spark, and necessary Python libraries installed.

Configure Airflow: Set up Airflow according to your environment (e.g., local machine or cloud setup).

Run the DAG: Upload the DAG file (car_prices_prediction_dag.py) to your Airflow environment and trigger the DAG.

View Results: Monitor the Airflow UI to view DAG execution progress and check generated plots for predictive analytics insights.

Directory Structure:
car_prices_prediction_dag.py: Airflow DAG definition file.
requirements.txt: Python dependencies required for the project.
README.md: Instructions and description of the project.

Contributions are welcome! Feel free to fork this repository, make changes, and submit a pull request. Please adhere to the existing coding style and maintain clear commit messages.
