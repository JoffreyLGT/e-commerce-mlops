from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
import datetime
import subprocess

dag = DAG(
    dag_id="generation_dataset",
    description="Il s'agit de pouvoir genérer ou recupérer un dataset",
    tags=["worspaces","airflow"],
    schedule_interval="@daily", #Planification quotidienne, on ajustera selon les besoin
    default_args={
        "owner": "e-commerce",
        "start_date": days_ago(0, minute=5),
    },
    catchup=False
)

def generate_dataset():
    """Cette fonction permet de  genérer ou recuperer un dataset !!"""

    #python -m scripts.create_datasets --train-size 0.01 --test-size 0.01 --input-dir "/workspaces/datascience/data/originals" --output-dir "/workspaces/airflow/data/datasets/sample"

    command = "python -m scripts.create_datasets --train-size 0.01 --test-size 0.01 --input-dir '/workspaces/datascience/data/originals' --output-dir '/workspaces/airflow/data/datasets/sample'"
    try:
        subprocess.check_output(command, shell=True)
        print("Success")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

    #print("success")

recup_task = PythonOperator(
    task_id="recup_dataset",
    python_callable=generate_dataset,
    dag=dag,
    trigger_rule="all_success",
)


