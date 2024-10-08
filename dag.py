from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.sensors.filesystem import FileSensor

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
from sqlalchemy import create_engine

from model import CNNModel, load_model, train_and_evaluate_model, evaluate_model
from data_processing import preprocess_image_cv
from database import init_db, save_prediction


# 데이터베이스 연결 설정
DB_CONNECTION = "/Users/alookso/1007MLpj/ml-project-ml-pjt-7/database.db"
engine = create_engine(DB_CONNECTION)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 7),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG 정의
dag = DAG(
    'mnist_model_workflow',
    default_args=default_args,
    description='A DAG for MNIST model workflow',
    schedule_interval=timedelta(days=1),
    catchup=False
)

#새로운 데이터 모니터링
new_data_sensor = FileSensor(
    task_id='new_data_sensor',
    filepath='/Users/alookso/1007MLpj/ml-project-ml-pjt-7/uploads/*.{jpg,png,jpeg}',  # 새 이미지 파일을 모니터링할 경로
    poke_interval=3600,  # 1시간마다 확인
    dag=dag
)

def process_new_data():
    # 새 데이터 처리 및 예측
    new_data_folder = '/Users/alookso/1007MLpj/ml-project-ml-pjt-7/data'
    model = load_model()
    
    for filename in os.listdir(new_data_folder):
        if filename.endswith('.jpg'):
            file_path = os.path.join(new_data_folder, filename)
            input_data_list = preprocess_image_cv(file_path)
            predictions = predict(model, input_data_list)
            save_prediction(file_path, predictions)

process_new_data_task = PythonOperator(
    task_id='process_new_data',
    python_callable=process_new_data,
    dag=dag
)


def retrain_model():
    train_and_evaluate_model()

retrain_model_task = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model,
    dag=dag
)

   

def update_model_if_better(ti):
    
    current_model = load_model()
    new_model = load_model('saved_model.pth')

    # 테스트 데이터 준비
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1 - x),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    
    current_accuracy = evaluate_model(current_model, test_loader)
    new_accuracy = evaluate_model(new_model, test_loader)
    
    if new_accuracy > current_accuracy:
        torch.save(new_model.state_dict(), 'best_model.pth')
        print(f"모델이 업데이트되었습니다. 새 정확도: {new_accuracy:.2f}%")
    else:
        print(f"현재 모델이 더 우수합니다. 현재 정확도: {current_accuracy:.2f}%")

evaluate_and_update_model_task = PythonOperator(
    task_id='evaluate_and_update_model',
    python_callable=evaluate_and_update_model,
    dag=dag
)



# 태스크 순서
new_data_sensor >> process_new_data_task >> retrain_model_task >> evaluate_and_update_model_task
