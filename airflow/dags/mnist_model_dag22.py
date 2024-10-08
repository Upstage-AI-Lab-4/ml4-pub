import sys
import os
# model.py 파일이 있는 디렉토리를 Python 경로에 추가
sys.path.append('/Users/alookso/1007MLpj/ml-project-ml-pjt-7')

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator 
from airflow.operators.empty import EmptyOperator 
from airflow.sensors.filesystem import FileSensor

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
from sqlalchemy import create_engine

# 절대 경로를 사용하여 모듈 import
from model import CNNModel, load_model, train_and_evaluate_model, evaluate_model, predict
from data_processing import preprocess_image_cv
from database import init_db, save_prediction

# 데이터베이스 연결 설정
DB_CONNECTION = f"sqlite:////{os.path.abspath('/Users/alookso/1007MLpj/ml-project-ml-pjt-7/database.db')}"
engine = create_engine(DB_CONNECTION)

def process_new_data(**context):
    # 새 데이터 처리 및 예측
    new_data_folder = '/Users/alookso/1007MLpj/ml-project-ml-pjt-7/uploads'
    model = load_model()
        
    for filename in os.listdir(new_data_folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            try:
                file_path = os.path.join(new_data_folder, filename)
                #이미지 전처리
                input_data_list = preprocess_image_cv(file_path)
                #학습된 모델이 예측
                predictions = predict(model, input_data_list)
                #DB에 저장
                save_prediction(file_path, predictions)
                processed_files += 1
                logging.info(f"Processed file: {filename}")
            except Exception as e:
                logging.error(f"Error processing file {filename}: {str(e)}")
    logging.info(f"Total files processed: {processed_files}")
    return processed_files
    

def retrain_model(**kwargs):
    final_model_path = train_and_evaluate_model(use_all_data=True)
    kwargs['ti'].xcom_push(key='model_path', value=final_model_path)

def update_model_if_better(**kwargs):
    ti = kwargs['ti']
    new_model_path = ti.xcom_pull(task_ids='retrain_model', key='model_path')
    current_model_path = 'best_model.pth'

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


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 8),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=20),
}

# DAG 정의
with DAG('mnist_model_workflow_2',
        default_args=default_args,
        description='A DAG for MNIST model workflow',
        schedule_interval='*/5 * * * *',  # 5분마다 실행
        start_date=datetime(2024, 10, 8),  # 시작 날짜 설정
        catchup=False,
        max_active_runs=1
    ) as dag:

    #새로운 데이터 모니터링
    new_data_sensor = FileSensor(
        task_id="new_data_sensor",
        filepath="/Users/alookso/1007MLpj/ml-project-ml-pjt-7/uploads}",  # 새 이미지 파일을 모니터링할 경로
        poke_interval=60,  # 1분마다 확인
        timeout=300, # 5분 후 타임아웃
        dag=dag
    )

    process_new_data_task = PythonOperator(
        task_id='process_new_data',
        python_callable=process_new_data,
        provide_context=True,
        dag=dag
    )

    retrain_model_task = PythonOperator(
        task_id='retrain_model',
        python_callable=retrain_model,
        provide_context=True,
        dag=dag
    )
    

    evaluate_and_update_model_task = PythonOperator(
        task_id='evaluate_and_update_model',
        python_callable=update_model_if_better,
        dag=dag
    )



# 태스크 순서
new_data_sensor >> process_new_data_task >> retrain_model_task >> evaluate_and_update_model_task
