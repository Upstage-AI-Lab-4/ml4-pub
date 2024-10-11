import sys
import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator 
from airflow.operators.empty import EmptyOperator 
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sqlalchemy import create_engine
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from model import CNNModel, load_model, train_and_evaluate_model, evaluate_model, predict
from data_processing import preprocess_canvas_images, preprocess_image_file
from database import init_db, save_prediction


def preprocessing_new_data(**context):
    new_images_folder = '/Users/alookso/1010LMpj/ml-project-ml-pjt-7/saved_images'
    processed_data_folder = '/Users/alookso/1010LMpj/ml-project-ml-pjt-7/processed_data'
    os.makedirs(processed_data_folder, exist_ok=True)
    
    processed_files = []
    labels = []

    transform = transforms.Normalize((0.5,), (0.5,))

    for filename in os.listdir(new_images_folder):
        if filename.endswith('.png'):
            try:
                image_path = os.path.join(new_images_folder, filename)
                label = int(os.path.splitext(filename)[0])  # 파일명에서 레이블 추출
                
                image_array = preprocess_image_file(image_path)
                image_tensor = torch.from_numpy(image_array).float()
                
                if image_tensor.shape[0] != 1:
                    image_tensor = image_tensor.permute(2, 0, 1)
                
                image_tensor = transform(image_tensor)

                # 처리된 이미지를 파일로 저장
                processed_file_path = os.path.join(processed_data_folder, f"processed_{filename}.pt")
                torch.save(image_tensor, processed_file_path)
                
                processed_files.append(processed_file_path)
                labels.append(label)
                
                logging.info(f"Processed image: {filename}")
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")

    if processed_files:
        logging.info(f"Processed {len(processed_files)} new images")
        return {"processed_files": processed_files, "labels": labels}
    else:
        logging.warning("No new images processed")
        return None


#전처리한 데이터를 사용하기 위해 함수화
def use_processed_data(**context):
    ti = context['ti']
    processed_data = ti.xcom_pull(task_ids='preprocessing_new_data')
    if processed_data:
        processed_files = processed_data['processed_files']
        labels = processed_data['labels']
        
        images = []
        for file_path in processed_files:
            image_tensor = torch.load(file_path)
            images.append(image_tensor)
        
        images = torch.stack(images)
        labels = torch.tensor(labels)
        
        return images, labels
    else:
        return None, None

def retrain_model(**context):
    try:
        # 전처리된 데이터 가져오기
        images, labels = use_processed_data(**context)
        
        if images is None or labels is None:
            print("No new data to train on.")
            return
        
        # 기존 모델 로드
        model_path = os.path.join(os.path.dirname(__file__), '..', 'saved_model.pth')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(model_path)
        model.to(device)
        
        # 새 데이터로 데이터셋 생성
        new_dataset = TensorDataset(images, labels)
        new_dataloader = DataLoader(new_dataset, batch_size=32, shuffle=True)
        
            
        # 모델 재학습
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
        for epoch in range(5):  # 5 에폭 동안 재학습
            model.train()
            for batch_images, batch_labels in new_dataloader:
                optimizer.zero_grad()
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
        # 재학습된 모델 저장
        retrained_model_path = os.path.join(os.path.dirname(__file__), '..', 'retrained_model.pth')
        torch.save(model.state_dict(), retrained_model_path)
        logging.info(f"Model retrained and saved to {retrained_model_path}")
        
    except Exception as e:
        logging.error(f"Error in retrain_model: {str(e)}")
        raise
            

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 10),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG 정의
with DAG('handwriting_dag',
        default_args=default_args,
        description='A DAG for MNIST model workflow',
        schedule_interval=timedelta(days=1), 
        start_date=datetime(2024, 10, 10),  # 시작 날짜 설정
        catchup=False) as dag:

    #새로운 데이터 모니터링
    new_data_sensor = FileSensor(
        task_id="new_data_sensor",
        fs_conn_id="file_sensor",
        filepath="/Users/alookso/1010LMpj/ml-project-ml-pjt-7/saved_images",  # 새 이미지 파일을 모니터링할 경로
        poke_interval=60,  # 1분마다 확인
        timeout=300, # 5분 후 타임아웃
        mode='poke',
        dag=dag
    )

    preprocessing_new_data_task = PythonOperator(
        task_id='preprocessing_new_data',
        python_callable=preprocessing_new_data,
        provide_context=True,
        execution_timeout=timedelta(minutes=30),
        dag=dag
    )

    retrain_model_task = PythonOperator(
        task_id='retrain_model',
        python_callable=retrain_model,
        provide_context=True,
        execution_timeout=timedelta(minutes=30),
        dag=dag
    )

# 태스크 순서
new_data_sensor >> preprocessing_new_data_task >> retrain_model_task



