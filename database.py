# database.py

import sqlite3
from contextlib import closing
from datetime import datetime
import os

DATABASE = 'database.db'
IMAGE_SAVE_FOLDER = 'saved_images'

def init_db():
    if not os.path.exists(IMAGE_SAVE_FOLDER):
        os.makedirs(IMAGE_SAVE_FOLDER)

    with closing(sqlite3.connect(DATABASE)) as conn:
        with conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_paths TEXT,
                    predictions TEXT,
                    timestamp TEXT
                )
            ''')

def save_prediction(images, predictions):
    """
    예측 결과와 이미지를 데이터베이스와 파일로 저장합니다.

    Parameters:
    - images: PIL 이미지 객체의 리스트
    - predictions: 예측된 숫자의 리스트
    """
    image_filenames = []
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    for idx, (img, pred) in enumerate(zip(images, predictions)):
        filename = f"{timestamp}_{idx}_{pred}.png"  # 예측된 숫자를 파일명에 포함
        filepath = os.path.join(IMAGE_SAVE_FOLDER, filename)
        img.save(filepath)
        image_filenames.append(filepath)

    image_paths_str = ','.join(image_filenames)
    predictions_str = ','.join(map(str, predictions))

    with closing(sqlite3.connect(DATABASE)) as conn:
        with conn:
            conn.execute('''
                INSERT INTO predictions (image_paths, predictions, timestamp)
                VALUES (?, ?, ?)
            ''', (image_paths_str, predictions_str, timestamp))

def get_all_data():
    with closing(sqlite3.connect(DATABASE)) as conn:
        with conn:
            cursor = conn.execute('SELECT image_paths, predictions, timestamp FROM predictions')
            data = [{
                'image_paths': row[0],
                'predictions': row[1],
                'timestamp': row[2]
            } for row in cursor.fetchall()]
    return data