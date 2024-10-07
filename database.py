# database.py

import sqlite3
from contextlib import closing

DATABASE = 'database.db'

def init_db():
    with closing(sqlite3.connect(DATABASE)) as conn:
        with conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT,
                    label TEXT
                )
            ''')

def save_prediction(image_path, predictions):
    """
    예측 결과를 데이터베이스에 저장합니다.

    Parameters:
    - image_path: 입력 이미지의 경로
    - predictions: 예측된 숫자의 리스트
    """
    predictions_str = ','.join(map(str, predictions))
    with closing(sqlite3.connect(DATABASE)) as conn:
        with conn:
            conn.execute('''
                INSERT INTO predictions (image_path, label)
                VALUES (?, ?)
            ''', (image_path, predictions_str))

def get_all_data():
    with closing(sqlite3.connect(DATABASE)) as conn:
        with conn:
            cursor = conn.execute('SELECT image_path, label FROM predictions')
            data = [{'image_path': row[0], 'label': row[1]} for row in cursor.fetchall()]
    return data