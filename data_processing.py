# data_processing.py

import base64
from io import BytesIO
from PIL import Image
import numpy as np
import os

IMAGE_SAVE_FOLDER = 'saved_images'

def preprocess_canvas_images(images_data):
    """
    캔버스에서 그린 이미지를 모델 예측을 위해 전처리합니다.

    Parameters:
    - images_data: 캔버스에서 받은 base64 인코딩된 이미지 문자열의 리스트

    Returns:
    - digits: 전처리된 이미지 배열의 리스트
    - raw_images: 원본 이미지를 저장하기 위한 PIL 이미지 객체의 리스트
    """
    digits = []
    raw_images = []
    for idx, image_data in enumerate(images_data):
        # 'data:image/png;base64,' 헤더 제거
        if ',' in image_data:
            header, encoded = image_data.split(',', 1)
        else:
            encoded = image_data
        decoded = base64.b64decode(encoded)
        img = Image.open(BytesIO(decoded)).convert('L')  # 그레이스케일 변환

        # 이미지의 크기를 28x28로 조정하면서 고품질의 리샘플링 필터 사용
        img = img.resize((28, 28), Image.LANCZOS)

        # 픽셀 값이 0~255 사이인지 확인
        img_array = np.array(img).astype(np.float32)

        # 픽셀 값 정규화
        img_array = img_array / 255.0

        # 배열 형태 변경
        img_array = img_array.reshape(1, 28, 28)

        digits.append(img_array)
        raw_images.append(img)

        # 디버깅을 위해 전처리된 이미지를 저장
        if not os.path.exists(IMAGE_SAVE_FOLDER):
            os.makedirs(IMAGE_SAVE_FOLDER)
        img.save(os.path.join(IMAGE_SAVE_FOLDER, f"processed_image_{idx}.png"))

    return digits, raw_images


def preprocess_image_file(image_path):
    """
    파일 시스템에서 이미지를 로드하고 전처리합니다.

    Parameters:
    - image_path: 이미지 파일의 경로

    Returns:
    - img_array: 전처리된 이미지 배열
    """
    img = Image.open(image_path).convert('L')  # 그레이스케일 변환
    img = img.resize((28, 28), Image.LANCZOS)  # 크기 조정
    img_array = np.array(img).astype(np.float32)
    img_array = img_array / 255.0  # 정규화
    img_array = img_array.reshape(1, 28, 28)  # 형태 변경
    return img_array

