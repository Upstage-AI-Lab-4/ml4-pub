# data_processing.py

import cv2
import numpy as np

def preprocess_image_cv(image_path):
    """
    OpenCV를 사용하여 이미지에서 숫자 영역을 검출하고 전처리합니다.

    Parameters:
    - image_path: 입력 이미지의 파일 경로

    Returns:
    - digits: 전처리된 개별 숫자 이미지의 리스트
    """
    # 이미지 로드 및 그레이스케일 변환
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 이미지 블러 적용 (노이즈 감소)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # 이미지 이진화 (배경을 흰색, 글자를 검은색으로)
    _, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 윤곽선 검출
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 검출된 윤곽선을 바운딩 박스로 변환하고 x 좌표 기준으로 정렬
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])  # x 좌표 기준 정렬

    digits = []
    for rect in bounding_boxes:
        x, y, w, h = rect
        # 너무 작은 영역은 무시 (노이즈 제거)
        if w < 10 or h < 10:
            continue

        # 숫자 영역 추출
        digit_img = thresh[y:y+h, x:x+w]

        # 이미지를 28x28 크기로 리사이즈
        digit_img = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)

        # 픽셀 값 정규화 및 차원 변경
        # digit_array = digit_img.astype(np.float32) / 255.0
        digit_array = (digit_img.astype(np.float32) - 128) / 128

        digit_array = digit_array.reshape(1, 28, 28)  # 배치 차원 제거
        digits.append(digit_array)

    return digits  # 전처리된 숫자 이미지 배열의 리스트 반환