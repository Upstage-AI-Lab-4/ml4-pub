# 4자리 숫자 비밀번호 인식 모델
## Team

| ![유재현](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김동규](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김효원](https://avatars.githubusercontent.com/u/156163982?v=4) | ![문기중](https://avatars.githubusercontent.com/u/156163982?v=4) | ![한성범](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [유재현](https://github.com/hyeon3730)             |            [김동규](https://github.com/Lumiere001)             |            [김효원](https://github.com/hannakhw)             |            [문기중](https://github.com/provismoon)             |            [한성범](https://github.com/winterbeom)             |
|                            팀장, ML 모델                             |                            FastAPI, 백엔드 서버, DB                             |                            Airflow                             |                            MLflow                             |                            Airflow                             |

## 0. Overview
### Environment
- Python3.10~11
  

### Requirements
- Requirements.txt 참조

## 1. Competiton Info

### Overview

- 시각 장애인들에게 현재 OTP 인증 과정에서의 불편함은 접근성의 큰 장벽으로 작용하고 있습니다. 본 프로젝트는 이러한 문제를 해결하려고 합니다. 시각 장애인이 손글씨로 비밀번호를 입력하면 이를 인식하여 음성으로 제공하는 서비스를 개발하는 것을 목표로 합니다.


### Timeline

- Sep 26, 2024 : 첫 회의
- Sep 30 : 부족한 공부 보충 
- Oct 2 : 주제 선정 및 역할 분담
- Oct 7 : 모델 완료, 프론트 페이지 초안 완료
- Oct 8 : 모델 고도화 및 MLflow, Airflow 개발
- Oct 11 : 프로젝트 마감


## 2. Module

- 사용자 입력: 시각 장애인은 웹 페이지의 캔버스에 손글씨로 숫자를 입력합니다.
- 이미지 전처리: 입력된 이미지 데이터는 서버로 전송되어 전처리 과정을 거칩니다.
- 모델 예측: 전처리된 이미지는 머신러닝 모델에 의해 비밀번호로 인식됩니다.
- 결과 제공: 인식된 비밀번호는 음성으로 변환되어 사용자에게 전달됩니다.
- 데이터 저장: 입력된 이미지와 예측 결과는 데이터베이스에 저장되어 모델의 성능 향상을 위해 활용됩니다.
- 모델 관리 및 배포: MLflow와 airflow를 통해 모델의 실험을 추적하고, 최적의 모델을 등록 및 배포합니다.


```
├── code
│   ├── api_module.py
│   ├── data_processing.py
│   ├── model.py
│   ├── database.py
│   ├── mlflow_model.py
└────── monitoring.py
│── dags 
│    ├──dags.py
│
└─── data

```

## 3. Data descrption

### Dataset overview

- 사람이 직접 손으로 입력한 손글씨 숫자 4개 데이터

### Data Preprocessing

- 캔버스에 그린 이미지를 모델에 입력하기 위한 전처리 과정
- images_data: 캔버스에서 받은 base64 인코딩된 이미지 문자열의 리스트
- 전처리된 이미지 배열의 리스트와 PIL 이미지 객체의 리스트로 반환

## 4. Modeling

- CNNMode
- 데이터 정확도 %


## 5. Result

![스크린샷 2024-10-10 오후 5 24 35](https://github.com/user-attachments/assets/27f7befb-18e2-45c3-8eae-c90082fef366)


### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- [_4조 노션 링크_](https://www.notion.so/4-558866cebfc14b4f87864a9f4cc46c84)

