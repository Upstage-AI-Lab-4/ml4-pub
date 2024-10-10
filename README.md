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

- 태블릿이나 키오스크 등을 통해 비밀번호를 직접 손으로 쓰는 경우, 입력한 손글씨를 정확한 숫자로 판별하는 모델을 개발합니다. 손글씨 인식을 위해 MNIST dataset을 기반으로 모델을 개발하고, FastAPI를 통해 고객에게 숫자 손글씨를 입력받아 정확한 숫자를 데이터베이스에 저장합니다. 저장된 이미지들은 모델 재학습에 사용되어 일정 양의 숫자 이미지가 쌓였을 때 모델이 자동으로 재학습을 하고, 재학습된 모델과 기존 모델의 성능을 비교하여 높은 성능의 모델을 저장하는 자동화 파이프라인을 구축합니다.

### Timeline

- Sep 26, 2024 : 첫 회의
- Sep 30 : 부족한 공부 보충 
- Oct 2 : 주제 선정 및 역할 분담
- Oct 7 : 모델 완료, 프론트 페이지 초안 완료
- Oct 8 : 모델 고도화 및 MLflow, Airflow 개발
- Oct 11 : 프로젝트 마감


## 2. Components

### Directory

- _Insert your directory structure_

e.g.
```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
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

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- [_Insert your meeting log link like Notion or Google Docs_](https://www.notion.so/4-558866cebfc14b4f87864a9f4cc46c84)

