# api_module.py

import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from data_processing import preprocess_canvas_images
from model import load_model, predict
from monitoring import log_info, log_error
from database import save_prediction

def create_app():
    app = FastAPI()

    # 모델 로드
    model = load_model()

    # 템플릿 설정
    templates = Jinja2Templates(directory="templates")

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        return templates.TemplateResponse("index.html", {"request": request, "result": None, "input_value": ""})

    @app.post("/", response_class=HTMLResponse)
    async def predict_route(
        request: Request,
        image1: str = Form(...),
        image2: str = Form(...),
        image3: str = Form(...),
        image4: str = Form(...)
    ):
        try:
            images_data = [image1, image2, image3, image4]

            # 이미지 전처리
            input_data_list, raw_images = preprocess_canvas_images(images_data)

            # 예측 수행
            predictions = predict(model, input_data_list)

            # 결과 저장 (이미지와 예측 결과를 함께 저장)
            save_prediction(raw_images, predictions)

            # 로그 정보
            log_info(f"Predictions: {predictions}")

            # 결과 반환
            return templates.TemplateResponse("index.html", {"request": request, "predictions": predictions, "error": None, "image1": image2,"image2": image3,"image3": image1,"image4": image4})

        except Exception as e:
            log_error(f"Error in predict_route: {str(e)}")
            return templates.TemplateResponse("result.html", {"request": request, "predictions": None, "error": "내부 서버 오류"})

    return app