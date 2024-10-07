# api_module.py

import os
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from data_processing import preprocess_image_cv
from model import load_model, predict
from monitoring import log_info, log_error
from database import save_prediction

def create_app():
    app = FastAPI()

    # Upload folder configuration
    UPLOAD_FOLDER = 'uploads/'
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Load model
    model = load_model()

    # Template configuration
    templates = Jinja2Templates(directory="templates")

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.post("/predict", response_class=HTMLResponse)
    async def predict_route(request: Request, image: UploadFile = File(...)):
        try:
            # Save file
            filename = image.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await image.read())

            # Preprocess image
            input_data_list = preprocess_image_cv(file_path)

            if not input_data_list:
                return templates.TemplateResponse("result.html", {"request": request, "predictions": None, "error": "No digits found in image"})

            # Perform prediction
            predictions = predict(model, input_data_list)

            # Save results
            save_prediction(file_path, predictions)

            # Log info
            log_info(f"File {filename} uploaded, predictions: {predictions}")

            # Return results
            return templates.TemplateResponse("result.html", {"request": request, "predictions": predictions, "error": None})

        except Exception as e:
            log_error(f"Error in predict_route: {str(e)}")
            return templates.TemplateResponse("result.html", {"request": request, "predictions": None, "error": "Internal server error"})

    return app
