from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import os

# Initialize FastAPI
app = FastAPI()

# Load model
model = joblib.load("text_rf_pipeline.pkl")

# Set up HTML templates
templates = Jinja2Templates(directory="templates")

# Mount static files (if any CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Route for the form UI
@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# Route for form submission
@app.post("/", response_class=HTMLResponse)
async def post_form(request: Request, user_input: str = Form(...)):
    prediction = model.predict([user_input])[0]
    return templates.TemplateResponse("index.html", {"request": request, "result": prediction, "user_input": user_input})
