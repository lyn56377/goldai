from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib

app = FastAPI()

templates = Jinja2Templates(directory="templates")
model = joblib.load("text_rf_pipeline.pkl")

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/", response_class=HTMLResponse)
def predict(request: Request, text: str = Form(...)):
    prediction = model.predict([text])[0]
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": prediction
    })
