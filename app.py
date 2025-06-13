from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import requests
import base64
import uuid
import os

app = FastAPI()

# Mount static files (for generated images, CSS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Ensure output directory exists
output_dir = os.path.join("static", "output")
os.makedirs(output_dir, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result_url": None, "error": None})

@app.post("/", response_class=HTMLResponse)
async def generate_tryon(
    request: Request,
    human_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    category: str = Form(...),
    garment_desc: str = Form(...)
):
    # Read uploaded files and convert to base64 strings
    human_bytes = await human_image.read()
    cloth_bytes = await cloth_image.read()
    human_base64 = base64.b64encode(human_bytes).decode("utf-8")
    cloth_base64 = base64.b64encode(cloth_bytes).decode("utf-8")
    
    # Segmind IDM-VTON API details
    API_KEY = "SG_148ff614671b19a9"  # anushkamane.10 Replace with your actual API key if needed
    # API_KEY = "SG_098d0d2b0eccec60"  #anushka.1012
    # API_KEY ="SG_806a89c36592210a"
    API_URL = "https://api.segmind.com/v1/idm-vton"
    
    # Prepare the payload for the API call
    payload = {
        "crop": False,
        "seed": 42,
        "steps": 30,
        "category": category,
        "force_dc": False,
        "human_img": human_base64,
        "garm_img": cloth_base64,
        "mask_only": False,
        "garment_des": garment_desc
    }
    
    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    
    # Call the Segmind API
    response = requests.post(API_URL, json=payload, headers=headers)
    
    if response.status_code == 200:
        # Save the output image in static/output folder
        output_filename = f"output_{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "wb") as f:
            f.write(response.content)
        # Build the URL for the generated image
        result_url = f"/static/output/{output_filename}"
        error = None
    else:
        result_url = None
        error = f"Error: {response.status_code} - {response.text}"
    
    # Render the template with result and any error message
    return templates.TemplateResponse("index.html", {"request": request, "result_url": result_url, "error": error})
@app.get("/details", response_class=HTMLResponse)
async def details(request: Request):
    return templates.TemplateResponse("details.html", {"request": request})
