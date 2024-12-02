from fastapi import APIRouter, Form, BackgroundTasks, UploadFile, File
import os
from fastapi.responses import JSONResponse
from pathlib import Path
from src.services.vision_services import evaluate_image_service

eval = APIRouter(
    prefix="/Vision",
    responses={
        200: {"description": "Successful"},
        400: {"description": "Bad Request"},
        403: {"description": "Unauthorized"},
        500: {"description": "Internal Server Error"}
    },
    tags=["Image Evaluation"]
)

# Define the directory for saving images
IMAGES_DIR = "images"

# Ensure the directory exists
Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)


@eval.post("/evaluate")
async def evaluate_image(
        file: UploadFile = File(default=None, description="The file attached")
):
    # Save the file -- UNCOMMENT LINES 30-32 IF YOU WOULD LIKE TO SEE THE IMAGE IN IMAGES DIRECTORY
    # file_path = os.path.join(IMAGES_DIR, file.filename)
    # with open(file_path, "wb") as buffer:
    #     buffer.write(await file.read())

    # Evaluate the image
    result = await evaluate_image_service(file)
    return JSONResponse(content=result)
