from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from src.controllers.vision_controller import eval
from fastapi.staticfiles import StaticFiles

# App Details
vision = FastAPI(
    title="Vision Demo",
    summary="Vision",
    version="1",
    swagger_ui_parameters={
        "syntaxHighlight.theme": "obsidian",
        "docExpansion": "none"
    }
)
vision.mount("/vision", StaticFiles(directory="static", html=True), name="static")
vision.include_router(eval)

vision.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)


@vision.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/vision")
