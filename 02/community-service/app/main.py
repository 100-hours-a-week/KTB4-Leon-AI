from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import create_tables
from app.routes.posts import router as posts_router

app = FastAPI(title="Community Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    create_tables()


@app.get("/")
def read_root():
    return {"message": "Community Service API"}


@app.get("/ai/recommend-title")
def recommend_title(keyword: str):
    return {"title": f"{keyword}에 대한 이야기"}


app.include_router(posts_router)
