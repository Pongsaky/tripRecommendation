from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model.model import TFID_MODEL

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return "Hello Platea.co"

@app.post("/recommend/")
def recommend(art_level:int, history_level:int, nature_level:int, shopping_level:int, k:int=10):
    model = TFID_MODEL()
    k_recommend = model.recommend([art_level, history_level, nature_level, shopping_level], k)
    return k_recommend