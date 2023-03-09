from fastapi import FastAPI
from model.model import TFID_MODEL

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello" : "Platea.co"}

@app.post("/recommendV1/{tripLevel}")
def recommend(art_level:int, history_level:int, nature_level:int, shopping_level:int, k:int=10):
    model = TFID_MODEL()
    k_recommend = model.recommend([art_level, history_level, nature_level, shopping_level], k)
    return k_recommend