from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model.model import Recommendation_MODEL
import datetime
import time

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

@app.get("/recommend/")
def recommend(art_level:int, history_level:int, nature_level:int, shopping_level:int, k:int=10):
    model = Recommendation_MODEL([art_level, history_level, nature_level, shopping_level])
    k_recommend = model.recommend(k)
    return k_recommend

@app.get("/get_planning/")
def planning(art_level:int, history_level:int, nature_level:int, shopping_level:int, milli_start_time:int, placePerDay:int, day:int, timePerDay:int):
    model = Recommendation_MODEL([art_level, history_level, nature_level, shopping_level])
    planning = model.planning(milli_start_time=milli_start_time, placePerDay=placePerDay, day=day, timePerDay=timePerDay)
    return planning