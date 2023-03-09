from model.model import TFID_MODEL

model = TFID_MODEL()

trip_level = {"art":5, "history":3, "nature":3, "shopping":5}
k_recommend = model.recommend(trip_level= list(trip_level.values()), k=10)
print(k_recommend)