import pandas as pd
import numpy as np
import random
import datetime
import time
import pickle
from math import radians

from gensim.models import KeyedVectors

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import haversine_distances

class Recommendation_MODEL:
    def __init__(self, trip_level:list):
        self.trip_level = trip_level
        
        self.detail = pd.read_csv("./model/detailTFIDF.csv")
        self.webDetail = self.get_webDetail()

        self.term_vector_tfidf = self.get_term_vector_tfidf()
        self.term_vector_word2vec = self.get_term_vector_word2vec()
        self.type_list = self.get_type_list()
        self.type_vector = self.get_type_vector()

        self.visited = np.zeros((len(self.detail)+1,))
        self.visited[62] = 1

    def get_webDetail(self):
        webDetail = pd.read_csv('./data/file/finalPlace.csv')
        webDetail['use'] = np.NaN
        placeName = self.detail['name']
        for idx, place in enumerate(webDetail['placeName']):
            for p in placeName:
                if place==p:
                    webDetail.loc[idx, "use"] = 1

        webDetail.dropna(inplace=True)
        webDetail = webDetail.reset_index(drop=True)

        return webDetail

    # matrix of tfidf score
    def get_term_vector_tfidf(self):
        term_vector = np.array([np.fromstring(self.detail['tfidf'].to_numpy()[0][1:-1], sep=',')])

        for i in range(len(self.detail) - 1):
            term_vector_i = np.array([np.fromstring(self.detail['tfidf'].to_numpy()[i+1][1:-1], sep=',')])
            term_vector = np.append(term_vector, term_vector_i, axis=0)
        
        return term_vector
    
    def get_term_vector_word2vec(self):
        with open('./model/term_word2vec.pkl', 'rb') as file:
            term_vctor_word2vec = pickle.load(file)
            file.close()
        return term_vctor_word2vec

    def get_type_list(self):
        # Type from Place API (Too much type)
        type_set = set()

        for idx, type_i in enumerate(self.detail['type2']):
            for t in type_i.split(','):
                type_set.add(t)

        type_list = sorted(list(type_set))
        return type_list

    def get_type_vector(self):
        # Vector ['art', 'history', 'nature', 'special', 'shopping']
        type_vector = np.zeros((len(self.detail), len(self.type_list)))

        for idx in range(len(self.detail)):
            type_vector_i = np.zeros(5)
            for t in self.detail['type2'][idx].split(','):
                for i, t_i in enumerate(self.type_list):
                    if t==t_i:
                        type_vector_i[i]=1
            
            type_vector[idx] = type_vector_i
        return type_vector

    def get_random_index(self, trip_level:list):
        # Find index of each type from type vector (Adding more History before TFIDF)
        idx_type_vector = {
            "art":[],
            "history":[],
            "nature":[],
            "shopping":[],
            "special":[]
        }

        for idx, type_vector_i in enumerate(self.type_vector):
            for i, x in enumerate(type_vector_i):
                if x==1:
                    if i==0:
                        idx_type_vector["art"].append(idx)
                    if i==1:
                        idx_type_vector["history"].append(idx)
                    if i==2:
                        idx_type_vector["nature"].append(idx)
                    if i==3:
                        idx_type_vector["shopping"].append(idx)
                    if i==4:
                        idx_type_vector["special"].append(idx)

        # Random choices place index follow Trip level
        all_random_index = set()
        # Number of place art is 4
        try:
            random_index_art = random.sample(idx_type_vector['art'], k=trip_level[0])
        except:
            random_index_art = random.sample(idx_type_vector['art'], k=trip_level[0]-1)
        random_index_history = random.sample(idx_type_vector['history'], k=trip_level[1])
        random_index_nature = random.sample(idx_type_vector['nature'], k=trip_level[2])
        random_index_shopping = random.sample(idx_type_vector['shopping'], k=trip_level[3])

        for i in random_index_art: all_random_index.add(i)
        for i in random_index_history: all_random_index.add(i)
        for i in random_index_nature: all_random_index.add(i)
        for i in random_index_shopping: all_random_index.add(i)

        all_random_index = list(all_random_index)
        
        return all_random_index


    def get_term_score(self, all_random_index, model):
        # Term vector
        # Select vector from upper index
        if model=='tfidf':
            # Term vector
            # Select vector from upper index
            trip_term_vector = np.zeros(self.term_vector_tfidf[0].shape)

            for idx_vector in all_random_index:
                trip_term_vector += self.term_vector_tfidf[idx_vector]

            trip_term_vector = trip_term_vector / len(all_random_index)
            trip_term_score = cosine_similarity(np.append(self.term_vector_tfidf, trip_term_vector.reshape(1,-1), axis=0))
            return trip_term_score
        if model=='word2vec':
            # Term vector
            # Select vector from upper index
            trip_term_vector = np.zeros(self.term_vector_word2vec["เซ็นทรัลเวิลด์"].shape)

            for idx_vector in all_random_index:
                trip_term_vector += self.term_vector_word2vec[self.detail.iloc[idx_vector, 1]]

            trip_term_vector = trip_term_vector / len(all_random_index)
            trip_term_score = cosine_similarity(np.append(np.array(list(self.term_vector_word2vec.values())), trip_term_vector.reshape(1,-1), axis=0))
            return trip_term_score


    def get_type_score(self, all_random_index):
        # Type vector
        trip_type_vector = np.zeros(self.type_vector[0].shape)

        for idx_vector in all_random_index:
            trip_type_vector += self.type_vector[idx_vector]

        trip_type_vector = trip_type_vector / len(all_random_index)
        trip_type_score = cosine_similarity(np.append(self.type_vector, trip_type_vector.reshape(1,-1), axis=0))

        return trip_type_score
    
    # Distance (lat,long)
    def get_distance_score(self, place_index):
        start_idx = place_index

        distance = np.zeros((62,))
        geometry_list = self.detail['geometry'].map(lambda x  : eval(x)).to_list()
        
        current_point = eval(self.detail.iloc[start_idx]['geometry'])
        self.visited[start_idx] = 1

        current_radian = [radians(_) for _ in current_point]

        for idx, dest_point in enumerate(geometry_list):
            dest_radian = [radians(_) for _ in dest_point]
            distance[idx] = haversine_distances([current_radian, dest_radian])[0][1] * 6371 # 6371 is radius of the Earth
        
        return np.append(1/distance, np.Inf)
    
    def recommend(self, k:int, model="word2vec"):
        random_index = self.get_random_index(self.trip_level)
        trip_term_score = self.get_term_score(random_index, model=model)
        trip_type_score = self.get_type_score(random_index)

        # Rating score
        rating_score = self.webDetail['rating'].to_numpy()
        rating_score = np.append(rating_score, 5)
        # Distance score
        trip_distance_score = np.zeros((len(self.detail),))
        trip_distance_score = np.append(trip_distance_score, np.Inf)
        
        k_recommed = []
        for i in range(k):
            # tfidf term_score is important score because the different of value are far
            relevance_score = trip_term_score[-1]*2 + trip_type_score[-1]*1 + rating_score*0.05 + trip_distance_score * 1
            # Recommendation
            place_recommended = sorted(list(enumerate(relevance_score.reshape(-1))), reverse=True, key=lambda x:x[1])
            for idx, similar in place_recommended:
                # print(idx)
                if not self.visited[idx]:
                    k_recommed.append({
                        "idx" : idx,
                        "name" : self.detail.iloc[idx]['name'],
                        "address" : self.detail.iloc[idx]['address'],
                        "geometry" : { "lat" : self.detail.iloc[idx]['geometry'].split(',')[0][1:],
                                    "lon": self.detail.iloc[idx]['geometry'].split(',')[-1][:-1]
                                    },
                        "phone_number" : self.detail.iloc[idx]['phone_number'],
                        })
                    self.visited[idx]=1
                    break

        return k_recommed
    
    def planning(self, milli_start_time, placePerDay:int, day:int, timePerDay:int):
        if placePerDay*day > len(self.detail):
            return {"Error" : "input error"}
        # input time_start is milliseonds form
        k_recommend = self.recommend(len(self.detail))
        HourPerPlace = int(timePerDay / placePerDay)
        MinPerPlace = ((timePerDay/placePerDay)*60) - (HourPerPlace*60)

        planning = {}
        idx = 0

        for d in range(day):
            placeIn = []
            time_start = datetime.datetime.fromtimestamp(milli_start_time/1000) + datetime.timedelta(days=d)
            for j in range(placePerDay):
                time_end = time_start + datetime.timedelta(hours=HourPerPlace, minutes=MinPerPlace)
                milli_time_start = int(time.mktime(time_start.timetuple())) * 1000
                milli_time_end = int(time.mktime(time_end.timetuple())) * 1000

                placeIn.append({
                    'name': k_recommend[idx]["name"],
                    'geometry': k_recommend[idx]['geometry'],
                    'address' : k_recommend[idx]['address'],
                    'phone_number': k_recommend[idx]['phone_number'],
                    'time': {
                            "time_start": milli_time_start,
                            "time_end" :  milli_time_end
                            }
                })
                time_start = time_end
                idx+=1
            planning[str(d+1)] = placeIn
        return planning
