import pandas as pd
import numpy as np
import random

from sklearn.metrics.pairwise import cosine_similarity

class TFID_MODEL:
    def __init__(self):
        self.detail = pd.read_csv("./model/detailTFIDF.csv")
        self.webDetail = self.get_webDetail()

        self.term_vector = self.get_termVector()
        self.type_list = self.get_type_list()
        self.type_vector = self.get_type_vector()

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
    def get_termVector(self):
        term_vector = np.array([np.fromstring(self.detail['tfidf'].to_numpy()[0][1:-1], sep=',')])

        for i in range(len(self.detail) - 1):
            term_vector_i = np.array([np.fromstring(self.detail['tfidf'].to_numpy()[i+1][1:-1], sep=',')])
            term_vector = np.append(term_vector, term_vector_i, axis=0)
        
        return term_vector

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


    def get_term_score(self, all_random_index):
        # Term vector
        # Select vector from upper index
        trip_term_vector = np.zeros(self.term_vector[0].shape)

        for idx_vector in all_random_index:
            trip_term_vector += self.term_vector[idx_vector]

        trip_term_vector = trip_term_vector / len(all_random_index)
        trip_term_score = cosine_similarity(np.append(self.term_vector, trip_term_vector.reshape(1,-1), axis=0))

        return trip_term_score

    def get_type_score(self, all_random_index):
        # Type vector
        trip_type_vector = np.zeros(self.type_vector[0].shape)

        for idx_vector in all_random_index:
            trip_type_vector += self.type_vector[idx_vector]

        trip_type_vector = trip_type_vector / len(all_random_index)
        trip_type_score = cosine_similarity(np.append(self.type_vector, trip_type_vector.reshape(1,-1), axis=0))

        return trip_type_score
    
    def recommend(self, trip_level:list, k:int):
        random_index = self.get_random_index(trip_level)
        trip_term_score = self.get_term_score(random_index)
        trip_type_score = self.get_type_score(random_index)

        # Rating score
        rating_score = self.webDetail['rating'].to_numpy()
        rating_score = np.append(rating_score, 5)

        relevance_score = trip_term_score[-1]*0.5 + trip_type_score[-1]*0.3 + rating_score*0.2

        # Recommendation
        k_recommed = {}
        distances = sorted(list(enumerate(relevance_score.reshape(-1))), reverse=True, key=lambda x:x[1])
        for idx, i in enumerate(distances[1:k+1]): # Top 10
            k_recommed[idx] = {
                "ID" : self.detail.iloc[i[0]]['placeID'],
                "name" : self.detail.iloc[i[0]]['name'],
                "address" : self.detail.iloc[i[0]]['address'],
                "geometry" : self.detail.iloc[i[0]]['geometry'],
                "phone_number" : self.detail.iloc[i[0]]['phone_number'],
                }
            # print(self.detail.iloc[i[0]]['name'])
        return k_recommed