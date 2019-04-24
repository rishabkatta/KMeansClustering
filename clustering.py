'''
@author-name: Rishab Katta

Python program for performing k-means clustering on MongoDB IMDB Database.
All the movies released in close period of time and have close avgratings are clustered together.

NOTE: This program assumes that there's already a MongoDB Database by the name IMDB.
'''

from pymongo import MongoClient
import time
import pymongo
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt



class MongoDBManagement:

    def __init__(self,host,port):
        '''
        Constructor used to connect to both the databases and initialize a MongoDB database

        :param host: Hostname for MongoDB and Postgres databases
        :param port: Port number for MongoDB database
        :param pdb: Postgres database name
        :param pun: Postgres user name
        :param ppwd: Postgres password
        '''
        self.client = MongoClient(host, port)
        self.database = self.client['IMDB']


    def kmeansNorm(self):
        '''
        Calculate K-means Norm for all the movies in the movies collection
        :return: None
        '''
        self.collection = self.database['Movies']

        pipeline_max_min = [{"$match": {"$and": [
                           {"type": "movie"}, {"numvotes": {"$gt": 10000}},
                           {"startyear": {"$exists": True}},{"avgrating": {"$exists": True}}]}},
                            { "$group": {"_id": None, "max_sy": { "$max": "$startyear"},"min_sy": { "$min": "$startyear"},
                                         "max_ar": { "$max": "$avgrating"}, "min_ar": { "$min": "$avgrating"}}},
                           {"$project": {"_id": 0,"max_sy":1, "min_sy":1, "max_ar":1, "min_ar":1 }}]

        min_max= self.collection.aggregate(pipeline_max_min)
        min_max_list = list(min_max)
        max_sy = min_max_list[0]["max_sy"]
        min_sy = min_max_list[0]["min_sy"]
        max_ar = min_max_list[0]["max_ar"]
        min_ar = min_max_list[0]["min_ar"]


        pipeline_q1 = [{"$match": {"$and": [
                           {"type": "movie"}, {"numvotes": {"$gt": 10000}},
                           {"startyear": {"$exists": True}},{"avgrating": {"$exists": True}}]}},
                       {"$project": {"_id": 1}}]

        for doc in (self.collection.aggregate(pipeline_q1)):
            sa_doc = self.collection.find({ "_id": doc["_id"] }, { "startyear": 1, "avgrating": 1, "_id":0})
            sa_list=list(sa_doc)
            v_sy= sa_list[0]["startyear"]
            v_ar = sa_list[0]["avgrating"]
            norm_sy = (v_sy - min_sy)/(max_sy-min_sy)
            norm_ar = (v_ar - min_ar)/(max_ar-min_ar)
            kmeansnorm = [norm_sy,norm_ar]
            self.collection.update_one({"_id": doc["_id"]}, {"$set": { "kmeansnorm" : kmeansnorm}})




    def centroids(self,g,k):
        '''
        Generate random k centroids for genre g and create & populate Centroids Collection
        :param g: genre of the movie
        :param k: number of centroids
        :return: None
        '''
        self.collection = self.database['Movies']
        pipeline_q2 = [{"$match": {"$and": [
                           {"genres": g}, {"kmeansnorm": {"$exists": True}}]}},
                       { "$sample" : {"size": k}},
                       {"$project": {"kmeansnorm": 1, "_id":0}}]
        norm_values=[]
        for doc in (self.collection.aggregate(pipeline_q2)):
            norm_values.append(doc['kmeansnorm'])

        # self.database.create_collection('Centroids')
        self.collection = self.database['Centroids']
        for i in range(k):
            cdoc={}
            cdoc["_id"] = i+1
            cdoc['kmeansnorm'] = norm_values[i]
            self.collection.insert_one(cdoc)




    def kmeans_onestep(self,g):
        '''
        Run one step of the kmeans algorithm
        :param g: genre of the movie
        :return: None
        '''
        self.movies_collection = self.database['Movies']
        pipeline_q2 = [{"$match": {"$and": [
            {"genres": g}, {"kmeansnorm": {"$exists": True}}]}},
            {"$project": {"kmeansnorm": 1, "_id": 1}}]

        self.cent_collection = self.database['Centroids']
        pipeline_q3 = [{"$project": {"kmeansnorm": 1, "_id": 1}}]

        for m_doc in (self.movies_collection.aggregate(pipeline_q2)):
            min_dst = 10000000
            cid=0
            for c_doc in (self.cent_collection.aggregate(pipeline_q3)):
                a = m_doc['kmeansnorm']
                b = c_doc['kmeansnorm']
                dst = distance.euclidean(b, a)
                if dst < min_dst:
                    min_dst = dst
                    cid  = c_doc["_id"]
            self.movies_collection.update_one({"_id": m_doc["_id"]}, {"$set": {"cluster": cid}})

        for c_doc in (self.cent_collection.aggregate(pipeline_q3)):
            cid = c_doc["_id"]
            pipeline_q4 = [{"$match": {"cluster": cid}},
                           {"$group": {"_id": None, "x_avg": {"$avg": { "$arrayElemAt": [ "$kmeansnorm", 0 ] }},
                                       "y_avg": {"$avg": { "$arrayElemAt": [ "$kmeansnorm", 1 ] }}}},
                           {"$project": {"x_avg": 1, "y_avg": 1}}]
            for m_doc in (self.movies_collection.aggregate(pipeline_q4)):
                x_avg = m_doc['x_avg']
                y_avg = m_doc['y_avg']
                kmnorm_updated = [x_avg, y_avg]
                self.cent_collection.update_one({"_id": cid}, {"$set": {"kmeansnorm": kmnorm_updated}})




    def kmeans_iterations(self):
        '''
        Run kmeans clustering for all the genres and for number of from 10 to 50 in steps of 5 and plot K vs SSE for each genre.
        :return:
        '''
        genrelist = [ "Action", "Horror", "Romance", "Sci-Fi", "Thriller"]
        self.movies_collection = self.database['Movies']
        self.cent_collection = self.database['Centroids']


        for g in range(0,len(genrelist)):
            k_sse={}
            for k in range(10,51, 5):
                iterations=0
                converged = False
                self.cent_collection.delete_many({})
                self.centroids(genrelist[g], k)
                old_centroid = []
                pipeline1 = [{"$project": {"kmeansnorm": 1, "_id": 0}}]
                for c_doc in (self.cent_collection.aggregate(pipeline1)):
                    kmnorm = c_doc['kmeansnorm']
                    old_centroid.append(kmnorm)
                while iterations<=1 or not converged:                            #changed no of iterations from 100 to 1 so that it's faster to test.
                    self.kmeans_onestep(genrelist[g])
                    new_centroid=[]
                    for c_doc in (self.cent_collection.aggregate(pipeline1)):
                        kmnorm = c_doc['kmeansnorm']
                        new_centroid.append(kmnorm)
                    a = np.matrix(old_centroid)
                    b=np.matrix(new_centroid)
                    diff = np.sum(a-b)
                    if diff == 0:
                        converged = True
                    old_centroid = new_centroid
                    iterations +=1
                sse = self.calculate_SSE(genrelist[g])
                k_sse[k] = sse
            labels, data = k_sse.keys(), k_sse.values()
            fig = plt.figure()
            plt.plot(labels, data)
            fig.suptitle(genrelist[g], fontsize=20)
            plt.xlabel('K', fontsize=18)
            plt.ylabel('SSE', fontsize=16)
            plt.show()




    def calculate_SSE(self,g):
        '''
        Calculate sum of squared errors for each genre and number of clusters k
        :param g: genre of the movie
        :return: None
        '''
        self.movies_collection = self.database['Movies']
        self.cent_collection = self.database['Centroids']

        pipeline_q2 = [{"$match": {"$and": [
            {"genres": g}, {"kmeansnorm": {"$exists": True}}, {"cluster": {"$exists": True}}]}},
            {"$project": {"kmeansnorm": 1, "cluster": 1}}]

        sse=0
        c_kmnorm=[[0,0]]
        for m_doc in (self.movies_collection.aggregate(pipeline_q2)):
            m_kmnorm = m_doc['kmeansnorm']
            m_cluster = m_doc['cluster']
            pipeline_q3 = [{"$match": {"_id": m_cluster}},{"$project": {"kmeansnorm":1, "_id":0}}]
            for c_doc in (self.cent_collection.aggregate(pipeline_q3)):
                c_kmnorm = c_doc['kmeansnorm']
            a=np.matrix(m_kmnorm)
            b=np.matrix(c_kmnorm)
            c = np.sum(a - b)
            sse +=c**2
        return sse



    def create_indexes(self):
        '''
        Function used to create indexes on MongoDB Fields for faster runtime of kmeans algorithm.
        :return: None
        '''
        self.movies_collection = self.database['Movies']
        self.cent_collection = self.database['Centroids']

        self.movies_collection.create_index([('genres', pymongo.TEXT)], name='search_index', default_language='english')
        self.movies_collection.create_index([('cluster', pymongo.ASCENDING)])
        self.movies_collection.create_index([('kmeansnorm', pymongo.ASCENDING)])
        self.cent_collection.create_index([('kmeansnorm', pymongo.ASCENDING)])




    def get_clustered_movies(self):
        '''
        Print sample 5 movies for each genre grouped together in a particular cluster.
        :return:
        '''
        genrelist = ["Action", "Horror", "Romance", "Sci-Fi", "Thriller"]

        #let's assume a random cluster, let's say cluster 4.

        self.movies_collection = self.database['Movies']

        for g in range(0,len(genrelist)):
            pipeline1 = [{"$match": {"$and": [
                {"genres": genrelist[g]}, {"kmeansnorm": {"$exists": True}}, {"cluster": {"$exists": True}}, {"cluster": 4}]}},
                {"$project": {"_id":1, "title":1, "startyear": 1, "avgrating": 1}}]
            count=0
            print("For genre "+ str(genrelist[g])+ " movies grouped together in cluster 4 are ")
            for m_doc in (self.movies_collection.aggregate(pipeline1)):
                print(m_doc)
                count+=1
                if count ==5:
                    print(" ")
                    break


if __name__ == '__main__':
    port = int(input("Enter port MongoDB's running on"))
    host = input("Enter host for both MongoDB")

    mongodb =MongoDBManagement(host,port)

    mongodb.create_indexes()

    start_time = time.time()
    mongodb.kmeansNorm()
    print("--- %s seconds for calculating kmeansnorm for all movies in movies collection ---" % (time.time() - start_time))

    start_time = time.time()
    mongodb.kmeans_iterations()
    print("--- %s seconds for running kmeans clustering algorithm on all genres  ---" % (time.time() - start_time))

    start_time = time.time()
    mongodb.get_clustered_movies()
    print("--- %s seconds for sampling movies of a similar cluster ---" % (time.time() - start_time))