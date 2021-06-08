import numpy as np
from sklearn.cluster import KMeans
from csv import reader
#   from scipy.stats import norm


# def points_reader():
#     with open("D:\Users\Mateusz\PL-II_st\I_semestr\PwOI\Wykład_3_Tydz_4\Dopasowanie powierzchni"
#               "LidarData_3.xyz", newline='') as csvfile:
#         reader(csvfile, delimiter=',')

def kmeans():
    clusterer = KMeans(n_clusters=3)
    X = np.array()
    pred = clusterer.fit_predict(X)
