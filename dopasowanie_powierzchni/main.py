import numpy as np # np- 'alias'
from scipy.stats import norm  # funckja do generacji punktów w przestrzeni o argumentach loc,scale
from csv import writer, reader # zapis i odczyt pliku csv
import random # losowanie wartości z zakresu

from math import sin, cos, sqrt, pi # funkcje do generacji punktów w obrycie walcowatym

import matplotlib.pyplot as plot_fig # funkcja graficzna

import pyransac3d
from sklearn.cluster import KMeans


#  generacja punktów w płaszczyźnie poziomej Y
def generate_points_Y(num_points: int = 2000, x_loc=-300, y_loc=-300):
    distribution_x = norm(loc=x_loc, scale=20)
    distribution_y = norm(loc=y_loc, scale=200)
    distribution_z = norm(loc=0.2, scale=0.05)

    x = distribution_x.rvs(size=num_points)
    y = distribution_y.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)

    points = zip(x, y, z) # sklejenie współrzędnych x,y,z w jeden punkt
    return points


#  generacja punktów w płaszczyźnie pionowej Z
def generate_points_Z(num_points: int = 2000, x_loc=0, z_loc=0):
    distribution_x = norm(loc=x_loc, scale=20)
    distribution_y = norm(loc=0, scale=0.05)
    distribution_z = norm(loc=z_loc, scale=200)

    x = distribution_x.rvs(size=num_points)
    y = distribution_y.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)

    points = zip(x, y, z)
    return points


#  generowanie punktów w obrycie cylindra
def generate_points_C(num_points: int = 5000, x_center: int = 400, y_center: int = 400, radius: int = 10):
    x = []
    y = []
    z = []
    for i in range(num_points):
        phi = random.uniform(0, 2 * pi)
        C_area = radius**2 * pi # **2 - rise to the power 2
        length = sqrt(random.uniform(0, C_area/pi))
        x.append(x_center + length * cos(phi))
        y.append(y_center + length * sin(phi))
        z.append(random.randrange(0, 200))

    points = zip(x, y, z)
    return points

# zapis chmur punktów do pliku csv
if __name__ == '__main__':     # część bloku wykonana jeśli skrypt jest jako 'main'
    cloud_points = generate_points_Y(2000, 0, 0)
    with open('LidarData.xyz', 'a', encoding='utf-8', newline='\n') as csvfile:
        csvwriter = writer(csvfile)
        for p in cloud_points:
            csvwriter.writerow(p)

    cloud_points = generate_points_Z(3000, 500, 200)
    with open('LidarData.xyz', 'a', encoding='utf-8', newline='\n') as csvfile:
        csvwriter = writer(csvfile)
        for p in cloud_points:
            csvwriter.writerow(p)

    cloud_points = generate_points_C(5000, -300, -300, 20)
    with open('LidarData.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
        csvwriter = writer(csvfile)
        for p in cloud_points:
            csvwriter.writerow(p)

    read_LidarData = [] # odczyt chmur z pliku csv
    with open("LidarData.xyz", newline='') as csvfile:
        r = reader(csvfile, delimiter=',')
        for line in r:
            read_LidarData.append([float(line[0]), float(line[1]), float(line[2])])
        print(read_LidarData)

    #Klasteryzacja
    clusterer = KMeans(n_clusters=3)
    X = np.array(read_LidarData)
    pred = clusterer.fit_predict(X)
    yellow = pred == 0
    blue = pred == 1
    green = pred == 2
    plot_fig.figure()
    plot_fig.scatter(X[yellow, 0], X[yellow, 1], c="y")
    plot_fig.scatter(X[blue, 0], X[blue, 1], c="b")
    plot_fig.scatter(X[green, 0], X[green, 1], c="g")
    plot_fig.show()

    #   Dopasowanie płaszczyzn - Ransac
    R_cloud = np.array(read_LidarData)
    plane = pyransac3d.Plane()
    equation, inliers = plane.fit(R_cloud, thresh=0.01, minPoints=100, maxIteration=1000)

    print(f'best inliers:{inliers}')
    print(f'plane equation:{equation}')

