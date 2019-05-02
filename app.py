import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import operator

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#fungsi buat dapetin jarak eucledian
#data1 = data pertama
#data2 = data pembanding
#length = jumlah variabel yang dibandingkan
def eucledianDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += pow((float(data1[x]) - float(data2[x])), 2)
        # print(str(data1) + "-" + str(data2) + ": " + str(distance))
    return math.sqrt(distance)

#fungsi untuk mendapatkan tetangga terdekat
#trainingset = data training / ground truth
#testdata = data testing / data yang diuji
#k = tetangga
def getNeighbors(trainingSet, testData, k):
    #array distance untuk nyimpan jarak, nanti akan disort yang tedekat
    distance = []
    length = len(testData)
    for x in range(len(trainingSet)):
        dist = eucledianDistance(testData, trainingSet[x], length)
        distance.append((trainingSet[x], dist))
        # print((trainingSet[x], dist))
    distance.sort(key=operator.itemgetter(1))
    # print(distance)
    neighbors = []
    for x in range(k):
        #habis disort, data tetangga terdekat disimpan di array neighbors
        neighbors.append(distance[x][0])
    # print(trainingSet[x])
    return neighbors

#fungsi getResponse buat nyari datatest masuk kelas mana / voting kelasnya
def getResponse(neighbors):
    #classvote disiapin buat penampung proses voting kelas
    classVote = {}
    # print(neighbors)
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        # print(response)
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1
    # print(classVote)
    # sortedVotes buat milih vote kelas terbanyak
    sortedVotes = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b'], [5, 5, 5, 'b'], [5.5, 5.5, 5.5, 'b'], [5.1, 5.2, 5.1, 'a']]
testData = [5, 5, 5]
k = 3
neighbors = getNeighbors(trainSet, testData, k)
# print(neighbors)

response = getResponse(neighbors)
print("testData termasuk ke dalam kelas: " + response)

# path = "dataset/iris.data"
#
# names = [
#     'sepal-length',
#     'sepal-width',
#     'petal-width',
#     'petal-length',
#     'class'
# ]
#
# dataset = pd.read_csv(path, names=names)
# dataset.dropna(how="all", inplace=True)
#
# x = dataset.iloc[:, 0:4].values
# y = dataset.iloc[:, 4].values
#
# print(x)
# print(((x[:, 0] - x[:, 0].min()) / (x[:, 0].max() - x[:, 0].min())))
#
# scaler = MinMaxScaler()
# print(scaler.fit(x))
# print(scaler.transform(x))
#
#
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
#
# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors=5)
# classifier.fit(x_train, y_train)
#
# y_pred = classifier.predict(x_test)
#
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
