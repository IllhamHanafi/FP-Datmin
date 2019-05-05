import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import operator

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

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
def getNeighbors(trainingSet, testData, testClass, k):
    #array distance untuk nyimpan jarak, nanti akan disort yang tedekat
    # print(trainingSet[0])
    distance = []
    length = len(testData) - 1
    for x in range(len(trainingSet)):
        dist = eucledianDistance(testData, trainingSet[x], length)
        distance.append((trainingSet[x], testClass[x], dist))
        # print((trainingSet[x], dist))
    distance.sort(key=operator.itemgetter(2))
    # print(distance)
    neighbors = []
    for x in range(k):
        #habis disort, data tetangga terdekat disimpan di array neighbors
        neighbors.append(distance[x][0:2])
    # print(neighbors)
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
    # print(sortedVotes[0][0])
    return sortedVotes[0][0]

#fungsu getAccuracy untuk mendapatkan nilai akurasi
#testData adalah kelas dari data yang dicek
#prediction adalah prediksi kelas dari data yang dicek
def getAccuracy(testData, predictions):
    correct = 0
    # print(len(testData))
    # print(predictions)
    for x in range(len(testData)):
        #jika prediksi benar, maka nilai correct bertambah +1
        if testData[x] == predictions[x]:
            correct += 1
    #mengembalikan nilai return persentase
    return (correct/float(len(testData))) * 100.0

#Load dataset
path = "dataset/iris.data"
# path = "dataset/transfusion.data"
# path = "dataset/bupa.data"

namesIris = [
    'sepal-length',
    'sepal-width',
    'petal-width',
    'petal-length',
    'class'
]

namesTransfusion = [
    'Recency',
    'Frequency',
    'Monetary',
    'Time',
    'Yes/No Donate'
]

namesBupa = [
    'mcv',
    'alkphos',
    'sgpt',
    'sgot',
    'gammagt',
    'drinks',
    'selector'
]
dataset = pd.read_csv(path, names=namesIris)


##Dataset preprocessing
#membagi jadi 2, X untuk nilai numerik, Y untuk nama kelas
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# print(X)
# print(Y)

#split dataset 20% train-80% test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#MinMaxNormalization
#Mengubah nilai numerik menjadi antara 0-1
#ini kalo gak pake fungsi library
# print(((x[:, 0] - x[:, 0].min()) / (x[:, 0].max() - x[:, 0].min())))
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
scaler.fit(x_test)
x_test = scaler.transform(x_test)

prediction = []
for i in range(len(x_test)):
    #mendapatkan tetangga terdekat
    neighbors = getNeighbors(x_train, x_test[i], y_train, 3)
    #mendapatkan kelas berdasarkan tetangga terdekat
    result = getResponse(neighbors)
    #memasukkan nilai kelas ke array prediction
    prediction.append(result)
    print('> predicted=' + repr(result) + ', actual=' + repr(y_test[i]))
    neighbors.clear()
#menghitung akurasi dengan mengecek antara akurasi dengan nama kelas yang sebenarnya
accuracy = getAccuracy(y_test, prediction)
prediction.clear()
# print(y_test)
# print(prediction)
print('Non-KFold Accuracy: ' + repr(accuracy) + "%")
fold = 1
KF_xtrain = []
KF_xtest = []
KF_ytrain = []
KF_ytest = []
kfold = KFold(n_splits=3, shuffle=True, random_state=True)
for train, test in kfold.split(X, Y):
    # print('train: %s, test: %s' % (X[train], X[test]))
    KF_xtrain.append(X[train])
    KF_xtest.append(X[test])
    KF_ytrain.append(Y[train])
    KF_ytest.append(Y[test])
    for j in range(len(KF_xtest[0])):
        neighbors = getNeighbors(KF_xtrain[0], KF_xtest[0][j], KF_ytrain[0], 3)
        result = getResponse(neighbors)
        prediction.append(result)
        neighbors.clear()
    accuracy = getAccuracy(KF_ytest[0], prediction)
    prediction.clear()
    # print(y_test)
    # print(prediction)
    print('KFold' + str(fold) + 'Accuracy: ' + repr(accuracy) + "%")
    fold += 1
    KF_xtest.clear()
    KF_xtrain.clear()
    KF_ytest.clear()
    KF_ytrain.clear()
    # print(train)
    # print(test)
    # print(KF_xtrain)
    # for j in range(len(test)):
        # KFold_neighbors = getNeighbors()