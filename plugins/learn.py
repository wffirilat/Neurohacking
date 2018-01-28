# -*- coding: utf-8 -*-
"""
Project: neurohacking
File: learn.py
Author: kindler
"""

import numpy as np
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import plugin_interface as plugintypes
from open_bci_v3 import OpenBCISample

class PluginLearn(plugintypes.IPluginExtended):
    def __init__(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        dataset = pandas.read_csv(url, names=names)
        # dataset.plot(kind = 'box', subplots=True, layout=(2,2), sharex=False)
        # dataset.hist()
        # scatter_matrix(dataset)
        array = dataset.values
        X = array[:, 0:4]
        Y = array[:, 4]
        validation_size = 0.20
        seed = 7
        scoring = 'accuracy'
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                        random_state=seed)
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('LDR', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))
        results = []
        names = []
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
            results.append(cv_results)
            names.append(name)
            msg = "%s:%f(%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
        fig = plt.figure()
        fig.suptitle('Alg comp')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        # plt.show()
        knn = KNeighborsClassifier()
        knn.fit(X_train, Y_train)
        predictions = knn.predict(X_validation)
        print(accuracy_score(Y_validation, predictions))
        print(confusion_matrix(Y_validation, predictions))
        print(classification_report(Y_validation, predictions))
        self.packetnum = -1
        self.ticknum = None
        self.storelength = 1024
        self.rawdata = np.zeros((8, self.storelength))
        self.data = np.zeros((8, self.storelength))

    def activate(self):
        print("learning starting")

    # called with each new sample
    def __call__(self, sample: OpenBCISample):
        if sample.id == 0:
            self.packetnum += 1
        self.rawdata[:, (sample.id + 256 * self.packetnum) % self.storelength] = sample.channel_data
        self.data[:, (sample.id + 256 * self.packetnum) % self.storelength] = [v - avg for avg, v in zip(
            [sum(self.rawdata[i, :]) / self.storelength for i in range(8)],
            sample.channel_data
        )]
        #self.print(sample)

    def minmax(self, sample):
        print(min(self.data[3, :]), max(self.data[3, :]))

    def print(self, sample):
        print(self.data[3,-1])

    def thresholdDetect(self, sample):
        if self.data[3, (sample.id + 256 * self.packetnum) % self.storelength] > 1000:
            print((sample.id + 256 * self.packetnum))