# -*- coding: utf-8 -*-
"""
Project: neurohacking
File: neuralnet.py
Author: kindler
"""
import random

import numpy
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas

class Model:
    LR = LogisticRegression
    LDA = LinearDiscriminantAnalysis
    KNN = KNeighborsClassifier
    CART = DecisionTreeClassifier
    NB = GaussianNB
    SVM = SVC

class NeuralNet:
    """ Inits the NN """
    def __init__(self, validation_size: int, model: Model, scoring: str):
        """This is the part where you tell it what kind of training you want to do"""
        self.validation_size = validation_size  # 0.2 = 20%
        self.dataset = numpy.zeros([8,0])#Idk if this works 10 is sample size
        self.model = model()
        self.scoring = scoring  # Possible options = only scoring I think

    def addData(self, subset):
        """This part should add an item to the data to train from"""
        """Currently all this does is add data for training"""
        self.dataset.append(subset)#subset better be dim 2

    def train(self, data: numpy.ndarray):
        """Does all the training all at once"""
        #I suspect this doesn't work
        assert len(data.shape) == 2
        X = data[:, 0:-1]
        Y = data[:, -1]

        seed = random.randint(1, 9)
        # This splits the data into training and validation
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = (
            model_selection.train_test_split(X, Y, test_size=self.validation_size, random_state=seed)
        )
        # Idk what n_splits does, but i think this is where it decides where to start training on the data
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        # And this is where it actually does the training
        cv_results = model_selection.cross_val_score(self.model, self.X_train, self.Y_train, cv=kfold, scoring=self.scoring)
        msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
        print(msg)

    def quality(self):
        self.model.fit(self.X_train, self.Y_train)
        predictions = self.model.predict(self.X_validation)
        print(accuracy_score(self.Y_validation, predictions))
        print(confusion_matrix(self.Y_validation, predictions))
        print(classification_report(self.Y_validation, predictions))

    def get(self, pdata) -> str:
        """So the theory behind this function is that it makes a prediction about what the data you gave it was"""
        self.model.fit(self.X_train, self.Y_train)  # What does this do? Who knows! When I take it away here there are no problems, but in quality it causes crashes when absent
        predictions = self.model.predict(pdata)
        print(predictions)
        return (predictions)

    def validate(self, action, pdata) -> bool:
        """So this takes the data you put it and sees if it made the correct guess"""
        return self.get(pdata) == action

def main():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pandas.read_csv(url, names=names)
    print (dataset)
    nn = NeuralNet(0.2, Model.LR,'accuracy')
    print (dataset.values.shape)
    nn.train(dataset.values)
    nn.quality()
    nn.get(dataset.head(1).values[:, 0:4]) #This needs to only reference the data itself, or I need to parse it in function
    print(dataset.head(1).values[:,4])


if __name__ == '__main__':
    main()
