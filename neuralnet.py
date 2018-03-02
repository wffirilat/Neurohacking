# -*- coding: utf-8 -*-
"""
Project: neurohacking
File: neuralnet.py
Author: kindler
"""
import random
import pandas
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
class NeuralNet:
    """ Takes in a data set and trains with it"""
    def __init__(self, dataset, validation_size, model, scoring):
        """This is the part where you tell it what kind of training you want to do"""
        self.dataset = dataset #Must be in a
        self.validation_size = validation_size #0.2 = 20%
        if model == "LR":
            self.model = LogisticRegression()
        elif model == "LDA":
            self.model = LinearDiscriminantAnalysis()
        elif model == "KNN":
            self.model = KNeighborsClassifier()
        elif model == "CART":
            self.model = DecisionTreeClassifier()
        elif model == "NB":
            self.model =GaussianNB()
        elif model == "SVM":
            self.model =SVC()
        self.scoring = scoring #Possible options = only scoring I think

    def add(self, subset):
        """This part should add an item to the data to train from"""
        self.dataset.add(subset)

    def train(self):
        """Does all the training all at once"""
        array = self.dataset.values
        X = array[:, 0:4]
        Y = array[:, 4]
        '''
        This seperates the number from the actual data I think, number needs to be changed to meet the amount of data it is given(8?)
        For example List number E1, E2, E3,E4, E5, E6, E7, E8 Right/Left
        '''
        seed = random.randint(1,9)
        # This splits the data into training and validation
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = model_selection.train_test_split(X, Y, test_size=self.validation_size, random_state=seed)
        # Idk what n_splits does, but i think this is where it decides where to start training on the data
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        #And this is where it actually does the training
        cv_results = model_selection.cross_val_score(self.model, self.X_train, self.Y_train, cv=kfold, scoring=self.scoring)
        msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
        print(msg)
    def quality(self):
        self.model.fit(self.X_train, self.Y_train)
        predictions = self.model.predict(self.X_validation)
        print(accuracy_score(self.Y_validation, predictions))
        print(confusion_matrix(self.Y_validation, predictions))
        print(classification_report(self.Y_validation, predictions))
    def get(self, pdata):
        """So the theory behind this function is that it makes a prediction about what the data you gave it was"""
        self.model.fit(self.X_train, self.Y_train) # What does this do? Who knows! What happens when I take it away?
        predictions = self.model.predict(pdata)
        return (predictions)

    def validate(self, action, pdata):
        """So this takes the data you put it and sees if it made the correct guess"""
        return self.get(pdata)==action
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
nn = NeuralNet(dataset, 0.2, 'LR','accuracy')
nn.train()
nn.quality()
#nn.get(dataset.head(1)) This needs to only reference the data itself, or I need to parse it in function
"""
Good news! This works. Problem is it doesn't work the way that we want it to
I am not sure if splitting into validation data is useful if we are going to run our own tests. I mean I'm happy to move the code which currently
in get to there if we want... but it probably deserves its own funciton
"""