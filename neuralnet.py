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
import numpy
class NeuralNet:
    """ Takes in a data set and trains with it"""
    def __init__(self, validation_size, model, scoring):
        """This is the part where you tell it what kind of training you want to do"""
        self.validation_size = validation_size #0.2 = 20%
        self.dataset = numpy.empty
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

    def data(self, subset):
        """This part should add an item to the data to train from"""
        """Currently all this does is add data for training"""
        self.dataset = subset

    def train(self):
        """Does all the training all at once"""
        array = self.dataset
        X = array[:, 0:-1]
        Y = array[:, -1]
        '''
        This seperates the number from the actual data I think, number needs to be changed to meet the amount of data it is given(8?)
        For example List number E1, E2, E3,E4, E5, E6, E7, E8 Right/Left
        I think that I could probably write some code that used the length of the element in the list to do this automatically,
        but keeper will do that better
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
        self.model.fit(self.X_train, self.Y_train) # What does this do? Who knows! When I take it away here there are no problems, but in quality it causes crashes when absent
        predictions = self.model.predict(pdata)
        print(predictions)
        return (predictions)

    def validate(self, action, pdata):
        """So this takes the data you put it and sees if it made the correct guess"""
        return self.get(pdata)==action

'''def main():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pandas.read_csv(url, names=names)
    nn = NeuralNet(0.2, 'LR','accuracy')
    print (type(dataset.values))
    nn.add(dataset.values)
    nn.train()
    nn.quality()
    nn.get(dataset.head(1).values[:, 0:4]) #This needs to only reference the data itself, or I need to parse it in function
    print(dataset.head(1).values[:,4])
    """
    Good news! This works. Some bugs that are left
    A) We use validation data and I'm not sure if that is useful considering we are doing our own valiadation 
    B) There are a couple of elements that I am not sure how they work
    C) This only works for lists with the example amount of variables and fixing that is not my problem
    D) 
    """
main()'''