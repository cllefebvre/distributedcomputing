# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:56:52 2019

@author: lfbcl
"""
######################## EX2 ######################################
from sklearn.model_selection import train_test_split
import time
import pandas as pd
iris=pd.read_csv('iris_UCI.csv')
sonar=pd.read_csv('sonar_UCI.csv')
votes=pd.read_csv('votes_UCI.csv')

#Change dtypes to help my RAM
from compress import reduce_mem_usage
iris=reduce_mem_usage(iris)
sonar=reduce_mem_usage(sonar)
votes=reduce_mem_usage(votes)

#Split database in features and result
iris_Y=iris[' CLASS']
sonar_Y=sonar[' class']
votes_Y=votes['CLASS']
iris_X=iris.drop(columns=' CLASS')
sonar_X=sonar.drop(columns=' class')
votes_X=votes.drop(columns='CLASS')

#We must encode the text in the vote database to use a NN
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
Enc=LabelEncoder()
Encoded_X=votes_X.apply(Enc.fit_transform)
HotEnc=OneHotEncoder()
HotEnc.fit(Encoded_X)
onehot_encoded=HotEnc.transform(Encoded_X).toarray()
votes_X=pd.DataFrame(data=onehot_encoded)

#We process the 3 datasets with our NN
datasets=[iris_X,sonar_X,votes_X]
results=[iris_Y,sonar_Y,votes_Y]
names=['iris','sonar','votes']
activations=['relu','tanh']
#Here are our outputs
title_list=[]
score_list=[]
training_time_list=[]
activation_function_list=[]
fold_list=[]

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#First, we do not do any cross validation
for number in range(3):
    for activ in activations:
        #split between train and test
        X_test, X_train, Y_test, Y_train = train_test_split(datasets[number], results[number], test_size=0.8, random_state=1234)
        #We try the 3 layers proposed by the exercise
        model=MLPClassifier(hidden_layer_sizes=(5,20,100,), activation=activ)
        t1=time.time()
        model.fit(X_train,Y_train)
        t2=time.time()
        training_time=round(t2-t1,2)
        predictions=model.predict(X_test)
        title_list.append(names[number])
        score_list.append(accuracy_score(predictions, Y_test))
        training_time_list.append(training_time)
        activation_function_list.append(activ)
        fold_list.append(0)

#Then K-fold
from sklearn.model_selection import KFold
nsplits=[5,10]
for number in range(3):
    features=datasets[number]
    categories=results[number]
    for activ in activations :
        model=MLPClassifier(hidden_layer_sizes=(5,20,100,), activation=activ)
        for n_splits in nsplits:
            Kfold=KFold(n_splits=n_splits)
            score=0
            t1=time.time()
            for train_indices, test_indices in Kfold.split(features):
                X_train, X_test = features.iloc[train_indices], features.iloc[test_indices]
                Y_train, Y_test = categories.iloc[train_indices], categories.iloc[test_indices]
                mdl_trained=model.fit(X_train, Y_train)
                predictions=mdl_trained.predict(X_test)
                score+=accuracy_score(predictions,Y_test)/n_splits
            t2=time.time()
            training_time=round(t2-t1,2)
            score_list.append(score)
            training_time_list.append(training_time)
            title_list.append(names[number])
            activation_function_list.append(activ)
            fold_list.append(n_splits)
#Display results       
dico={'title':title_list,'activation':activation_function_list,'K fold':fold_list, 'scores':score_list,'training_time (s)':training_time_list}
results=pd.DataFrame(data=dico)
print(results)