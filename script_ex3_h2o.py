
import sys
import h2o
h2o.init()

iris=h2o.import_file(path='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
votes=h2o.import_file(path='https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data')
sonar=h2o.import_file(path=r'C:\Users\lfbcl\OneDrive\Bureau\Paralel\sonar_UCI.csv')

#Split  into Train/Test/Validation
train_iris,test_iris = iris.split_frame(ratios=[.7])
train_votes,test_votes = votes.split_frame(ratios=[.7])
train_sonar,test_sonar = sonar.split_frame(ratios=[.7])
#The different datasets we want to test
x_iris=train_iris.names[0:4]
y_iris='C5'
x_votes=train_votes.names[1:17]
y_votes='C1'
x_sonar=train_sonar.names[0:59]
y_sonar='class'
datasets=[['iris',x_iris,y_iris,train_iris,test_iris],['votes',x_votes,y_votes,train_votes,test_votes],['sonar',x_sonar,y_sonar,train_sonar,test_sonar]]

#The parameters we are going to test
nfolds=[0,4,9]
activations=['Tanh', 'RectifierWithDropout']

#The outputs we want in our table
activation_final=[]
nfolds_final=[]
rmse_final=[]
title_final=[]

from h2o.estimators.deeplearning import H2ODeepLearningEstimator
for dataset in datasets:
    for nfold in nfolds :
        for activation_function in activations:
            model = H2ODeepLearningEstimator(
                                            hidden=[5,20,100],
                                            nfolds=nfold,
                                            activation=activation_function
                                            )
            model.train(x=dataset[1], y=dataset[2], training_frame=dataset[3])
            perf_iris=model.model_performance(dataset[4])
            #Outputs
            activation_final.append(model.activation)
            nfolds_final.append(model.nfolds+1)
            rmse_final.append(perf_iris.rmse())
            title_final.append(dataset[0])
            
#Now let's show our results
import pandas as pd
dico={'title':title_final,'activation': activation_final, 'nfolds':nfolds_final, 'rmse': rmse_final}
results=pd.DataFrame(data=dico)
print(results)