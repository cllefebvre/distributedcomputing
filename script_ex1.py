######################## EX1 ######################################
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
scores=[]
names=['iris','sonar','votes']
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

for number in range(3):
    #split between train and test
    X_test, X_train, Y_test, Y_train = train_test_split(datasets[number], results[number], test_size=0.8, random_state=1234)
    neurons=[5,10,50]
    a=0
    scores_indiv=[]
    #We try the 3 layers proposed by the exercise
    for i in neurons:
        model=MLPClassifier(hidden_layer_sizes=(i,), activation='relu')
        t1=time.time()
        model.fit(X_train,Y_train)
        t2=time.time()
        training_time=t2-t1
        predictions=model.predict(X_test)
        scores_indiv.append([accuracy_score(predictions, Y_test),training_time])
    scores.append([names[number],scores_indiv])
    
print(scores)