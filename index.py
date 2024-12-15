import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier , ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier , BaggingClassifier , GradientBoostingClassifier , AdaBoostClassifier
from sklearn.metrics import accuracy_score
import pickle

crop = pd.read_csv("C:\Crop Recommender System\Crop_recommendation.csv")
#print(crop.head())
a = crop.shape
#print(a)
b = crop.info()
#print(b)
c = crop.isnull().sum()             #For checking null values
#print(c)
d = crop.duplicated().sum()
#print(d)
e = crop.describe()
#print(e)
f = crop.label.value_counts()       #counts of crops
#print(f)
g = crop['label'].unique().size      # No. of labels
#print(g)
h = crop.label.unique()              #Gives all the values for labe;
#print(h)
crop_dict = {
    'rice' : 1,
    'maize' : 2,
    'jute' : 3,         
    'cotton' : 4,
    'coconut' : 5,
    'papaya' : 6,      
    'orange' : 7,        
    'apple' : 8,      
    'muskmelon' : 9,
    'watermelon' : 10,
    'grapes' : 11,
    'mango' : 12,
    'banana' : 13,
    'pomegranate' : 14,
    'lentil' : 15,
    'blackgram' : 16,
    'mungbean' : 17,
    'mothbeans' : 18,
    'pigeonpeas' : 19,
    'kidneybeans' : 20,
    'chickpea' : 21,
    'coffee' : 22
}
crop['label'] = crop['label'].map(crop_dict)      # Converting the values into the labels
i = crop.head()
#print(i)
#print(crop.label.unique())
j = crop.label.value_counts()
#print(j)
x = crop.drop('label' , axis = 1)
y = crop['label']
k = x.head()
#print(k)
l = y.head()
#print(l)
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2 , random_state= 42)
m = x_train.shape
#print(m)
mx = MinMaxScaler()
x_train = mx.fit_transform(x_train)
x_test = mx.transform(x_test)
#print(x_train)
sc = StandardScaler()
sc.fit(x_train)
n_train = sc.transform(x_train)
n_test = sc.transform(x_test)
models = {
    'LogisticRegression' : LogisticRegression(),
    'GaussianNB' : GaussianNB(),
    'SVC' : SVC(),
    'KNeighborsClassifier' : KNeighborsClassifier(),
    'DecissionTreeClassifier' : DecisionTreeClassifier(),
    'ExtraTreeClassifier' : ExtraTreeClassifier(),
    'RandomForestClassifier' : RandomForestClassifier(),
    'BaggingClassifier' : BaggingClassifier(),
    'GradientBoostingClassifier' : GradientBoostingClassifier(),
    'AdaBoostClassifier' : AdaBoostClassifier()
}
for name , model in models.items():
    model.fit(x_train , y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test , y_pred)
    #print(f"{name} model with accuracy : {score}")

randclf = RandomForestClassifier()
randclf.fit(x_train , y_train)
y_pred = randclf.predict(x_test)
o = accuracy_score(y_test , y_pred)
#print(o)

p = crop.columns
#print(p)

def recommendations(N,P,K,temperature , humidity , ph , rainfall):
    features = np.array([[N,P,K,temperature , humidity , ph , rainfall]])
    mx_features = mx.fit_transform(features)
    sc_mx_features = sc.fit_transform(mx_features)
    prediction = randclf.predict(sc_mx_features).reshape(1 , -1)
    return prediction[0]

N = 90
P = 42
K = 43
temperature = 40.0
humidity = 22
ph = 7
rainfall = 202

predict = recommendations(N,P,K,temperature , humidity , ph , rainfall)
#print(predict)

pickle.dump(randclf  , open('model.pkl' , 'wb'))
pickle.dump(mx  , open('minmaxscaler.pkl' , 'wb'))
pickle.dump(sc  , open('standscaler.pkl' , 'wb'))
