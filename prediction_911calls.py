import numpy as np
import pandas as pd
#this function for ingoring warnings from sklearn
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#from imblearn.over_sampling import RandomOverSampler
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn.over_sampling import SMOTE


#this function simply reads the data file and initialize some information about the data
def initializeData():
    data = pd.read_csv('911.csv', sep=",", header=None, low_memory=False)
    data = data.iloc[1:, [0,1,4]]
    data.columns = ["Lat","Long","Label"]
    data = data.astype(dtype = {"Lat":"float64", "Long":"float64", "Label":"object"})
    return data

#this function reduces the class count by doing merging the classes
#for example: EMS: VEHICLE ACCIDENT and Traffic: VEHICLE ACCIDENT are reduced to VEHICLE ACCIDENT
def reduceClassCount(data):
    splittedColumn = data["Label"].str.split(": ", expand = True)
    againSplit = splittedColumn[1].str.split(" -", expand = True)
    data["Label"] = againSplit[0]

def dropDuplicateValues(data):
    data.drop_duplicates()
    
#this function encodes the class names to numbers
def encodeLabels(data):
    labelencoder = LabelEncoder()
    data["Label"] = labelencoder.fit_transform(data["Label"])

#this function scales the latitude and longtitude features
def scaleFeatures(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data[["Lat","Long"]] = scaler.fit_transform(data[["Lat","Long"]])

#this function returns the count of class type
def getClassNames(df):
    return df.Label.unique()

#this function returns the counts of each class type in the data
def getCountOfType(df):
    priors = []
    classNames = getClassNames(df)
    for i in classNames:
        priors.append([i,len(df[df.Label == i])])
    return priors

#this function returns features and labels as numpy array 
def getDataAsArray(data):
    features = data[["Lat","Long"]].values
    labels = data["Label"].values
    return features, labels

#base approach using naive bayes
def performBaseApproach(features,labels):
    clf = GaussianNB()
    print("Base approach: ", np.mean(cross_val_score(clf, features, labels, cv=5))*100)   #cross validation with 5 set

#knn classifier
def performKNN(features, labels, neighbors):
    clf = KNeighborsClassifier(n_neighbors = neighbors)
    print("KNN Classifier: ", np.mean(cross_val_score(clf, features, labels, cv=5))*100)

#decision tree classifier
def performDecisionTree(features,labels):
    clf = DecisionTreeClassifier()
    print("Decision Tree: ", np.mean(cross_val_score(clf, features, labels, cv=5))*100)

#merging classes according to relevance between classes
def mergeClasses(data):
    data['Label'] = data['Label'].replace(['CARDIAC ARREST','ANIMAL BITE','HEAT EXHAUSTION', 
                                           'ACTIVE SHOOTER','BOMB DEVICE FOUND','SUICIDE ATTEMPT',
                                           'STANDBY FOR ANOTHER CO','CHOKING','SEIZURES','CARBON MONOXIDE DETECTOR',
                                           'PUBLIC SERVICE','HIT + RUN','PRISONER IN CUSTODY'], 
                                          ['CARDIAC EMERGENCY','ANIMAL COMPLAINT','DEHYDRATION', 
                                           'SHOOTING','BOMB THREAT','SUICIDE THREAT', 'TRANSFERRED CALL',
                                           'RESPIRATORY EMERGENCY','CVA/STROKE','FIRE ALARM','RESCUE','STABBING',
                                           'SUSPICIOUS'])
    data['Label'] = data['Label'].replace(['WARRANT SERVICE','ARMED SUBJECT'], 
                                          ['SUSPICIOUS','SHOOTING'])
    data['Label'] = data['Label'].replace(['POLICE INFORMATION','SUICIDE THREAT'], 
                                          ['SUSPICIOUS','OVERDOSE'])

#applying over and under sampling to the training data
def applyOverAndUnderSampling(X_train, y_train):
    rus = RandomUnderSampler('all')
    ros = RandomOverSampler('not majority')
    x_s, y_s = ros.fit_sample(X_train, y_train)
    x_res, y_res = rus.fit_sample(x_s, y_s)
    return x_res,y_res

#applying SMOTE to the training data
def applySMOTE(X_train, y_train):
    smote = SMOTE('not majority')
    x_res, y_res = smote.fit_sample(X_train, y_train)
    return x_res,y_res

def main():
    data = initializeData()
    
    features, labels = getDataAsArray(data)
    
    print("Algorithms for raw data: ")
    performBaseApproach(features,labels)
    performKNN(features,labels,25)
    performDecisionTree(features,labels)
    
    reduceClassCount(data)
    features, labels = getDataAsArray(data)
    print("Algorithms after applying class reduction: ")

    performBaseApproach(features,labels)
    performKNN(features,labels,25)
    performDecisionTree(features,labels)
    
main()
#this loop stands for avoiding console disappearing
while True:
    takeInput = input("Hit x to quit")
    if takeInput.lower() == "x":
        break
