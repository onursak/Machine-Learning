from math import sqrt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d.axes3d import Axes3D

#Defining prior probabilities of each class
prior_C1 = 59/178
prior_C2 = 71/178
prior_C3 = 48/178


def predict(parametersOfClasses, testSet):
    global prior_C1,prior_C2,prior_C3
    true_values = [i for i in testSet['Class']]
    samples = testSet.iloc[:, [1,2]]
    sample_list = [list(samples.iloc[i]) for i in range(len(samples))] #collecting sample vectors
    predictions = []
    for i in sample_list:
        discriminants = []
        #calculating discriminants for all classes and add them into list for finding max of them
        discriminants.append(calculateDiscriminant(i, parametersOfClasses[0][0], parametersOfClasses[0][1], prior_C1))
        discriminants.append(calculateDiscriminant(i, parametersOfClasses[1][0], parametersOfClasses[1][1], prior_C2))
        discriminants.append(calculateDiscriminant(i, parametersOfClasses[2][0], parametersOfClasses[2][1], prior_C3))
        predictions.append(discriminants.index(max(discriminants))+1) 
    return (accuracy_score(true_values, predictions)*100) 

#calculating posterior by using bivariate gaussian distribution formula
def posterior(x, nu, cov):
    dot1 = np.dot(np.subtract(x,nu),np.linalg.inv(cov))    #matrix multiplication 1
    dot2 = np.dot(dot1, np.subtract(x,nu))                 #matrix multiplication 2
    return ((np.exp(-1/2*dot2))/(np.pi*sqrt(np.linalg.det(cov))))

#calculating discriminant by using unnormalized posterior
def calculateDiscriminant(x, nu, cov, prior):
    return posterior(x,nu,cov)*prior

#estimating parameters by using maximum likelihood estimation
def estimateParameters(classData):
    nu_vector = np.array([np.mean(classData.iloc[:,1]), np.mean(classData.iloc[:,2])])
    covariance_matrix = (np.cov(classData.iloc[:,1], classData.iloc[:,2]))
    return (nu_vector, covariance_matrix)

def plotGaussian(nu, cov, title):
    x = np.linspace(nu[0]-2.1, nu[0]+2.1 , 400) 
    y = np.linspace(nu[1]-2.1, nu[1]+2.1, 400)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    multi_variate = multivariate_normal(nu, cov)
    Z = multi_variate.pdf(pos)  #calculating z values by using probability density function for multivariate gaussian
    figure = plt.figure()
    a = figure.add_subplot(111, projection='3d')
    a.set_title(title)
    a.plot_surface(X, Y, Z, cmap='winter')
    figure.show()

#I used this function for feature selection by observing distribution of training data
#It simply visualize the data samples onto two dimension
def plotOfFeatures(train_data):
    plt.scatter(train_data[0]['Feature1'], train_data[0]['Feature2'], c='blue') #for class1
    plt.scatter(train_data[1]['Feature1'], train_data[1]['Feature2'], c='red') #for class2
    plt.scatter(train_data[2]['Feature1'], train_data[2]['Feature2'], c='green') #for class3
    plt.show()


def main():
    data = pd.read_csv('wine.data.txt', sep=",", header=None)
    data = data.iloc[:, [0,1,7]]   #Feature selection(I have selected Alcohol and Flavanoids columns) 
    data.columns = ["Class","Feature1","Feature2"]
    
    train_data=data.sample(frac=0.8)
    test_data=data.drop(train_data.index)
    
    #Partitioning train data according to their classes
    c1_train = train_data.loc[train_data['Class'] == 1]
    c2_train = train_data.loc[train_data['Class'] == 2]
    c3_train = train_data.loc[train_data['Class'] == 3]
    
    #Parameters: nu and cov matrix
    c1_parameters = estimateParameters(c1_train) 
    c2_parameters = estimateParameters(c2_train)
    c3_parameters = estimateParameters(c3_train)
    parameters = [c1_parameters,c2_parameters,c3_parameters]
    
    #plotOfFeatures([c1_train,c2_train,c3_train])
    
    #Sending all these nu and cov matrices to the predict function and predict according to test data
    accuracy_score = predict(parameters, test_data)
    
    print("Accuracy score is: ",accuracy_score)
    
    plotGaussian(c1_parameters[0],c1_parameters[1],"Class1 Gaussian Distribution")
    plotGaussian(c2_parameters[0],c2_parameters[1],"Class2 Gaussian Distribution")
    plotGaussian(c3_parameters[0],c3_parameters[1],"Class3 Gaussian Distribution")

main()

while True:    #This loop to avoid disappearing plots from the screen
    q_input = input("Please enter q for quit")
    if q_input.lower() == "q":
        break