#Dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

#sklearn
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import svm

from sklearn.linear_model import LogisticRegression



#Test training Kfold crossvalidation
from sklearn import cross_validation
from sklearn.cross_validation import KFold, train_test_split, cross_val_score



print "Hello world2"
sales_train = pd.read_csv('sales_train.csv',sep = ';',header=0,names=['accountName','saleDate','totaltransactions','avgprice','lastprice','differentip','lifespan','number_of_week','weekend','lastmethod','country','orderfreq','return'])
sales_test = pd.read_csv('sales_test.csv',sep = ';',header=0,names=['accountName','saleDate','totaltransactions','avgprice','lastprice','differentip','lifespan','number_of_week','weekend','lastmethod','country','orderfreq','return'])

attributes = sales_train.loc[:,['totaltransactions','avgprice','lastprice','differentip','lifespan','weekend','country','orderfreq','return']]
attributes = attributes.astype(float)

data_train = sales_train.loc[:,['totaltransactions','avgprice','lastprice','differentip','lifespan','weekend','country','orderfreq']]
target_train = sales_train['return']

data_test = sales_test.loc[:,['totaltransactions','avgprice','lastprice','differentip','lifespan','weekend','country','orderfreq']]

#print sales_test
x = data_train
y = target_train
z = data_test



#preprocessing data normalize and starndardize
x = preprocessing.normalize(x)
x = preprocessing.scale(x)

z = preprocessing.normalize(z)
z = preprocessing.scale(z)

#Feature selection
model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)


#Feature selection
model = LogisticRegression()
rfe = RFE(model, 3)
rfe = rfe.fit(x,y)
print (rfe.support_)
print (rfe.ranking_)

#define the training and test sets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1)
print x_test.shape


# Decisiontree algorithm
#train the model on test set
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
print(model)
# make predictions for the test set
expected = y_test
predicted = model.predict(x_test)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

sales_test['return'] = model.predict(z)
print(sales_test['return'].value_counts())


Kscore = cross_val_score(model, x, y, cv=10, scoring='accuracy')
print(Kscore)
print(Kscore.mean())

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators=10)
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(x_train, y_train)
print(clf)
expected = y_test
predicted = clf.predict(x_test)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

sales_test['return'] = clf.predict(z)
print(sales_test['return'].value_counts())

Kscore = cross_val_score(clf, x, y, cv=10, scoring='accuracy')
print(Kscore)
print(Kscore.mean())


#correlation matrix
from pandas.tools.plotting import scatter_matrix
plt.style.use('ggplot')
scatter = scatter_matrix(attributes, alpha=0.2, figsize=(6, 6), diagonal='kde')
#plt.show()


def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);

plot_corr(attributes)
plt.show()