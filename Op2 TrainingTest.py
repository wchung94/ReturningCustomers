
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

import datetime as dt

print "Hello world"

sales = pd.read_csv('sales.csv',sep = ';',header=0,names=['saleID','saleDateTime','accountName','amount','priceCurrency','price','currency','methodId','ip','ipCountry'])
sales['ipCountry'] = pd.Categorical.from_array(sales['ipCountry']).codes
sales['saleDateTime'] = sales['saleDateTime'].astype('datetime64[ns]')
sales['saleDate'] = pd.DatetimeIndex(sales['saleDateTime']).normalize().year

sales_train = sales[sales['saleDate'] < 2014]
sales_test = sales[sales['saleDate'] > 2013]

sales_train = sales_train.reset_index()
sales_test = sales_test.reset_index()

sales_train['totalamount'] = sales_train.groupby('accountName')['amount'].transform('sum') #voeg totalamount aan originele tabel toe
sales_train['totaltransactions'] = sales_train.groupby('accountName')['amount'].transform('count') #voeg totaltransactions aan originele tabel toe
sales_train['totalprice'] = sales_train.groupby('accountName')['price'].transform('sum') #voeg totalprice aan originele tabel toe
sales_train['avgprice'] = sales_train.groupby('accountName')['price'].transform('mean') #voeg avg price per transaction per customer aan tabel toe
sales_train['lastprice'] = sales_train.groupby('accountName')['price'].transform('last') #voeg land van laatste aankoop
sales_train['differentip'] = sales_train.groupby('accountName')['ip'].transform('nunique') #voeg different IP's aan originele tabel toe
sales_train['firsttime'] = sales_train.groupby('accountName')['saleDateTime'].transform('min') #voeg eerste aankoopdatum aan originele tabel toe
sales_train['lasttime'] = sales_train.groupby('accountName')['saleDateTime'].transform('max') #voeg laatste aankoopdatum aan originele tabel toe
sales_train['lifespan'] = 1 + (sales_train['lasttime'] - sales_train['firsttime']).dt.days #voeg de leeftijdsspan in dagen toe
sales_train['number_of_week'] = sales_train['lasttime'].dt.dayofweek #voeg nummer van de week van laatste aankoop
'''
days = {0:'Mon',1:'Tues',2:'Weds',3:'Thurs',4:'Fri',5:'Sat',6:'Sun'} #verander nummer naar dagen van de week converter
sales['day_of_week'] = sales['number_of_week'].apply(lambda x: days[x]) #voeg dagvan de week van laatste aankoop
'''
sales_train['weekend'] = np.where(sales_train['number_of_week'] > 4, 1, 0) #voeg toe of laatste aankoopdatum in weekend was

sales_train['lastmethod'] = sales_train.groupby('accountName')['methodId'].transform('last') #voeg gebruikte methode van laatste aankoop
sales_train['country'] = sales_train.groupby('accountName')['ipCountry'].transform('last') #voeg land van laatste aankoop

sales_train['orderfreq'] = (sales_train['lifespan']/(sales_train['totaltransactions']))

sales_train['return'] = sales_train['accountName'].isin(sales_test['accountName']) #voeg toe of of de klant al een returning customer is
sales_train['return'] = sales_train['return'].apply(lambda x: 1 if x else 0)

sales_train.drop_duplicates(subset ='accountName', inplace = True) #verwijder alle duplicates van accountName.


sales2 = sales_train.ix[:,['accountName','saleDate','totaltransactions','avgprice','lastprice','differentip','lifespan','number_of_week','weekend','lastmethod','country','orderfreq','return']]
sales2.to_csv('sales_train.csv',sep=';',encoding='utf-8')


sales_test['totalamount'] = sales_test.groupby('accountName')['amount'].transform('sum') #voeg totalamount aan originele tabel toe
sales_test['totaltransactions'] = sales_test.groupby('accountName')['amount'].transform('count') #voeg totaltransactions aan originele tabel toe
sales_test['totalprice'] = sales_test.groupby('accountName')['price'].transform('sum') #voeg totalprice aan originele tabel toe
sales_test['avgprice'] = sales_test.groupby('accountName')['price'].transform('mean') #voeg avg price per transaction per customer aan tabel toe
sales_test['lastprice'] = sales_test.groupby('accountName')['price'].transform('last') #voeg land van laatste aankoop
sales_test['differentip'] = sales_test.groupby('accountName')['ip'].transform('nunique') #voeg different IP's aan originele tabel toe
sales_test['firsttime'] = sales_test.groupby('accountName')['saleDateTime'].transform('min') #voeg eerste aankoopdatum aan originele tabel toe
sales_test['lasttime'] = sales_test.groupby('accountName')['saleDateTime'].transform('max') #voeg laatste aankoopdatum aan originele tabel toe
sales_test['lifespan'] = 1 + (sales_test['lasttime'] - sales_test['firsttime']).dt.days #voeg de leeftijdsspan in dagen toe
sales_test['number_of_week'] = sales_test['lasttime'].dt.dayofweek #voeg nummer van de week van laatste aankoop
'''
days = {0:'Mon',1:'Tues',2:'Weds',3:'Thurs',4:'Fri',5:'Sat',6:'Sun'} #verander nummer naar dagen van de week converter
sales['day_of_week'] = sales['number_of_week'].apply(lambda x: days[x]) #voeg dagvan de week van laatste aankoop
'''
sales_test['weekend'] = np.where(sales_test['number_of_week'] > 4, 1, 0) #voeg toe of laatste aankoopdatum in weekend was
sales_test['lastmethod'] = sales_test.groupby('accountName')['methodId'].transform('last') #voeg gebruikte methode van laatste aankoop
sales_test['country'] = sales_test.groupby('accountName')['ipCountry'].transform('last') #voeg land van laatste aankoop
sales_test['orderfreq'] = (sales_test['lifespan']/(sales_test['totaltransactions']))

sales_test['return'] = sales_test['accountName'].isin(sales_train['accountName']) #voeg toe of of de klant al een returning customer is
sales_test['return'] = sales_test['return'].apply(lambda x: 1 if x else 0)

sales_test.drop_duplicates(subset ='accountName', inplace = True) #verwijder alle duplicates van accountName.


sales3 = sales_test.ix[:,['accountName','saleDate','totaltransactions','avgprice','lastprice','differentip','lifespan','number_of_week','weekend','lastmethod','country','orderfreq','return']]
sales3.to_csv('sales_test.csv',sep=';',encoding='utf-8')




