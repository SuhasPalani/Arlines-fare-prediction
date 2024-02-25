import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_data = pd.read_excel("Data_Train.xlsx")

train_data.head(4)



train_data.tail(4)

train_data.info()


train_data.isnull().sum()

train_data['Total_Stops'].isnull()

train_data[train_data['Total_Stops'].isnull()]

train_data.dropna(inplace=True)

train_data.isnull().sum()

train_data.dtypes

data = train_data.copy()

data.columns
data.head(2)
data.dtypes

def change_into_Datetime(col):
    data[col] = pd.to_datetime(data[col])

import warnings 
from warnings import filterwarnings
filterwarnings("ignore")

data.columns

for feature in ['Dep_Time', 'Arrival_Time' , 'Date_of_Journey']:
    change_into_Datetime(feature)
data.dtypes

data["Journey_day"] = data['Date_of_Journey'].dt.day

data["Journey_month"] = data['Date_of_Journey'].dt.month

data["Journey_year"] = data['Date_of_Journey'].dt.year

data.head(3)


def extract_hour_min(df , col):
    df[col+"_hour"] = df[col].dt.hour
    df[col+"_minute"] = df[col].dt.minute
    return df.head(3)

data.columns

extract_hour_min(data , "Dep_Time")
extract_hour_min(data , "Arrival_Time")


cols_to_drop = ['Arrival_Time' , "Dep_Time"]

data.drop(cols_to_drop , axis=1 , inplace=True )

data.head(3)

data.shape


data.columns


def flight_dep_time(x):
    '''
    This function takes the flight Departure time 
    and convert into appropriate format.
    
    '''
    
    if (x>4) and (x<=8):
        return "Early Morning"
    
    elif (x>8) and (x<=12):
        return "Morning"
    
    elif (x>12) and (x<=16):
        return "Noon"
    
    elif (x>16) and (x<=20):
        return "Evening"
    
    elif (x>20) and (x<=24):
        return "Night"
    
    else:
        return "late night"


data['Dep_Time_hour'].apply(flight_dep_time).value_counts().plot(kind="bar" , color="g")



import plotly
import cufflinks as cf
from cufflinks.offline import go_offline
from plotly.offline import plot , iplot , init_notebook_mode , download_plotlyjs
init_notebook_mode(connected=True)
cf.go_offline()


data['Dep_Time_hour'].apply(flight_dep_time).value_counts().iplot(kind="bar")

data.head(3)

def preprocess_duration(x):
    if 'h' not in x:
        x = '0h' + ' ' + x
    elif 'm' not in x:
        x = x + ' ' +'0m'
        
    return x

data['Duration'] = data['Duration'].apply(preprocess_duration)

data['Duration']

data['Duration'][0]

'2h 50m'.split(' ')

'2h 50m'.split(' ')[0]

'2h 50m'.split(' ')[0][0:-1]

type('2h 50m'.split(' ')[0][0:-1])

int('2h 50m'.split(' ')[0][0:-1])

int('2h 50m'.split(' ')[1][0:-1])

data['Duration_hours'] = data['Duration'].apply(lambda x : int(x.split(' ')[0][0:-1]))

data['Duration_mins'] = data['Duration'].apply(lambda x : int(x.split(' ')[1][0:-1]))

data.head(2)

pd.to_timedelta(data["Duration"]).dt.components.hours


data["Duration_hour"] = pd.to_timedelta(data["Duration"]).dt.components.hours

data["Duration_minute"] = pd.to_timedelta(data["Duration"]).dt.components.minutes



data['Duration'] ## convert duration into total minutes duration ..


'2*60'

eval('2*60')


data['Duration_total_mins'] = data['Duration'].str.replace('h' ,"*60").str.replace(' ' , '+').str.replace('m' , "*1").apply(eval)

data['Duration_total_mins']

data.columns

sns.scatterplot(x="Duration_total_mins" , y="Price" , data=data)

sns.lmplot(x="Duration_total_mins" , y="Price" , data=data)

sns.scatterplot(x="Duration_total_mins" , y="Price" , hue="Total_Stops", data=data)

data['Airline']=='Jet Airways'
data[data['Airline']=='Jet Airways'].groupby('Route').size().sort_values(ascending=False)

data.columns

sns.boxplot(y='Price' , x='Airline' , data=data.sort_values('Price' , ascending=False))
plt.xticks(rotation="vertical")
plt.show()

data.head(2)


cat_col = [col for col in data.columns if data[col].dtype=="object"]

num_col = [col for col in data.columns if data[col].dtype!="object"]


cat_col

data['Source'].unique()

data['Source'].apply(lambda x : 1 if x=='Banglore' else 0)


for sub_category in data['Source'].unique():
    data['Source_'+sub_category] = data['Source'].apply(lambda x : 1 if x==sub_category else 0)

data.head(3)

cat_col

data.head(2)

data['Airline'].nunique()

data.groupby(['Airline'])['Price'].mean().sort_values()

airlines = data.groupby(['Airline'])['Price'].mean().sort_values().index

airlines

dict_airlines = {key:index for index , key in enumerate(airlines , 0)}

dict_airlines

data['Airline'] = data['Airline'].map(dict_airlines)
data['Airline']

data.head(3)

data['Destination'].unique()

data['Destination'].replace('New Delhi' , 'Delhi' , inplace=True)


data['Destination'].unique()

dest = data.groupby(['Destination'])['Price'].mean().sort_values().index

dest

dict_dest = {key:index for index , key in enumerate(dest , 0)}


dict_dest

data['Destination'] = data['Destination'].map(dict_dest)

data['Destination']

data.head(3)

data.head(3)

data['Total_Stops']


data['Total_Stops'].unique()

stop = {'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}

data['Total_Stops'] = data['Total_Stops'].map(stop)

data['Total_Stops']


data.head(1)

data.columns


data['Additional_Info'].value_counts()/len(data)*100

data.head(4)


data.columns

data['Journey_year'].unique()

data.drop(columns=['Date_of_Journey' , 'Additional_Info' , 'Duration_total_mins' , 'Source' , 'Journey_year'] , axis=1 , inplace=True)

data.columns


data.head(4)

data.drop(columns=['Route'] , axis=1 , inplace=True)

data.head(3)


data.drop(columns=['Duration'] , axis=1 , inplace=True)

data.head(3)

def plot(df, col):
    fig , (ax1 , ax2 , ax3) = plt.subplots(3,1)
    
    sns.distplot(df[col] , ax=ax1)
    sns.boxplot(df[col] , ax=ax2)
    sns.distplot(df[col] , ax=ax3 , kde=False)

plot(data , 'Price')


q1 = data['Price'].quantile(0.25)
q3 = data['Price'].quantile(0.75)

iqr = q3- q1

maximum = q3 + 1.5*iqr
minimum = q1 - 1.5*iqr

print(maximum)


print(minimum)


print([price for price in data['Price'] if price> maximum or price<minimum])

len([price for price in data['Price'] if price> maximum or price<minimum])


data['Price'] = np.where(data['Price']>=35000 , data['Price'].median() , data['Price'])

plot(data , 'Price')

X = data.drop(['Price'] , axis=1)

y = data['Price']


from sklearn.feature_selection import mutual_info_regression


imp = mutual_info_regression(X , y)


imp


imp_df = pd.DataFrame(imp , index=X.columns)

imp_df.columns = ['importance']


imp_df

imp_df.sort_values(by='importance' , ascending=False)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.25, random_state=42)

from sklearn.ensemble import RandomForestRegressor

ml_model = RandomForestRegressor()

ml_model.fit(X_train , y_train)

y_pred = ml_model.predict(X_test)

y_pred


from sklearn import metrics

metrics.r2_score(y_test , y_pred)


import pickle

file = open(r'rf_random.pkl' , 'wb')

pickle.dump(ml_model , file)
model = open(r'rf_random.pkl' , 'rb')

forest = pickle.load(model)

y_pred2 = forest.predict(X_test)

metrics.r2_score(y_test , y_pred2)

def mape(y_true , y_pred):
    y_true , y_pred = np.array(y_true) , np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape(y_test , y_pred)

from sklearn import metrics

def predict(ml_model):
    model = ml_model.fit(X_train , y_train)
    print('Training score : {}'.format(model.score(X_train , y_train)))
    y_predection = model.predict(X_test)
    print('predictions are : {}'.format(y_predection))
    print('\n')
    r2_score = metrics.r2_score(y_test , y_predection)
    print('r2 score : {}'.format(r2_score))
    print('MAE : {}'.format(metrics.mean_absolute_error(y_test , y_predection)))
    print('MSE : {}'.format(metrics.mean_squared_error(y_test , y_predection)))
    print('RMSE : {}'.format(np.sqrt(metrics.mean_squared_error(y_test , y_predection))))
    print('MAPE : {}'.format(mape(y_test , y_predection)))
    sns.distplot(y_test - y_predection)

predict(RandomForestRegressor())

from sklearn.tree import DecisionTreeRegressor

predict(DecisionTreeRegressor())

reg_rf = RandomForestRegressor()


np.linspace(start =100 , stop=1200 , num=6)

n_estimators = [int(x) for x in np.linspace(start =100 , stop=1200 , num=6)]

max_features = ["auto", "sqrt"]

max_depth = [int(x) for x in np.linspace(start =5 , stop=30 , num=4)]

min_samples_split = [5,10,15,100]

    'n_estimators' : n_estimators , 
    'max_features' : max_features , 
    'max_depth' : max_depth , 
    'min_samples_split' : min_samples_split
}

random_grid

rf_random = RandomizedSearchCV(estimator=reg_rf , param_distributions=random_grid , cv=3 , n_jobs=-1 , verbose=2)

rf_random.fit(X_train , y_train)

rf_random.best_params_

rf_random.best_estimator_


rf_random.best_score_
