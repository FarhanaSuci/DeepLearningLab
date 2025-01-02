import numpy as np
import pandas as pd
df=pd.read_csv('Alcohol_Sales.csv')
print(df)
df['Sale_LastMonth']=df['Sales'].shift(+1)
df['Sale_2Monthsback']=df['Sales'].shift(+2)
df['Sale_3Monthback']=df['Sales'].shift(+3)
print(df)
df=df.dropna()
x=df.iloc[:,2:]
y=df.iloc[:,1:2]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.fit_transform(xtest)
ytrain=scaler.fit_transform(ytrain)
ytest=scaler.fit_transform(ytest)
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Input

def create_model(input_dim, optimizer='adam'):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))  # Input shape
    model.add(Dense(12, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification output
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

# Wrapping the Keras model using KerasClassifier
model = KerasRegressor(build_fn=create_model, input_dim=xtrain.shape[1], verbose=0)

# Define the grid search parameters
param_grid = {
    'optimizer' :['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
    'batch_size': [5, 10],
    'epochs': [100, 200]
}

# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(xtrain, ytrain)

# Summarize the results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

