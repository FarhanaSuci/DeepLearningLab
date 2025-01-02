# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import keras_tuner as kt

# Step 1: Load the data
df = pd.read_csv('Alcohol_Sales.csv')  # Ensure the file path is correct

# Step 2: Explore and preprocess the data
print("Original columns:", df.columns)  # Display column names
df.columns = ['DATE', 'Sales']  # Rename columns if needed
print(df.head())  # Display the first few rows

# Convert the DATE column to datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Plot the sales data
plt.figure(figsize=(12, 8))
plt.plot(df['Sales'], label='Sales')
plt.title('Sales Over Time')
plt.ylabel('Sales')
plt.show()

# Step 3: Create lagged features
df['Sale_LastMonth'] = df['Sales'].shift(1)
df['Sale_2Monthsback'] = df['Sales'].shift(2)
df['Sale_3Monthback'] = df['Sales'].shift(3)

# Drop rows with missing values due to shifting
df = df.dropna()

# Step 4: Prepare the data
x = df.iloc[:, 2:].values  # Use lagged features as input
y = df.iloc[:, 1:2].values  # Use current sales as target

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

# Normalize the data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
xtrain = scaler_x.fit_transform(xtrain)
xtest = scaler_x.transform(xtest)
ytrain = scaler_y.fit_transform(ytrain)
ytest = scaler_y.transform(ytest)

# Reshape the data for LSTM input
xtrain = xtrain.reshape((xtrain.shape[0], xtrain.shape[1], 1))
xtest = xtest.reshape((xtest.shape[0], xtest.shape[1], 1))

# Step 5: Define the model with hyperparameter tuning
def build_model(hp):
    model = Sequential()
    model.add(LSTM(
        units=hp.Int('units', min_value=32, max_value=128, step=16),
        activation='relu',
        input_shape=(xtrain.shape[1], 1)
    ))
    model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(
        units=hp.Int('dense_units', min_value=16, max_value=64, step=16),
        activation='relu'
    ))
    model.add(Dense(1))
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    return model

# Step 6: Hyperparameter optimization
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='lstm_sales_tuning'
)

# Run the tuner
tuner.search(xtrain, ytrain, epochs=50, validation_data=(xtest, ytest), batch_size=5)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The optimal number of units in the LSTM layer is {best_hps.get('units')},
dropout rate is {best_hps.get('dropout')},
and the number of units in the dense layer is {best_hps.get('dense_units')}.
""")

# Step 7: Build and train the best model
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(xtrain, ytrain, epochs=60, batch_size=5, validation_data=(xtest, ytest))

# Step 8: Evaluate the best model
loss, mae = best_model.evaluate(xtest, ytest)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Step 9: Predict and visualize results
predictions = best_model.predict(xtest)
predictions_rescaled = scaler_y.inverse_transform(predictions)
ytest_rescaled = scaler_y.inverse_transform(ytest)

plt.figure(figsize=(12, 8))
plt.plot(predictions_rescaled, label='Predicted Sales')
plt.plot(ytest_rescaled, label='Actual Sales')
plt.legend()
plt.title('Predicted vs Actual Sales')
plt.ylabel('Sales')
plt.show()
