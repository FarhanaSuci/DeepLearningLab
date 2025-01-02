import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout

# Load the data
df = pd.read_csv('Alcohol_Sales.csv')  # Update with the correct file path

# Debug: Print column names to identify the sales column
print("Original columns:", df.columns)

# Adjust column names based on actual dataset
# Replace 'Sales' with the actual sales column name
df.columns = ['DATE', 'Sales']  # Adjust if necessary

# Debug: Verify the updated DataFrame
print(df.head())

# Plot the sales data

df['DATE'] = pd.to_datetime(df['DATE'])

# Plot the data
plt.figure(figsize=(12, 8))  # Create a new figure
plt.plot(df['Sales'],label = 'Sales')
  # Set 'DATE' as index for the plot
plt.title('Sales Over Time')                      # Label for x-axis
plt.ylabel('Sales')                   # Label for y-axis

# Save the plot
plt.show()

# Create lag features for past months
df['Sale_LastMonth'] = df['Sales'].shift(+1)
df['Sale_2Monthsback'] = df['Sales'].shift(+2)
df['Sale_3Monthback'] = df['Sales'].shift(+3)

# Drop rows with missing values (due to shifting)
df = df.dropna()

# Prepare input (X) and output (y)
x = df.iloc[:, 2:].values  # Lag features
y = df.iloc[:, 1:2].values  # Current sales


# Split the data into train and test sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

# Normalize the data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

xtrain = scaler_x.fit_transform(xtrain)
xtest = scaler_x.transform(xtest)
ytrain = scaler_y.fit_transform(ytrain)
ytest = scaler_y.transform(ytest)

# Reshape input for CNN (samples, time steps, features)
xtrain = xtrain.reshape((xtrain.shape[0], xtrain.shape[1], 1))
xtest = xtest.reshape((xtest.shape[0], xtest.shape[1], 1))

# Build the CNN model
model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(xtrain.shape[1], 1)),
    Dropout(0.2),  # Dropout for regularization
    Conv1D(filters=32, kernel_size=2, activation='relu'),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
model.fit(xtrain, ytrain, epochs=100, batch_size=5, validation_data=(xtest, ytest))

# Evaluate the model
loss, mae = model.evaluate(xtest, ytest)
print(f"Test Loss: {loss}, Test MAE: {mae}")
predict=model.predict(xtest)
plt.figure(figsize=(12,8))
plt.plot(predict,label = 'Convolutional Neural Networks')
plt.plot(ytest,label = 'Actual Sales')
plt.legend(loc= 'upper left')
plt.show()
