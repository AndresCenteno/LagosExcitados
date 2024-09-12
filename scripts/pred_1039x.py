import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load the data
df = pd.read_csv("data/all_terciaria_merged/pivoted_quarterly.csv")
df['hour'] = pd.to_datetime(df['hour'])
dtime = pd.to_datetime(df['hour'])
df['H'] = df['hour'].dt.hour
df['D'] = df['hour'].dt.day
df['M'] = df['hour'].dt.month
df = df.drop(columns=["Unnamed: 0","hour"])
# Extract relevant columns
target_cols = ['10394.00', '10394.15', '10394.30', '10394.45', '10395.00', '10395.15', '10395.30', '10395.45']
target_idx = [df.columns.get_loc(col) for col in target_cols]
# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df[:])
scaler_y = MinMaxScaler(feature_range=(0,1))
data_y_scaled = scaler_y.fit_transform(df[target_cols])
# Prepare the sequences with two rows of delay
def create_sequences(data, input_len=24, delay=2):
    X, y = [], []
    for i in range(len(data) - input_len - delay):
        X.append(data[i:i + input_len])   # 24 hours as input
        y.append(data[i + input_len + delay - 1,target_idx])  # 26th hour as output (input_len + delay)
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled)

# Split the data into training and testing sets (80% training, 20% testing)
split = int(0.97 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))  # 24-hour input sequence
model.add(Dense(len(target_cols)))  # Output for each of the columns
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the predictions and actual values
predictions = scaler_y.inverse_transform(predictions)
y_test = scaler_y.inverse_transform(y_test)

pred_94 = predictions[:,:4].flatten()
pred_95 = predictions[:,4:].flatten()
test_94 = y_test[:,:4].flatten()
test_95 = y_test[:,4:].flatten()

import matplotlib.pyplot as plt

len(pred_94)
time_axis = dtime[len(dtime)-len(pred_94):]

assert len(time_axis) == len(pred_94), "Time axis length does not match the length of data."
fig, axs = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

# Plot for pred_94 vs. test_94
axs[0].plot(time_axis, pred_94, label='Predictions', color='blue', alpha=0.7)
axs[0].plot(time_axis, test_94, label='Observations', color='orange', alpha=0.7)
axs[0].set_title('2h Ahead Predictions of 10394', fontsize=16)
axs[0].set_ylabel('Values', fontsize=14)
axs[0].legend(loc='upper right', fontsize=12)
axs[0].grid(True, linestyle='--', linewidth=0.7)
axs[0].tick_params(axis='both', which='major', labelsize=12)

# Plot for pred_95 vs. test_95
axs[1].plot(time_axis, pred_95, label='Predictions', color='green', alpha=0.7)
axs[1].plot(time_axis, test_95, label='Observations', color='red', alpha=0.7)
axs[1].set_title('2h Ahead Predictions of 10395', fontsize=16)
axs[1].set_xlabel('Time', fontsize=14)
axs[1].set_ylabel('Values', fontsize=14)
axs[1].legend(loc='upper right', fontsize=12)
axs[1].grid(True, linestyle='--', linewidth=0.7)
axs[1].tick_params(axis='both', which='major', labelsize=12)

# Adjust layout to ensure subplots fit within the figure area
plt.tight_layout()

# Save the figure as a PNG file
plt.savefig("figs/2h_ahead_terciaria.png", dpi=300)

# Show the plots
plt.show()