import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

def load_preprocessed_data(file_path, timesteps):
    """
    Load the preprocessed data from a CSV file and reshape it for LSTM input.
    
    Parameters:
    file_path (str): Path to the preprocessed data file.
    timesteps (int): Number of timesteps to be used for LSTM input.
    
    Returns:
    np.array: Reshaped data ready for LSTM input.
    """
    df = pd.read_csv(file_path)
    data = df.iloc[:, 1:].values
    
    # Reshape data to fit LSTM input shape (batch_size, timesteps, features)
    num_samples = data.shape[0] // timesteps
    data = data[:num_samples * timesteps]  # Truncate data to fit an integer number of timesteps
    data = data.reshape((num_samples, timesteps, data.shape[1]))
    
    return data

def detect_anomalies(model, data, threshold=0.01):
    """
    Use the trained model to detect anomalies.
    
    Parameters:
    model (Model): The trained LSTM Autoencoder model.
    data (np.array): Input data for anomaly detection.
    threshold (float): Threshold for anomaly detection.
    
    Returns:
    np.array: Boolean array indicating anomalies.
    np.array: Reconstruction errors for each data point.
    """
    predictions = model.predict(data)   
    # Calculate the mean squared error for each time step in each sequence
    mse = np.mean(np.power(data - predictions, 2), axis=2)  # shape is (num_samples, timesteps)
    # Reduce to a single MSE per sample by taking the mean over the timesteps
    mse = np.mean(mse, axis=1)  # shape is (num_samples,)    
    # Determine which samples are anomalies
    anomalies = mse > threshold
    return anomalies, mse

def save_anomaly_figure(sensor_id, mse, anomalies, threshold):
    """
    Save a figure showing reconstruction errors, anomalies, and the threshold.
    
    Parameters:
    sensor_id (int): ID of the sensor.
    mse (np.array): Reconstruction errors.
    anomalies (np.array): Boolean array indicating anomalies.
    threshold (float): Threshold for anomaly detection.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(mse, label='Reconstruction Error')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.scatter(np.where(anomalies)[0], mse[anomalies], color='red', label='Anomalies')
    # plt.title(f'Sensor {sensor_id}: Reconstruction Error and Anomalies')
    plt.xlabel('Data Points', fontsize=28, fontname='Times New Roman', labelpad=18)  # labelpad increases spacing between label and axis
    plt.ylabel('Reconstruction Error', fontsize=28, fontname='Times New Roman', labelpad=18)
    plt.xticks(fontsize=22, fontname='Times New Roman',)
    plt.yticks(fontsize=22, fontname='Times New Roman',)
    plt.tight_layout(pad=1)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)  # Customize grid color, style, and width
    plt.legend(loc='upper left', prop={'size': 20, 'family': 'Times New Roman'})
    # Save the figure
    plt.savefig(f'Sensor_{sensor_id}_anomalies.png')
    plt.show()


def evaluate_anomalies(sensor_id, preprocessed_file_path, model_filename, threshold=None):
    """
    Evaluate anomalies in the dataset using the saved LSTM Autoencoder model.
    
    Parameters:
    sensor_id (int): ID of the sensor (1, 2, or 3).
    preprocessed_file_path (str): Path to the preprocessed data file.
    model_filename (str): Path to the saved LSTM Autoencoder model file.
    threshold (float): Threshold for anomaly detection. If None, it will be calculated.
    
    Returns:
    None
    """
    # Load the saved model
    model = load_model(model_filename)
    
    # Load the data
    with open('timesteps.txt', 'r') as f:
        timesteps = int(f.read())
    data = load_preprocessed_data(preprocessed_file_path, timesteps)
    
    # Detect anomalies
    anomalies, mse = detect_anomalies(model, data, threshold)
    
    # Save anomaly detection figures
    save_anomaly_figure(sensor_id, mse, anomalies, threshold)
    
    # Save anomalies to CSV
    anomaly_file = f'Sensor-{sensor_id}_anomalies.csv'
    df_anomalies = pd.DataFrame(anomalies, columns=['Anomaly'])
    df_anomalies.to_csv(anomaly_file, index=False)
    print(f"Anomalies saved to {anomaly_file}")

# Example Usage:
threshold_s1 = 0.03
threshold_s2 = 0.03
threshold_s3 = 0.03

# Load the model and evaluate anomalies for Sensor 1
evaluate_anomalies(1, './sensor1.csv', './sensor_1_lstm_autoencoder.h5', threshold=threshold_s1)

# Load the model and evaluate anomalies for Sensor 2
evaluate_anomalies(2, './sensor2.csv', './sensor_2_lstm_autoencoder.h5', threshold=threshold_s2)

# Load the model and evaluate anomalies for Sensor 3
evaluate_anomalies(3, './sensor3.csv', './sensor_3_lstm_autoencoder.h5', threshold=threshold_s3)
