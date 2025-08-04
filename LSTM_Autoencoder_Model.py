import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import MinMaxScaler

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

def build_lstm_autoencoder(timesteps, features):
    """
    Build and compile an LSTM Autoencoder.
    
    Parameters:
    timesteps (int): Number of timesteps (sequence length) for LSTM.
    features (int): Number of features in the input data.
    
    Returns:
    Model: Compiled LSTM Autoencoder model.
    """
    model = Sequential()
    model.add(LSTM(64, input_shape=(timesteps, features), return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(TimeDistributed(Dense(features)))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model

def train_autoencoder(model, data, epochs=50, batch_size=32, output_dir='./', sensor_id=1):
    """
    Train the LSTM Autoencoder and save training/validation loss.
    
    Parameters:
    model (Model): The LSTM Autoencoder model.
    data (np.array): Training data.
    epochs (int): Number of training epochs.
    batch_size (int): Training batch size.
    output_dir (str): Directory to save the outputs.
    sensor_id (int): Sensor ID for naming the output files.
    
    Returns:
    Model: Trained model.
    """
    history = model.fit(data, data, epochs=epochs, batch_size=batch_size, validation_split=0.2, shuffle=False)
    
    # Save training and validation loss to a CSV file
    loss_data = pd.DataFrame({
        'epoch': range(1, epochs + 1),
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    })
    loss_filename = f'{output_dir}/sensor_{sensor_id}_training_validation_loss.csv'
    loss_data.to_csv(loss_filename, index=False)
    print(f"Training and validation loss saved to {loss_filename}")
    
    # Plot and save training loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.legend(loc='upper right', prop={'size': 20, 'family': 'Times New Roman'})
    # plt.title('Training and Validation Loss')
    plt.xlabel('Epochs', fontsize=28, fontname='Times New Roman', labelpad=18)  # labelpad increases spacing between label and axis
    plt.ylabel('Loss', fontsize=28, fontname='Times New Roman', labelpad=18)
    plt.xticks(fontsize=22, fontname='Times New Roman',)
    plt.yticks(fontsize=22, fontname='Times New Roman',)
    plt.tight_layout(pad=1)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)  # Customize grid color, style, and width
    plt.savefig(f'{output_dir}/sensor_{sensor_id}_training_validation_loss.png')
    plt.show()
    
    return model

def save_model(model, filename, timesteps):
    """
    Save the trained model to a file.
    
    Parameters:
    model (Model): The trained Keras model.
    filename (str): The filename to save the model.
    """
    model.save(filename)
    with open('timesteps.txt', 'w') as f:
        f.write(str(timesteps))
    print(f"Model saved to {filename}")

def process_sensor_anomalies(sensor_id, preprocessed_file_path, output_dir, timesteps=1):
    """
    Process sensor data to detect anomalies using an LSTM Autoencoder.
    
    Parameters:
    sensor_id (int): ID of the sensor (1, 2, or 3).
    preprocessed_file_path (str): Path to the preprocessed data file.
    output_dir (str): Directory to save the outputs.
    timesteps (int): Number of timesteps for LSTM input.
    """
    # Step 1: Load preprocessed data
    data = load_preprocessed_data(preprocessed_file_path, timesteps)
    features = data.shape[2]  # Number of features
    
    # Step 2: Build LSTM Autoencoder
    model = build_lstm_autoencoder(timesteps, features)
    
    # Step 3: Train the Autoencoder and save the loss data
    model = train_autoencoder(model, data, output_dir=output_dir, sensor_id=sensor_id)
    
    # Step 4: Save the model
    model_filename = f'{output_dir}/sensor_{sensor_id}_lstm_autoencoder.h5'
    save_model(model, model_filename, timesteps)

# Example Usage:
timesteps_s1 = 20
timesteps_s2 = 20
timesteps_s3 = 20
# Process and save the model for Sensor 1
process_sensor_anomalies(1, './sensor1.csv', './', timesteps_s1)

# Process and save the model for Sensor 2
# process_sensor_anomalies(2, './sensor2.csv', './', timesteps_s2)

# Process and save the model for Sensor 3
# process_sensor_anomalies(3, './sensor3.csv', './', timesteps_s3)
