import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from keras.layers import LSTM, Dense, InputLayer
from sklearn.metrics import mean_squared_error as mse
from time import time
import matplotlib.pyplot as plt
import matplotlib
import warnings

matplotlib.use('qt5agg')
warnings.filterwarnings('ignore')


# === functions ===
def create_sequences_unistep(data, n_steps):
    data_t = data.to_numpy()
    X = []
    y = []

    for i in range(len(data_t)-n_steps):
      row = [a for a in data_t[i:i+n_steps]]
      X.append(row)

      label = data_t[i+n_steps][0]
      y.append(label)

    return np.array(X), np.array(y)


def train_model(X, y, X_val, y_val, n_steps, n_preds=1):
    n_features = X.shape[2]

    # Create lstm model
    model = Sequential()
    model.add(InputLayer((n_steps, n_features)))
    model.add(LSTM(4, return_sequences=True))
    model.add(LSTM(5))
    model.add(Dense(n_preds, activation='linear'))

    # Compile model
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

    model.summary()

    # Save model with the least validation loss
    checkpoint_filepath = 'cps/best_model_unistep.h5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',  # Monitor validation loss
        mode='min',          # Save the model with the minimum validation loss
        save_best_only=True)

    # Stop training if validation loss does not improve in 500 epochs
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=50,  # Stop training if no improvement in validation loss for 100 epochs
        mode='min',
        verbose=1,
        restore_best_weights=True)  # when finish train restore best model

    # Fit model
    ts = time()
    history = model.fit(X, y,
                        verbose=2,
                        epochs=250,
                        validation_data=(X_val, y_val),
                        callbacks=[model_checkpoint_callback, early_stopping_callback])
    tf = time()

    print('Time to train model: {} s'.format(round(tf - ts, 2)))

    # Plot loss evolution
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Load best model
    del model
    model = load_model(checkpoint_filepath)

    return model


def preprocess_input(X, mean, std):
    X[:, :, 0] = (X[:, :, 0] - mean) / std
    return X


def preprocess_output(y, mean, std):
    y = (y - mean) / std
    return y


def postprocess_output(y, mean, std):
    y = (y * std) + mean
    return y


def plot_predictions_unistep(model, X_test, y_test, mean_ref, std_ref):

    preds = model.predict(X_test).flatten().tolist()

    # preprocess preds to actual scale
    preds = [postprocess_output(i, mean_ref, std_ref) for i in preds]
    y_t = [postprocess_output(i, mean_ref, std_ref) for i in y_test.tolist()]

    er = mse(y_test, preds)

    plt.figure(figsize=(12, 8))
    plt.plot(y_t, label='Actual values')
    plt.plot(preds, label='Predictions', alpha=.7)
    plt.legend()
    plt.title('MSE = {}'.format(er))

    return preds


# Load data
path_data = r'data/filter_pt_data.csv'
df = pd.read_csv(path_data)
df.dropna(inplace=True)

print(df.head())
print(df.info())

# Convert dt data type from object to datetime
df['dt'] = pd.to_datetime(df['dt'])
df.set_index('dt', inplace=True)

# View data
plt.plot(df)
plt.title('Monthly average temperature in Portugal')
plt.xlabel('Date')
plt.ylabel('Temerature')

# Add time features
df['Seconds'] = df.index.map(pd.Timestamp.timestamp)

year_secs = 60 * 60 * 24 * 365  # Number of seconds in a year

df['year_signal_sin'] = np.sin(df['Seconds'] * (2 * np.pi / year_secs))
df['year_signal_cos'] = np.cos(df['Seconds'] * (2 * np.pi / year_secs))

df.drop(columns=['Seconds'], inplace=True)

plt.plot(df['year_signal_sin'], label='sin signal')
plt.plot(df['year_signal_cos'], label='cos signal')
plt.legend()


n_steps = 5
X, y = create_sequences_unistep(df, n_steps)

# Prepare train and validation data
nr_vals_train = 500
nr_vals_validation = 50

X_train = X[:nr_vals_train]
y_train = y[:nr_vals_train]

X_val = X[nr_vals_train: nr_vals_train + nr_vals_validation]
y_val = y[nr_vals_train: nr_vals_train + nr_vals_validation]

X_test = X[nr_vals_train:]
y_test = y[nr_vals_train:]

print('X train shape: {}'.format(X_train.shape))
print('y train shape: {}'.format(y_train.shape))

print('X validation shape: {}'.format(X_val.shape))
print('y validation shape: {}'.format(y_val.shape))

# Scale temp value with standard scaler -> mean 0 and std 1
mean_ref = np.mean(X_train[:, :, 0])
std_ref = np.std(X_train[:, :, 0])

# Scale X's
X_train = preprocess_input(X_train, mean_ref, std_ref)
X_val = preprocess_input(X_val, mean_ref, std_ref)
X_test = preprocess_input(X_test, mean_ref, std_ref)

# Scale y's
y_train = preprocess_output(y_train, mean_ref, std_ref)
y_val = preprocess_output(y_val, mean_ref, std_ref)
y_test = preprocess_output(y_test, mean_ref, std_ref)

model = train_model(X_train, y_train, X_val, y_val, n_steps)


# Plot train predictions set
plot_predictions_unistep(model, X_train, y_train, mean_ref, std_ref)
plot_predictions_unistep(model, X_test, y_test, mean_ref, std_ref)

