import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

data = pd.read_csv("/Users/paulbergeon/Documents/mémoire/Donnéescomplet.csv", delimiter=";")

# Supprimer la colonne "Date"
data = data.drop("Date", axis=1)

# Extraire les fonctionnalités et la cible
features = data.drop(["Prix"], axis=1)
target = data["Prix"]

# Normaliser les fonctionnalités
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, shuffle=False)

model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1],)))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")

model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print("Train Loss:", train_loss)
print("Test Loss:", test_loss)

predictions = model.predict(X_test)
