import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import chardet
from datetime import datetime
# Charger les données à partir du fichier CSV
filename = '/Users/paulbergeon/Documents/mémoireESSCA/Classeur10.csv' 

with open(filename, 'rb') as f:
    result = chardet.detect(f.read())
    
df = pd.read_csv(filename, encoding=result['encoding'], delimiter=';')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Normaliser le score de sentiment sur une plage de valeurs de 0 à 1
df['Sentiment'] = df['Sentiment'] / 100.0

# Prétraiter les données et les normaliser
scaler = MinMaxScaler(feature_range=(0, 1))
df['Close normalisé'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
df['Variation normalisée'] = scaler.fit_transform(df['Variation'].values.reshape(-1, 1))
df['Sentiment normalisé'] = scaler.fit_transform(df['Sentiment'].values.reshape(-1, 1))

# Diviser les données en ensembles d'entraînement et de test
train_size = int(len(df) * 0.7)
train_data, test_data = df[:train_size], df[train_size:]

# Créer les ensembles d'entrée et de sortie pour le modèle LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 2])
    return np.array(X), np.array(Y)

look_back = 20
train_X, train_Y = create_dataset(train_data[['Sentiment normalisé', 'Close normalisé', 'Variation normalisée']].values, look_back)
test_X, test_Y = create_dataset(test_data[['Sentiment normalisé', 'Close normalisé', 'Variation normalisée']].values, look_back)

# Créer et entraîner le modèle LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 3)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(train_X, train_Y, epochs=100, batch_size=1, verbose=2)

# Effectuer des prédictions et évaluer les performances du modèle
test_predict = model.predict(test_X)

# Inverser la normalisation des prédictions
test_predict = scaler.inverse_transform(test_predict)

# Créer un DataFrame pour stocker les prédictions avec leurs dates correspondantes
test_dates = test_data['Date'].iloc[look_back+1:].reset_index(drop=True)
test_predict_df = pd.DataFrame(data=test_predict.flatten(), columns=['Prediction'])
test_predict_df['Date'] = test_dates
test_predict_df = test_predict_df.set_index('Date')

# Inverser la normalisation des vraies valeurs de prix de clôture
test_Y = scaler.inverse_transform([test_Y])[0]

# Créer un DataFrame pour stocker les vraies valeurs avec leurs dates correspondantes
test_y_df = pd.DataFrame(data=test_Y, columns=['Close'])
test_y_df['Date'] = test_dates
test_y_df = test_y_df.set_index('Date')

# Afficher les prédictions avec les vraies valeurs de prix de clôture
print(test_y_df.join(test_predict_df))




# Effectuer des prédictions et évaluer les performances du modèle
test_predict = model.predict(test_X)

test_predict = scaler.inverse_transform(test_predict)
test_Y = scaler.inverse_transform([test_Y])

mae = np.mean(np.abs(test_predict - test_Y))
rmse = np.sqrt(np.mean(np.square(test_predict - test_Y)))

print("MAE:", mae)
print("RMSE:", rmse)
print(test_Y)
print(test_X)
import matplotlib.pyplot as plt

# Tracer les prédictions et les données réelles
plt.plot(test_Y.flatten(), label='Données réelles')
plt.plot(test_predict.flatten(), label='Prédictions')
plt.xlabel('Période')
plt.ylabel('Prix de clôture')
plt.title('Prédictions du modèle LSTM')
plt.legend()
plt.show()
