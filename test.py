import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import chardet

# Charger les données de prix du Bitcoin



with open('/Users/paulbergeon/Documents/mémoireESSCA/databtc.csv', 'rb') as f:
    result = chardet.detect(f.read())
    
df = pd.read_csv('/Users/paulbergeon/Documents/barique/databtc2.csv', encoding=result['encoding'])
print(df.head())

# Prétraiter les données
scaler = MinMaxScaler(feature_range=(0, 1))
df['Prix normalisé'] = scaler.fit_transform(df['Prix'].values.reshape(-1,1))

# Diviser les données en ensembles d'entraînement et de test
train_size = int(len(df) * 0.7)
train_data, test_data = df[:train_size], df[train_size:]

# Créer les ensembles d'entrée et de sortie pour le modèle LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 20
train_X, train_Y = create_dataset(train_data['Prix normalisé'].values.reshape(-1,1), look_back)
test_X, test_Y = create_dataset(test_data['Prix normalisé'].values.reshape(-1,1), look_back)

# Créer le modèle LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Entraîner le modèle
model.fit(train_X, train_Y, epochs=100, batch_size=1, verbose=2)

# Faire des prédictions sur les données de test
test_predict = model.predict(test_X)

# Inverser la normalisation des prédictions et des données de test
test_predict = scaler.inverse_transform(test_predict)
test_Y = scaler.inverse_transform([test_Y])

# Calculer l'erreur moyenne absolue et l'erreur moyenne quadratique
mae = np.mean(np.abs(test_predict - test_Y))
rmse = np.sqrt(np.mean(np.square(test_predict - test_Y)))

# Afficher les résultats
print("MAE:", mae)
print("RMSE:", rmse)
