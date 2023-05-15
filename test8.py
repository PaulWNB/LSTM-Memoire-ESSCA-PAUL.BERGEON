import pandas as pd
import numpy as np
import chardet
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

# Chemin d'accès au fichier CSV
filename = '/Users/paulbergeon/Documents/barique/Classeur10.csv'

# Lecture du fichier CSV et conversion de la colonne Date en format datetime
with open(filename, 'rb') as f:
    result = chardet.detect(f.read())

df = pd.read_csv(filename, encoding=result['encoding'], delimiter=';')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Sélection des colonnes nécessaires pour l'entraînement du modèle
data = df[['Sentiment', 'Close', 'Variation']]

# Normalisation des données à l'aide de MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Séparation des ensembles d'entraînement et de test
train_size = int(len(data_scaled) * 0.8)
test_size = len(data_scaled) - train_size
train_data, test_data = data_scaled[0:train_size,:], data_scaled[train_size:len(data_scaled),:]

# Création des ensembles X et y pour l'apprentissage et les tests
def create_dataset(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :])
        y.append(data[i + look_back, 2])  # On prédit le prix du Bitcoin (Close)
    return np.array(X), np.array(y)

look_back = 20 # Nombre de pas de temps à regarder en arrière
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

# Création du modèle LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compilation et entraînement du modèle
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=150, batch_size=5, verbose=2)

# Prédiction des valeurs de test et dénormalisation des résultats
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(np.hstack((np.zeros((len(y_pred), 1)), y_pred, np.zeros((len(y_pred), 1)))))[look_back:, 1]

# Calculez les mesures de performance MAE, RMSE et R2 score
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(np.mean(np.square(test_predict - test_Y)))
r2 = r2_score(test_Y[0], test_predict[:, 0])

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 score:", r2)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# Visualisez les résultats
plt.plot(test_data.index[look_back:], y_test, label='Actual Price Change')
plt.plot(data.index[train_size + look_back:], y_pred, label='Predicted Bitcoin Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
