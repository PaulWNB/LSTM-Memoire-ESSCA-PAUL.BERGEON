import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Chemin d'accès au fichier CSV
filename = '/Users/paulbergeon/Documents/barique/Classeur13.csv'

# Lecture du fichier CSV
df = pd.read_csv(filename, delimiter=';')


# Appliquez une normalisation MinMax aux colonnes "ETH" et "Close"
scaler = MinMaxScaler()
df[["ETH", "Close"]] = scaler.fit_transform(df[["ETH", "Close"]])
df.fillna(df.mean(), inplace=True)
# Divisez les données en ensembles d'entraînement et de test
train_size = int(len(df) * 0.8)
train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

# Convertissez les ensembles d'entraînement et de test en tableaux numpy
X_train, y_train = train_data.drop(columns=["Close"]).values, train_data[["Close"]].values
X_test, y_test = test_data.drop(columns=["Close"]).values, test_data[["Close"]].values

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Créez le modèle RNN
model = Sequential()
model.add(LSTM(32, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))

# Compilez le modèle
model.compile(loss='mse', optimizer='adam')

# Entraînez le modèle
model.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), y_train, epochs=50, batch_size=32, validation_data=(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)), y_test))

# Faites des prédictions sur l'ensemble de test
y_pred = model.predict(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)))

# Calculez les métriques d'évaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Affichez les métriques
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 score:", r2)
