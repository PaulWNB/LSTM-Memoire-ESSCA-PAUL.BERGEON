import pandas as pd
import numpy as np
import chardet
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import GRU, Dense, Activation
from keras.optimizers import RMSprop, SGD
import matplotlib.pyplot as plt

# Chemin d'accès au fichier CSV
filename = '/Users/paulbergeon/Documents/barique/donnéescomplet.csv'

# Lecture du fichier CSV et conversion de la colonne Date en format datetime
with open(filename, 'rb') as f:
    result = chardet.detect(f.read())

  

df = pd.read_csv(filename, encoding=result['encoding'], delimiter=';', engine='python', usecols=['Date', 'Prix', 'Variation', 'Sentiment', 'ETH', 'Vol', 'Haut', 'Bas','Volatilite'], converters={'Variation': lambda x: float(x.strip().replace(',', '.'))}, skipfooter=1)
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
data['Volatilite'] = data['Volatilite'].str.rstrip(';')
data['Volatilite'] = pd.to_numeric(data['Volatilite'])
# Normalisation des colonnes de données

sentiment_scaler = MinMaxScaler(feature_range=(0, 1))
df['Sentiment normalisé'] = sentiment_scaler.fit_transform(df['Sentiment'].values.reshape(-1, 1))

price_scaler = MinMaxScaler(feature_range=(0, 1))
df['Close normalisé'] = price_scaler.fit_transform(df['Prix'].values.reshape(-1, 1))

eth_scaler = MinMaxScaler(feature_range=(0, 1))
df['ETH normalisé'] = eth_scaler.fit_transform(df['ETH'].values.reshape(-1, 1))
vol_scaler = MinMaxScaler(feature_range=(0, 1))
df['Vol normalisé'] = eth_scaler.fit_transform(df['Vol'].values.reshape(-1, 1))
Haut_scaler = MinMaxScaler(feature_range=(0, 1))
df['Haut normalisé'] = eth_scaler.fit_transform(df['Haut'].values.reshape(-1, 1))
Bas_scaler = MinMaxScaler(feature_range=(0, 1))
df['Bas normalisé'] = eth_scaler.fit_transform(df['Bas'].values.reshape(-1, 1))
variation_scaler = MinMaxScaler(feature_range=(0, 1))
df['Variation normalisée'] = variation_scaler.fit_transform(df['Variation'].values.reshape(-1, 1))
volatilite_scaler = MinMaxScaler(feature_range=(0, 1))
df['Volatilite normalisée'] = variation_scaler.fit_transform(df['Volatilite'].values.reshape(-1, 1))





train_size = int(len(df) * 0.7)
train_data, test_data = df[:train_size], df[train_size:]

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 2])
    return np.array(X), np.array(Y)

look_back = 30
train_X, train_Y = create_dataset(train_data[['Sentiment normalisé', 'Close normalisé', 'Variation normalisée', 'ETH normalisé', 'Vol normalisé', 'Haut normalisé', 'Bas normalisé']].values, look_back)
test_X, test_Y = create_dataset(test_data[['Sentiment normalisé', 'Close normalisé', 'Variation normalisée', 'ETH normalisé', 'Vol normalisé', 'Haut normalisé', 'Bas normalisé']].values, look_back)

# Création du modèle GRU
model = Sequential()
model.add(GRU(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(1))

# Compilation du modèle
optimizer = RMSprop(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=optimizer)
model.summary()

# Entraînement du modèle
history = model.fit(train_X, train_Y, epochs=50, batch_size=16, validation_split=0.1, verbose=2, shuffle=False)

## Prédiction des valeurs sur l'ensemble de test
predicted_Y = model.predict(test_X)

# Calcul du score R2
mae = np.mean(np.abs(predicted_Y - test_Y))
rmse = np.sqrt(np.mean(np.square(predicted_Y - test_Y)))

print("MAE:", mae)
print("RMSE:", rmse)

r2 = r2_score(test_Y, predicted_Y)
print("R2 score:", r2)


# Transformation inverse de la normalisation pour les prédictions et les vraies valeurs de Close
predicted_Y = price_scaler.inverse_transform(predicted_Y)
test_Y = price_scaler.inverse_transform(test_Y.reshape(-1, 1))

# Affichage des résultats
plt.plot(test_Y, label="Valeurs réelles")
plt.plot(predicted_Y, label="Valeurs prédites")
plt.xlabel("Échantillons de test")
plt.ylabel("Valeur de 'Close'")
plt.legend()
plt.show()