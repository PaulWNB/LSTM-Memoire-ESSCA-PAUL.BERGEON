import pandas as pd
import numpy as np
import chardet
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Chemin d'accès au fichier CSV
filename = '/Users/paulbergeon/Documents/mémoireESSCA/Classeur10.csv'

# Lecture du fichier CSV et conversion de la colonne Date en format datetime
with open(filename, 'rb') as f:
    result = chardet.detect(f.read())

df = pd.read_csv(filename, encoding=result['encoding'], delimiter=';')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Normalisation des colonnes de données
variation_scaler = MinMaxScaler(feature_range=(0, 1))
df['Variation normalisée'] = variation_scaler.fit_transform(df['Variation'].values.reshape(-1, 1))

sentiment_scaler = MinMaxScaler(feature_range=(0, 1))
df['Sentiment normalisé'] = sentiment_scaler.fit_transform(df['Sentiment'].values.reshape(-1, 1))

price_scaler = MinMaxScaler(feature_range=(0, 1))
df['Close normalisé'] = price_scaler.fit_transform(df['Close'].values.reshape(-1, 1))

train_size = int(len(df) * 0.7)
train_data, test_data = df[:train_size], df[train_size:]

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 1])
    return np.array(X), np.array(Y)

look_back = 20
train_X, train_Y = create_dataset(train_data[['Sentiment normalisé', 'Close normalisé', 'Variation normalisée']].values, look_back)
test_X, test_Y = create_dataset(test_data[['Sentiment normalisé', 'Close normalisé', 'Variation normalisée']].values, look_back)

model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 3)))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


model.fit(train_X, train_Y, epochs=100, batch_size=3, verbose=5)

test_predict = model.predict(test_X)

test_predict = price_scaler.inverse_transform(test_predict)
test_Y = price_scaler.inverse_transform([test_Y])
# Comparaison des valeurs prédites avec les valeurs réelles pour prédire si le prix va baisser ou augmenter
test_predict_direction = np.zeros(test_Y.flatten().shape)

test_predict_direction[test_predict > test_Y] = 1 # Si le prix prédit est supérieur au prix réel, la direction est 1 (augmente)
test_predict_direction[test_predict < test_Y] = -1 # Si le prix prédit est inférieur au prix réel, la direction est -1 (baisse)
test_predict_direction[test_predict == test_Y] = 0 # Si le prix prédit est égal au prix réel, la direction est 0 (stable)

mae = np.mean(np.abs(test_predict - test_Y))
rmse = np.sqrt(np.mean(np.square(test_predict - test_Y)))
r2 = r2_score(test_Y[0], test_predict[:, 0])

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 score:", r2)

test_dates = test_data.iloc[look_back + 1:]['Date'].values

results_df = pd.DataFrame({'Date': test_dates, 'Real Prices': test_Y.flatten(), 'Predicted Prices': test_predict.flatten()})
results_df.to_csv('predictions_vs_real.csv', index=False)
results_df = pd.DataFrame({'Date': test_dates, 'Real Prices': test_Y.flatten(), 'Predicted Prices': test_predict.flatten(), 'Direction': test_predict_direction.flatten()})
results_df.to_csv('predictions_vs_real.csv', index=False)

plt.plot(test_dates, test_Y.flatten(), label='Données réelles')
plt.plot(test_dates, test_predict.flatten(), label='Prédictions')
plt.xlabel('Date')
plt.ylabel('Prix de clôture')
plt.title('Prédictions du modèle LSTM')
plt.legend()
plt.show()
plt.scatter(test_Y.flatten(), test_predict.flatten())
plt.xlabel('Données réelles')
plt.ylabel('Prédictions')
plt.title('Relation entre les données réelles et les prédictions du modèle LSTM')
plt.show()
plt.plot(test_dates, test_predict_direction.flatten(), label='Direction prédite')
plt.xlabel('Date')
plt.ylabel('Direction')
plt.title('Prédictions de la direction du prix')
plt.legend()
plt.show()
