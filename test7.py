import pandas as pd
import numpy as np
import chardet
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
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
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        future_price = dataset[i + look_back, 2]
        current_price = dataset[i + look_back - 1,2]
        if future_price > current_price:
            Y.append(1) # Prix augmentera
        elif future_price < current_price:
            Y.append(-1) # Prix baissera
        else:
            Y.append(0) # Prix restera stable
    return np.array(X), np.array(Y)

look_back = 20
train_X, train_Y = create_dataset(train_data[['Sentiment normalisé', 'Close normalisé', 'Variation normalisée']].values, look_back)
test_X, test_Y = create_dataset(test_data[['Sentiment normalisé', 'Close normalisé', 'Variation normalisée']].values, look_back)

model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 3)))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='tanh'))
model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(train_X, train_Y, epochs=100, batch_size=3, verbose=5)

test_predict = model.predict(test_X)

# Comparaison des valeurs prédites avec les valeurs réelles pour prédire si le prix va baisser ou augmenter
test_predict_direction = np.zeros(test_Y.flatten().shape)
test_predict_direction[test_predict.flatten() > test_Y] = -1
test_predict_direction[test_predict.flatten() < test_Y] = 1
test_predict_direction[test_predict.flatten() == test_Y] = 0

accuracy = np.mean(test_predict_direction == test_Y)
print("Accuracy:", accuracy)

test_dates = test_data.iloc[look_back + 1:]['Date'].values

results_df = pd.DataFrame({'Date': test_dates, 'Real Direction': test_Y.flatten(), 'Predicted Direction': test_predict_direction.flatten()})
results_df.to_csv('direction_predictions_vs_real.csv', index=False)

print(results_df)

# Visualisation des résultats
fig, axs = plt.subplots(2, 1, figsize=(12, 8))
axs[0].plot(test_dates, test_Y.flatten(), label='Real Direction')
axs[0].plot(test_dates, test_predict_direction.flatten(), label='Predicted Direction')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Direction')
axs[0].set_title('Direction Predictions for the Test Set')
axs[0].legend()

axs[1].scatter(test_Y.flatten(), test_predict_direction.flatten())
axs[1].set_xlabel('Real Direction')
axs[1].set_ylabel('Predicted Direction')
axs[1].set_title('Real vs Predicted Direction')
plt.show()
