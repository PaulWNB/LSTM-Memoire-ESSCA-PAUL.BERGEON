import pandas as pd
import numpy as np
import chardet
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, LeakyReLU
from keras.optimizers import RMSprop, SGD
import matplotlib.pyplot as plt

# Chemin d'accès au fichier CSV
filename = '/Users/paulbergeon/Documents/mémoireESSCA/Classeur11.csv'

# Lecture du fichier CSV et conversion de la colonne Date en format datetime
with open(filename, 'rb') as f:
    result = chardet.detect(f.read())

df = pd.read_csv(filename, encoding=result['encoding'], delimiter=';')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df = pd.read_csv(filename, encoding=result['encoding'], delimiter=';', usecols=['Date', 'Close', 'Variation', 'Sentiment', 'ETH', 'Vol', 'Haut', 'Bas'], converters={'Variation': lambda x: float(x.strip().replace(',', '.'))}, skipfooter=1)

# Normalisation des colonnes de données
variation_scaler = MinMaxScaler(feature_range=(0, 1))
df['Variation normalisée'] = variation_scaler.fit_transform(df['Variation'].values.reshape(-1, 1))

sentiment_scaler = MinMaxScaler(feature_range=(0, 1))
df['Sentiment normalisé'] = sentiment_scaler.fit_transform(df['Sentiment'].values.reshape(-1, 1))

price_scaler = MinMaxScaler(feature_range=(0, 1))
df['Close normalisé'] = price_scaler.fit_transform(df['Close'].values.reshape(-1, 1))
eth_scaler = MinMaxScaler(feature_range=(0, 1))
df['ETH normalisé'] = eth_scaler.fit_transform(df['ETH'].values.reshape(-1, 1))
vol_scaler = MinMaxScaler(feature_range=(0, 1))
df['Vol normalisé'] = eth_scaler.fit_transform(df['Vol'].values.reshape(-1, 1))
Haut_scaler = MinMaxScaler(feature_range=(0, 1))
df['Haut normalisé'] = eth_scaler.fit_transform(df['Haut'].values.reshape(-1, 1))
Bas_scaler = MinMaxScaler(feature_range=(0, 1))
df['Bas normalisé'] = eth_scaler.fit_transform(df['Bas'].values.reshape(-1, 1))

train_size = int(len(df) * 0.6)
train_data, test_data = df[:train_size], df[train_size:]

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 1])
    return np.array(X), np.array(Y)

look_back = 30
train_X, train_Y = create_dataset(train_data[['Sentiment normalisé', 'Close normalisé', 'Variation normalisée', 'ETH normalisé', 'Vol normalisé', 'Haut normalisé', 'Bas normalisé']].values, look_back)
test_X, test_Y = create_dataset(test_data[['Sentiment normalisé', 'Close normalisé', 'Variation normalisée', 'ETH normalisé', 'Vol normalisé', 'Haut normalisé', 'Bas normalisé']].values, look_back)


def create_model(activation_function='relu', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(100, input_shape=(look_back, 7)))
    
    model.add(Dense(50))
    if activation_function == 'leakyrelu':
        model.add(LeakyReLU(alpha=0.3))
    else:
        model.add(Activation(activation_function))
    
    model.add(Dense(50))
    if activation_function == 'leakyrelu':
        model.add(LeakyReLU(alpha=0.3))
    else:
        model.add(Activation(activation_function))
    
    model.add(Dense(50))
    if activation_function == 'leakyrelu':
        model.add(LeakyReLU(alpha=0.3))
    else:
        model.add(Activation(activation_function))
    
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Exemples d'utilisation de différentes fonctions d'activation et optimiseurs
activation_functions = ['tanh', 'relu', 'leakyrelu']
optimizers = ['adam', 'rmsprop', 'sgd']

for activation_function in activation_functions:
    for optimizer in optimizers:
        print(f"Training model with activation function '{activation_function}' and optimizer '{optimizer}'")
        model = create_model(activation_function=activation_function, optimizer=optimizer)
        model.fit(train_X, train_Y, epochs=150, batch_size=16, verbose=2)

        test_predict = model.predict(test_X)
        test_predict = price_scaler.inverse_transform(test_predict)
        test_Y = price_scaler.inverse_transform([test_Y])

        mae = np.mean(np.abs(test_predict - test_Y))
        rmse = np.sqrt(np.mean(np.square(test_predict - test_Y)))
        r2 = r2_score(test_Y[0], test_predict[:, 0])

        print("MAE:", mae)
        print("RMSE:", rmse)
        print("R2 score:", r2)

        test_dates = test_data.iloc[look_back + 1:]['Date'].values

        results_df = pd.DataFrame({'Date': test_dates, 'Real Prices': test_Y.flatten(), 'Predicted Prices': test_predict.flatten()})
        results_df.to_csv(f'predictions_vs_real_{activation_function}_{optimizer}.csv', index=False)

        # Affichage des graphiques
        fig, ax = plt.subplots()
        ax.plot(test_dates, test_Y.flatten(), label='Données réelles')
        ax.plot(test_dates, test_predict.flatten(), label='Prédictions')
        ax.set_xlabel('Date')
        ax.set_ylabel('Prix de clôture')
        ax.set_title(f'Prédictions du modèle LSTM ({activation_function}, {optimizer})')
        ax.legend()

        # Configurer les limites de l'axe x pour afficher une date tous les 2 mois
        start_date = test_dates[0]
        end_date = test_dates[-1]
        # Convertir test_dates en objets datetime
        test_dates_datetime = pd.to_datetime(test_dates, format='%d/%m/%Y')

        # Créer une liste d'étiquettes de date pour l'axe x
        date_labels = test_dates_datetime[::60].strftime('%d/%m/%Y') # un label tous les 2 mois

        # Configurer les limites et les étiquettes de l'axe x
        ax.set_xlim(start_date, end_date)
        ax.set_xticks(test_dates_datetime[::60])
        ax.set_xticklabels(date_labels, rotation=45)
        ax.xaxis.set_major_locator(plt.MaxNLocator(9)) # 9 est le nombre de dates que vous voulez afficher
        plt.xticks(rotation=45)
        
        plt.savefig(f'predictions_vs_real_{activation_function}_{optimizer}.png')
        plt.show()


