import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.regularizers import l1, l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement de données
df = pd.read_csv('Classeur11.csv', delimiter=';')

# Ajout d'une colonne pour le prix du btc
df['Prix'] = df['Haut'] - df['Bas']

# Analyse de corrélation
print(df.corr())

# Normalisation des données
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)

# Séparation en ensembles de formation, de validation et de test
train, test = train_test_split(scaled_data, test_size=0.2, shuffle=False)
train, val = train_test_split(train, test_size=0.2, shuffle=False)

# Préparation des données pour LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, -1])
    return np.array(X), np.array(Y)

look_back = 1
X_train, Y_train = create_dataset(train, look_back)
X_val, Y_val = create_dataset(val, look_back)
X_test, Y_test = create_dataset(test, look_back)

# Définition du modèle LSTM
def create_model(optimizer='adam', neurons=50, dropout_rate=0.2, weight_regularizer=None):
    model = Sequential()
    model.add(LSTM(neurons, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]),
                   kernel_regularizer=weight_regularizer))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(neurons, return_sequences=False, kernel_regularizer=weight_regularizer))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

model = KerasRegressor(build_fn=create_model, verbose=0)

# Optimisation des hyperparamètres
param_grid = dict(optimizer=['adam', 'rmsprop'], neurons=[50, 100], dropout_rate=[0.2, 0.3],
                  weight_regularizer=[None, l1(0.01), l2(0.01)], batch_size=[32, 64], epochs=[50, 100])
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, Y_train)

# Print out the results
print(f'Best: {grid_result.best_score_} using {grid_result.best_params_}')

# Entraînement du modèle avec les meilleurs hyperparamètres
model = create_model(**grid_result.best_params_)
history = model.fit(X_train, Y_train, epochs=grid_result.best_params_['epochs'],
                    batch_size=grid_result.best_params_['batch_size'], verbose=2, validation_data=(X_val, Y_val))

# Prédiction
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inversion des prédictions
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

# Calcul de l'erreur quadratique moyenne
print('Train Mean Squared Error:', mean_squared_error(Y_train[0], train_predict[:,0]))
print('Test Mean Squared Error:', mean_squared_error(Y_test[0], test_predict[:,0]))

# Plotting loss
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show();

