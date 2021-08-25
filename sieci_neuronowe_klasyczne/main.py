import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split # scikit - learn biblioteka do kodowania
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
#ODCZYT DANYCH
df = pd.read_csv('textures_data.csv', sep=',') #wczytanie danych
data = df.to_numpy() # zamiana na tablice typu numpy

x = data[:, :-1].astype('float') # wyodrebnienie cech
y = data[:, -1] # etykiety kategorii w osobnej kolumnie

label_encoder = LabelEncoder() # obiekt zamieniający etykiety na dane typu integer
integer_encoded = label_encoder.fit_transform(y) # tablica etykiet zakodowana binarnie

onehot_encoder = OneHotEncoder(sparse=False) # zamiana na kodowanie 1 z N
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

x_train, x_test, y_train, y_test = train_test_split(x, onehot_encoded, test_size=0.3)
#TWORZENIE MODELU SIECI
model = Sequential() # model można wykorzystac nie tylko w sieciach klasycznych ale i głębokich - info
model.add(Dense(10, input_dim=72, activation='sigmoid')) # dodane dwie warstwy do modelu sieci. '10' - liczba neuronów,input_dim-rozmiar wektora wejściowego,sygmoidalna funkcja aktywacji
model.add(Dense(3, activation='softmax')) # wyjście - 3 neurony, uogólniona funckja aktywacji
#BUDOWA MODELU
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.summary() #wypisanie wyniku

model.fit(x_train, y_train, epochs=100, batch_size=10, shuffle=True) # podobieństwo do klasyfikatora SVM (poprzednie zadanie). 100 kroków nauczania

y_pred = model.predict(x_test)
y_pred_int = np.argmax(y_pred, axis=1) # przejście z reprezentacji binarnej do całkowitej w celu przekazania do macierzy confiusion_matrix
y_test_int = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test_int, y_pred_int)
print(cm)
