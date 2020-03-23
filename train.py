from helper import *

movementDataLoader = DataReader('Movementdata')
train_data, valid_data = movementDataLoader.get_data()
(train_x, train_y), (valid_x, valid_y) = movementDataLoader.more_more_processing()

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input

model = Sequential()
#model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model.add(Bidirectional(LSTM(units=64)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', f1_m])

history = model.fit(train_x, train_y, validation_split=0.1, epochs=10)  # starts training