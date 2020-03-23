from helper import *

movementDataLoader = DataReader('Movementdata')
train_data, valid_data = movementDataLoader.get_data()

print(len(train_data), len(valid_data))