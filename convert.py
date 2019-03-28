from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
import coremltools

model = Sequential()
model.add(Conv1D(64, 5, activation='relu', input_shape=(2400,1)))
model.add(MaxPooling1D(pool_size=4))
model.add(Flatten())
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights('model.h5')
model = coremltools.converters.keras.convert(model)
model.save('deepdive.mlmodel')
