import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils

root_dir = "./train/"
categories = ['ファミリマート_0', 'ファミリマート_1', 'ファミマ!!_0', 'サンクス_0', 'ファミマ!!_1', 'サークルK_0']
nb_classes = len(categories)

def build_model(in_shape):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3,
                            border_mode='same',
                            input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model

def main():
    # load dataset
    trX, valX, trY, valY = np.load("data.npy")

    trX = trX.astype("float") / 255
    valX  = valX.astype("float")  / 255

    trY = np_utils.to_categorical(trY, nb_classes)
    valY  = np_utils.to_categorical(valY, nb_classes)

    # load model
    model = build_model(trX.shape[1:])

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # fit model
    history = model.fit(trX, trY,
                        nb_epoch=10,
                        verbose=2,
                        validation_data=(valX, valY))

    # save model
    model.save('CNN.h5')

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()
