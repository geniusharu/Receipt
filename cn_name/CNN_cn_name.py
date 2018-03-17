import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.utils import np_utils
from keras.constraints import maxnorm

root_dir = "./train/"
categories = ['ファミリマート', 'ファミマ!!', 'サンクス', 'サークルK']
nb_classes = len(categories)

def build_model(in_shape):
    model = Sequential()
    model.add(Convolution2D(64, 3, 3,
                            border_mode='same',
                            input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3, 3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model

def build_model2(in_shape):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=in_shape,
                            border_mode='same', activation='relu',
                            W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu',
                            border_mode='same', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model

def build_model3(in_shape):
    input = Input(in_shape)
    kernel_size = (3, 3)
    max_pool_size = (2, 2)

    cnn = Convolution2D(64, kernel_size, padding='same', activation='relu')(input)
    cnn = Dropout(0.1)(cnn)
    cnn = Convolution2D(64, kernel_size, padding='same', activation='relu')(cnn)
    cnn = Dropout(0.1)(cnn)
    cnn = Convolution2D(64, kernel_size, padding='same', activation='relu')(cnn)
    cnn = Dropout(0.1)(cnn)
    cnn = MaxPooling2D(pool_size=max_pool_size, strides=(2, 2))(cnn)

    cnn = Convolution2D(64, kernel_size, padding='same', activation='relu')(cnn)
    cnn = Dropout(0.1)(cnn)
    cnn = Convolution2D(64, kernel_size, padding='same', activation='relu')(cnn)
    cnn = Dropout(0.1)(cnn)
    cnn = Convolution2D(64, kernel_size, padding='same', activation='relu')(cnn)
    cnn = MaxPooling2D(pool_size=max_pool_size, strides=(2, 2))(cnn)

    fc = Flatten()(cnn)
    fc = Dense(1024, activation='relu')(fc)
    softmax = Dense(nb_classes, activation='softmax')(fc)
    model = Model(input=input, output=softmax)

    return model

def main():
    # load dataset
    trX, valX, trY, valY = np.load("data.npy")

    trX = trX.astype("float") / 255
    valX  = valX.astype("float")  / 255

    trY = np_utils.to_categorical(trY, nb_classes)
    valY  = np_utils.to_categorical(valY, nb_classes)

    # load model
    print(trX.shape)
    model = build_model2(trX.shape[1:])

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # fit model
    history = model.fit(trX, trY,
                        nb_epoch=1,
                        verbose=1,
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
