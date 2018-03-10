from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np

root_dir = "./train/"
categories = ['ファミリマート_0', 'ファミリマート_1', 'ファミマ!!_0', 'サンクス_0', 'ファミマ!!_1', 'サークルK_0']
nb_classes = len(categories)

def main():
    X_train, X_test, y_train, y_test = np.load("data.npy")

    X_train = X_train.astype("float") / 255
    X_test  = X_test.astype("float")  / 255

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test  = np_utils.to_categorical(y_test, nb_classes)

    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)

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
    model.compile(loss='categorical_crossentropy',
	optimizer='rmsprop',
	metrics=['accuracy'])
    return model

def model_train(X, y):
    model = build_model(X.shape[1:])
    model.fit(X, y, batch_size=32, nb_epoch=30)
    hdf5_file = "model.hdf5"
    model.save_weights(hdf5_file)
    return model

def model_eval(model, X, y):
    score = model.evaluate(X, y)
    print('loss=', score[0])
    print('accuracy=', score[1])
    print (time.time()-start_time)

if __name__ == "__main__":
    main()
