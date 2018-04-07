import numpy as np
import matplotlib.pyplot as plt

from sklearn import cross_validation
from keras.initializers import TruncatedNormal, Constant
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization
from keras.callbacks import Callback, EarlyStopping
from keras.utils.np_utils import to_categorical

from makedata import MakeMultiDataSets, MakeSingleDataSet

"""
AlexNetを使い、画像データから店舗種別を判別する。
"""

ROOT_DIR = "./rotateimage_num_items"
#ROOT_DIR = "./train2"
CATEGORIES = [1,2,3,4]
#CATEGORIES_DICT = {'ファミリマート':0, 'ファミマ!!':1, 'サンクス':2, 'サークルK':3}
NB_CLASSES = len(CATEGORIES)
IMAGE_SIZE_W = 128
IMAGE_SIZE_H = 128

# 拾い物
def conv2d(filters, kernel_size, strides=1, bias_init=1, **kwargs):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=bias_init)
    return Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding='same',
        activation='relu',
        kernel_initializer=trunc,
        bias_initializer=cnst,
        **kwargs
    )

def dense(units, **kwargs):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=1)
    return Dense(
        units,
        activation='tanh',
        kernel_initializer=trunc,
        bias_initializer=cnst,
        **kwargs
    )

def AlexNet():
    model = Sequential()

    # 第1畳み込み層
    model.add(conv2d(96, 11, strides=(4,4), bias_init=0, input_shape=(IMAGE_SIZE_H, IMAGE_SIZE_W, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 第２畳み込み層
    model.add(conv2d(256, 5, bias_init=1))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 第３~5畳み込み層
    model.add(conv2d(384, 3, bias_init=0))
    model.add(conv2d(384, 3, bias_init=1))
    model.add(conv2d(256, 3, bias_init=1))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 密結合層
    model.add(Flatten())
    model.add(dense(4096))
    model.add(Dropout(0.5))
    model.add(dense(4096))
    model.add(Dropout(0.5))

    # 読み出し層
    model.add(Dense(NB_CLASSES, activation='softmax'))

    model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
#    md = MakeMultiDataSets(ROOT_DIR, IMAGE_SIZE_W, IMAGE_SIZE_H, 10, isSaveData=True)
#    md = MakeSingleDataSet(ROOT_DIR, IMAGE_SIZE_W, IMAGE_SIZE_H)
#    X, Y = md.getDataSet()
    X = np.load('./num_items/dataX_num_items.npy')
    Y = np.load('./num_items/dataY_num_items.npy')
#    trX, valX, trY, valY = cross_validation.train_test_split(X, Y, test_size = 0.3)
#    for i in range(10):
#        X = np.load("./cn_name/dataX" + str(i) + ".npy")
#        Y = np.load("./cn_name/dataY" + str(i) + ".npy")
#    for i, d in enumerate(datasets):

#    trX = np.array(trX)
#    trY = to_categorical(trY, NB_CLASSES)

#    valX = np.array(valX)
#    valY = to_categorical(valY, NB_CLASSES)

    X = np.array(X)
    Y = to_categorical(Y, NB_CLASSES)

    model = AlexNet()

    # モデルのコンパイル
    model.compile(optimizer=SGD(lr=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # モデルの推定
    history = model.fit(X, Y,
                        nb_epoch=20,
                        verbose=1)#,
#                        validation_data=(valX, valY))

    # save model
    model.save('./num_items/CNN_num_items.h5')

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

#        del model #メモリからmodelを消去
#        del X
#        del Y

if __name__ == '__main__':
    main()
