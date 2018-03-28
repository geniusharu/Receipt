import numpy as np

from keras.initializers import TruncatedNormal, Constant
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization
from keras.callbacks import Callback, EarlyStopping
from keras.utils.np_utils import to_categorical

from makedata import MakeMultiDataSets

"""
keras内蔵のVGG16を用い、レシートデータを学習データとしてパラメータのfine-tuningを行う。
"""

ROOT_DIR = "./rotateimage"
CATEGORIES = ['ファミリマート', 'ファミマ!!', 'サンクス', 'サークルK']
CATEGORIES_DICT = {'ファミリマート':0, 'ファミマ!!':1, 'サンクス':2, 'サークルK':3}
NB_CLASSES = len(CATEGORIES)
IMAGE_SIZE_W = 64
IMAGE_SIZE_H = 64

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
    md = MakeMultiDataSets(ROOT_DIR, IMAGE_SIZE_W, IMAGE_SIZE_H, 10)
    datasets = md.getDataSet()
    for i, d in enumerate(datasets):
        X, Y = d
        X = np.array(X)
        Y = to_categorical(Y, NB_CLASSES)

        model = AlexNet()

        # モデルのコンパイル
        model.compile(optimizer=SGD(lr=0.01),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # モデルの推定
        history = model.fit(X, Y,
                            nb_epoch=10,
                            verbose=1)

        # save model
        model.save('CNN_' + str(i) + '.h5')

if __name__ == '__main__':
    main()
