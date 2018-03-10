import pandas as pd
import numpy as np
import os, glob

from sklearn import cross_validation
from PIL import Image

# カテゴリなどの設定
root_dir = "./train/"
filepath = 'train.tsv'
categories = ['ファミリマート_0', 'ファミリマート_1', 'ファミマ!!_0', 'サンクス_0', 'ファミマ!!_1', 'サークルK_0']
categories_dict = {'ファミリマート_0':0, 'ファミリマート_1':1, 'ファミマ!!_0':2, 'サンクス_0':3, 'ファミマ!!_1':4, 'サークルK_0':5}
nb_classes = len(categories)
image_size_w = 64
image_size_h = 60

# 新しいラベルを生成する関数
def get_New_cn_name(cn_name, new_flag):
    for cn, nf in zip(cn_name, new_flag):
        yield str(cn) + '_' + str(nf)

# 数値形式のラベルを取得する関数
def getNumericalLabel(cn_name2):
    for cn2 in cn_name2:
        yield categories.index(cn2)

# tsvファイルの加工用
def setData(filepath):
    # データをロード
    df = pd.read_csv(filepath, sep='\t')

    # 日付の変換と新しいラベルの定義
    df.bought_datetime = pd.to_datetime(df.bought_datetime)
    df['cn_name2'] = list(get_New_cn_name(df.cn_name, df.new_flag))

    return df

def main():
    # データを変換
    df = setData(filepath)

    # フォルダ以内のファイル名を取得
    filepaths = glob.glob(root_dir + "/*.jpg")
    filenames = os.listdir(root_dir)

    # 数値形式に変換
    X = [] # 画像データ
    Y = [] # ラベルデータ
    for fp, fn in zip(filepaths, filenames):
        img = Image.open(fp)
        img = img.convert("RGB")
        img = img.resize((image_size_w, image_size_h))
        data = np.asarray(img)
        label = df['cn_name2'][df.file_name == fn]
        label_num = categories_dict[label.iloc[0]]
        X.append(data)
        Y.append(label_num)
        print(fn + ' finished')

    X = np.array(X)
    Y = np.array(Y)

    # 学習データとテストデータを分ける
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, Y, test_size = 0.4)
    xy = (X_train, X_test, y_train, y_test)
    np.save("./train/data.npy", xy)

if __name__ == '__main__':
    main()
