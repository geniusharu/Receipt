import pandas as pd
import numpy as np
import os, glob

from PIL import Image
from tqdm import tqdm

class MakeDataSetBase(object):

    """
    root_dir以下のjpg画像からデータセットを生成するベースクラス
    """

    def __init__(self, root_dir, image_size_w, image_size_h):
        self.filepaths = glob.glob(root_dir + "/*.jpg") #ファイルパス
        self.filenames = os.listdir(root_dir) #ファイル名

        self.image_size_w = 64 #画像をリサイズする横幅
        self.image_size_h = 64 #画像をリサイズする縦幅

    def imagePreprocessing(self, filepath):
        #画像の前処理
        img = Image.open(filepath)
        img = img.convert("RGB") #RGBに変換
        img = img.resize((self.image_size_w, self.image_size_h)) #画像をリサイズ
        return np.asarray(img).astype(float)/255 # numpy array形式へ変換したものを返す

    def getDataSet(self):
        return None

class MakeSingleDataSet(MakeDataSetBase):

    """
    root_dir以下のjpg画像から単一のデータセットを生成するクラス

    input:
        root_dir
        image_size_w
        image_size_h
        labelpath
    output:
        (X, Y)
    """

    def __init__(self, root_dir, image_size_w, image_size_h):
        MakeDataSetBase.__init__(self, root_dir, image_size_w, image_size_h)
        self.labeldata = pd.read_csv('train.tsv',sep='\t') #ラベルのデータ
        self.categories = ['ファミリマート', 'ファミマ!!', 'サンクス', 'サークルK'] # カテゴリー名
        self.categories_dict = {'ファミリマート':0, 'ファミマ!!':1, 'サンクス':2, 'サークルK':3} # カテゴリー名のdict

    def getLabel(self, filename):
        # labeldataから当該ファイルのラベルを検索して返す
        df = self.labeldata
        label = df['cn_name'][df.file_name == filename[:-8] + ".jpg"] # ファイル名の前半部分からラベルを検索
        label_num = self.categories_dict[label.iloc[0]]
        return label_num

    def getDataSet(self):
        Y=[]
        X=[]
        for fp, fn in tqdm(zip(self.filepaths, self.filenames)):
            try:
                y = self.getLabel(fn)
                x = self.imagePreprocessing(fp)
                Y.append(y)
                X.append(x)
            except OSError:
                continue
            except IndexError:
                continue
        return (X,Y)

class MakeMultiDataSets(MakeSingleDataSet):
    """
    root_dir以下のjpg画像を分割し、複数のデータセットを生成するクラス
    """

    def __init__(self, root_dir, image_size_w, image_size_h, nb_datasets, isSaveData=False):
        MakeSingleDataSet.__init__(self, root_dir, image_size_w, image_size_h)
        self.nb_datasets =nb_datasets # 分割するデータセット数
        self.isSaveData = isSaveData #データを保存するかどうか

    def getDataSet(self):
        filepathlist = np.array_split(self.filepaths, self.nb_datasets)
        filenamelist = np.array_split(self.filenames, self.nb_datasets)
        res = []
        cnt=0
        for filepaths, filenames in zip(filepathlist, filenamelist):
            X=[]
            Y=[]
            for fp, fn in tqdm(zip(filepaths, filenames)):
                try:
                    y = self.getLabel(fn)
                    x = self.imagePreprocessing(fp)
                    Y.append(y)
                    X.append(x)
                except OSError:
                    continue
                except IndexError:
                    continue
            res.append((X,Y))
            if self.isSaveData:
                np.save("data" + str(cnt) +".npy", (X,Y)) #分割したデータを保存
            cnt+=1
            del (X,Y)
        return res

class MakeTestDataSet(MakeDataSetBase):
    """
    root_dir以下のjpg画像からテスト用のラベル無しデータセットを生成する。
    """

    def __init__(self, root_dir, image_size_w, image_size_h):
        MakeDataSetBase.__init__(self, root_dir, image_size_w, image_size_h)

    def getDataSet(self):
        X=[]
        for fp, fn in tqdm(zip(self.filepaths, self.filenames)):
            try:
                x = self.imagePreprocessing(fp)
                X.append(x)
            except OSError:
                continue
            except IndexError:
                continue
        return X

if __name__ == '__main__':
    ROOT_DIR = "./rotateimage"
    IMAGE_SIZE_W = 64
    IMAGE_SIZE_H = 64

    md = MakeMultiDataSets(ROOT_DIR, IMAGE_SIZE_W, IMAGE_SIZE_H, 10, isSaveData=True)
    datasets = md.getDataSet()
