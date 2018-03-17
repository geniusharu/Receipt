from PIL import Image
import os
import cv2
from tqdm import tqdm
import pandas as pd
from copy import deepcopy
import numpy as np
import math

# グレースケールへ変換
def getGrayScaleImage(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二値画像へ変換
def getThresholdImage(img_gray):
    img_blur = cv2.fastNlMeansDenoising(img_gray)
    img_th = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 15, 5)
    return img_th

# 切り取り用の二値画像を生成
def getThresholdImageForContours(path):
    img = cv2.imread(path)

    # ゼロ埋めの画像配列
    if len(img.shape) == 3:
        height, width, channels = img.shape[:3]
    else:
        height, width = img.shape[:2]
    zeros = np.zeros((height, width), img.dtype)

    # RGB分離
    img_blue_c1, img_green_c1, img_red_c1 = cv2.split(img)

    vmax = max(img_red_c1.max(),img_blue_c1.max(),img_green_c1.max())
    mean = max(img_red_c1.mean(), img_blue_c1.mean(), img_green_c1.mean())
    if vmax == 255 and mean > vmax*0.6:
        threshold = vmax*0.7
    else:
        threshold = mean

    # 変換用のlambda関数
    f = lambda x: 255 if x > threshold else 0

    # thresholdで二値のarrayに変換
    img_red_c1 = np.frompyfunc(f,1,1)(img_red_c1)
    img_blue_c1 = np.frompyfunc(f,1,1)(img_blue_c1)
    img_green_c1 = np.frompyfunc(f,1,1)(img_green_c1)

    img_blue_c3 = cv2.merge((img_blue_c1, zeros, zeros))
    img_green_c3 = cv2.merge((zeros, img_green_c1, zeros))
    img_red_c3 = cv2.merge((zeros, zeros, img_red_c1))

    img_red2gray = cv2.cvtColor(img_red_c3,  cv2.COLOR_BGR2GRAY)
    img_blue2gray = cv2.cvtColor(img_blue_c3, cv2.COLOR_BGR2GRAY)
    img_green2gray = cv2.cvtColor(img_green_c3, cv2.COLOR_BGR2GRAY)

    img_white = img_red2gray + img_blue2gray + img_green2gray

    return img_white

# 輪郭を抽出するための関数(拾い物)
def getRectByPoints(points):
    points = list(map(lambda x: x[0], points))

    points = sorted(points, key=lambda x:x[1])
    top_points = sorted(points[:2], key=lambda x:x[0])
    bottom_points = sorted(points[2:4], key=lambda x:x[0])
    points = top_points + bottom_points

    left = min(points[0][0], points[2][0])
    right = max(points[1][0], points[3][0])
    top = min(points[0][1], points[1][1])
    bottom = max(points[2][1], points[3][1])
    return (top, bottom, left, right)

# 切り取ります
def getContours(img_th,img_th_for_contours):
    img_th_for_contours2 = deepcopy(img_th_for_contours)
    image, contours, hierarchy = cv2.findContours(img_th_for_contours2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    tmp_dict = {i:cv2.contourArea(contours[i]) for i in range(len(contours))}
    tmp_df = pd.DataFrame.from_dict(tmp_dict,orient="index")
    tmp_df.columns = ["area"] # 面積
    tmp_df.loc[:,"rank"] = tmp_df.loc[:,"area"].rank(ascending=False,method='max')
    for ix in tmp_df[tmp_df.loc[:,"rank"] == 1.0].index: # 面積が大きい囲いのindexを取得
        id = ix

    def getPartImageByRect(rect):
        return img_th[rect[0]:rect[1], rect[2]:rect[3]]

    cnt =contours[id]
    arclen = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02*arclen, True)
    rect = getRectByPoints(approx)
    return getPartImageByRect(rect)

def createRotateImage(src):

    if os.path.exists(src):
        pass
    else:
        print(src + "フォルダが存在しないため終了します")
        quit()
    if os.path.exists("./rotateimage"):
        pass
    else:
        print("rotateimageフォルダを作成します")
        os.mkdir("./rotateimage")

    pict_name_list = os.listdir(src)
    #cnt = 0
    for pict_name in tqdm(sorted(pict_name_list)):

        # 画像をグレースケールへ変換
        img_gray = getGrayScaleImage(src + "/" + pict_name)

        # グレースケール画像を二値画像へ変換
        img_th = getThresholdImage(img_gray)

        # 切り取り用の二値画像を別途用意
        img_th_for_contours = getThresholdImageForContours(src + "/" + pict_name)

        # 切り取り
        img_ext = getContours(img_th,img_th_for_contours)

        # PIL形式へ変換
        raw_data = Image.fromarray(img_ext)

        rotate_000 = raw_data.rotate(0, expand = True) # 回転させない画像も保存するように変更
        rotate_000.save("./rotateimage/" + pict_name[:-4] +"_000.jpg")
        rotate_090 = raw_data.rotate(90, expand = True)
        rotate_090.save("./rotateimage/" + pict_name[:-4] +"_090.jpg")
        rotate_180 = raw_data.rotate(180, expand=True)
        rotate_180.save("./rotateimage/" + pict_name[:-4] + "_180.jpg")
        rotate_270 = raw_data.rotate(270, expand=True)
        rotate_270.save("./rotateimage/" + pict_name[:-4] + "_270.jpg")

        #print(pict_path + ' finished') # 変換の進捗確認用に追加
        #cnt = cnt + 1
        #if cnt > 4:
        #    break

if __name__ == "__main__":
    createRotateImage("./train")
