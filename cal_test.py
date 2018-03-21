import cv2
import math
import numpy as np
import pandas as pd
import os

from copy import deepcopy
from PIL import Image
from tqdm import tqdm

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

    # 元画像の明度を計測
    brightness_list = np.max(img, axis=2) /2.0 + np.min(img, axis=2) /2.0 # numpy使って爆速で走るように変更
    brightness_avg = np.mean(brightness_list)
    brightness_median = np.median(brightness_list)
    brightness = min(brightness_median, brightness_avg)

    threshold = brightness

    # ペナルティーを計測白っぽければRGBのばらつきが少ないという想定
    penalty_list = np.std(img, axis=2) # numpy使って爆速で走るように変更
    avg_std = np.mean(penalty_list)
    std_std = np.std(penalty_list)
    f1 = lambda x: 0 if x > avg_std + 2 * std_std else 1
    penalty_list = np.frompyfunc(f1,1,1)(penalty_list).astype("uint8")
    penalty_array = np.reshape(penalty_list,(img.shape[0],img.shape[1]))

    # 明度がひくい点とペナルティーが大きい点を0にして行きます。
    my_result_tmp = cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY) * penalty_array
    f2 = lambda x: x//1
    my_result_tmp = np.frompyfunc(f2,1,1)(my_result_tmp).astype('uint8')
    f3 = lambda x: x if x > threshold else 0
    my_result = np.frompyfunc(f3,1,1)(my_result_tmp).astype('uint8')

    return my_result

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
    if os.path.exists("./rotateimage_test"):
        pass
    else:
        print("rotateimageフォルダを作成します")
        os.mkdir("./rotateimage_test")

    pict_name_list = os.listdir(src)
    cnt = 0
    for pict_name in tqdm(sorted(pict_name_list)):

        # 画像をグレースケールへ変換
        img_gray = getGrayScaleImage(src + "/" + pict_name)

        # グレースケール画像を二値画像へ変換
        img_th = getThresholdImage(img_gray)

        # 切り取り用の二値画像を別途用意
        img_th_for_contours = getThresholdImageForContours(src + "/" + pict_name)

        tmp = Image.fromarray(img_th_for_contours)
#        tmp.save("test_" + pict_name)

        # 切り取り
        try:
            # とりあえず実行できるようにindex error時はスキップするように変更しときます。
            img_ext = getContours(img_th,img_th_for_contours)
        except IndexError:
            continue

        # PIL形式へ変換
        raw_data = Image.fromarray(img_ext)

        try:
            rotate_000 = raw_data.rotate(0, expand = True) # 回転させない画像も保存するように変更
            rotate_000.save("./rotateimage_test/" + pict_name[:-4] +"_000.jpg")
            rotate_090 = raw_data.rotate(90, expand = True)
            rotate_090.save("./rotateimage_test/" + pict_name[:-4] +"_090.jpg")
            rotate_180 = raw_data.rotate(180, expand=True)
            rotate_180.save("./rotateimage_test/" + pict_name[:-4] + "_180.jpg")
            rotate_270 = raw_data.rotate(270, expand=True)
            rotate_270.save("./rotateimage_test/" + pict_name[:-4] + "_270.jpg")
        except SystemError:
            continue

        #print(pict_path + ' finished') # 変換の進捗確認用に追加
#        cnt = cnt + 1
#        if cnt > 10:
#            break

if __name__ == "__main__":
    createRotateImage("./test")
