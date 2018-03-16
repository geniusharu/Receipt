from PIL import Image
import os
import cv2
from tqdm import tqdm
import pandas as pd
from copy import deepcopy

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

# 輪郭を抽出
def getContours(img_th):
    img_th2 = deepcopy(img_th)
    image, contours, hierarchy = cv2.findContours(img_th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tmp_dict = {i:cv2.contourArea(contours[i]) for i in range(len(contours))}
    tmp_df = pd.DataFrame.from_dict(tmp_dict,orient="index")
    tmp_df.columns = ["area"] # 面積
    tmp_df.loc[:,"rank"] = tmp_df.loc[:,"area"].rank(ascending=False,method='max')
    for ix in tmp_df[tmp_df.loc[:,"rank"] == 2.0].index: # 面積が大きい囲いのindexを取得
        id = ix
        print(ix)
    
    def getPartImageByRect(rect):
        return img_th[rect[0]:rect[1], rect[2]:rect[3]]

    cnt =contours[ix]
    arclen = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02*arclen, True)
    print(approx)
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

        # 切り取り（まだエラーがあるのでコメントアウト中）
        #img_ext = getContours(img_th)

        # PIL形式へ変換
        raw_data = Image.fromarray(img_th)

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
        #if cnt > 5:
        #    break

if __name__ == "__main__":
    createRotateImage("./train")
