from PIL import Image
import os
import cv2

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

    pict_path_list = os.listdir(src)
    for pict_path in pict_path_list:

        # 画像をグレースケールへ変換
        img_gray = getGrayScaleImage(src + "/" + pict_path)

        # グレースケール画像を二値画像へ変換
        img_th = getThresholdImage(img_gray)

        # PIL形式へ変換
        raw_data = Image.fromarray(img_th)

        rotate_000 = raw_data.rotate(0, expand = True) # 回転させない画像も保存するように変更
        rotate_000.save("./rotateimage/" + pict_path[:-4] +"_000.jpg")
        rotate_090 = raw_data.rotate(90, expand = True)
        rotate_090.save("./rotateimage/" + pict_path[:-4] +"_090.jpg")
        rotate_180 = raw_data.rotate(180, expand=True)
        rotate_180.save("./rotateimage/" + pict_path[:-4] + "_180.jpg")
        rotate_270 = raw_data.rotate(270, expand=True)
        rotate_270.save("./rotateimage/" + pict_path[:-4] + "_270.jpg")

        print(pict_path + ' finished') # 変換の進捗確認用に追加

        break

if __name__ == "__main__":
    createRotateImage("./train")
