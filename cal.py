from PIL import Image
import os

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

        print(pict_path) # 変換の進捗確認用に追加

        raw_data = Image.open(src + "/" + pict_path)

        rotate_000 = raw_data.rotate(0, expand = True) # 回転させない画像も保存するように変更
        rotate_000.save("./rotateimage/" + pict_path[:-4] +"_000.jpg")
        rotate_090 = raw_data.rotate(90, expand = True)
        rotate_090.save("./rotateimage/" + pict_path[:-4] +"_090.jpg")
        rotate_180 = raw_data.rotate(180, expand=True)
        rotate_180.save("./rotateimage/" + pict_path[:-4] + "_180.jpg")
        rotate_270 = raw_data.rotate(270, expand=True)
        rotate_270.save("./rotateimage/" + pict_path[:-4] + "_270.jpg")
#        break

if __name__ == "__main__":
    createRotateImage("./train")
