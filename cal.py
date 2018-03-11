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
        raw_data = Image.open(src + "/" + pict_path)
        rotate_90 = raw_data.rotate(90, expand = True)
        rotate_90.save("./rotateimage/" + pict_path[:-4] +"_90.jpg")
        rotate_180 = raw_data.rotate(180, expand=True)
        rotate_180.save("./rotateimage/" + pict_path[:-4] + "_180.jpg")
        rotate_270 = raw_data.rotate(270, expand=True)
        rotate_270.save("./rotateimage/" + pict_path[:-4] + "_270.jpg")
        break


if __name__ == "__main__":
    createRotateImage("./train")