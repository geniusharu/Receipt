from PIL import Image
#import cv2
import os

def createRotateImage():
    pict_path_list = os.listdir("./train")
    for pict_path in pict_path_list:
        raw_data = Image.open("./train/" + pict_path)
        rotate_90 = raw_data.rotate(90, expand = True)
        rotate_90.save("./rotateimage/" + pict_path[:-4] +"_90.jpg")
        break


if __name__ == "__main__":
    createRotateImage()