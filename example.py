import numpy as np
import cv2
import time
from grabscreen import grab_screen
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 爆显存的话可以在此调整程序的显存占用情况
session = tf.compat.v1.Session(config=config)

yolo = YOLO("yolo11n.pt")

while True:

    image_array = grab_screen(region=(0, 0, 1280, 720))
    # 获取屏幕，(0, 0, 1280, 720)表示从屏幕坐标（0,0）即左上角，截取往右1280和往下720的画面
    array_to_image = Image.fromarray(image_array, mode='RGB')  # 将array转成图像，才能送入yolo进行预测
    img = yolo.predict(array_to_image)  # 调用yolo文件里的函数进行检测

    for result in img:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="result.jpg")  # save to disk

    #cv2.imshow('window'q, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 将截取的画面从另一窗口显示出来，对速度会有一点点影响，不过也就截取每帧多了大约0.01s的时间
    if cv2.waitKey(25) & 0xFF == ord('q'):  # 按q退出，记得输入切成英语再按q
        cv2.destroyAllWindows()
        break
