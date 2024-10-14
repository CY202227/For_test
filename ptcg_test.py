import cv2
import keyboard
import numpy as np
import pytesseract
import requests
from PIL import Image
import time
from grabscreen import grab_screen


def process_image(image):
    # 将截图转换为灰度图以提高识别率
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 进行二值化处理
    _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
    return thresh_image


def extract_card_name():
    data = {'file': open('./rules01-img122.jpg', 'rb')}
    textdata = requests.post('http://127.0.0.1:5000/ocr', files=data)
    textdata = textdata.json()['data']['Results']
    return (textdata[1])


def main():
    # 捕获屏幕
    time1 = time.time()
    image_array = grab_screen(region=(4, 625, 1900, 1000))
    # 将数组转换为图像
    array_to_image = Image.fromarray(image_array, mode='RGB')

    # 将图像转换为 NumPy 数组供 OpenCV 处理
    open_cv_image = cv2.cvtColor(np.array(array_to_image), cv2.COLOR_RGB2BGR)

    # 处理图像
    processed_image = process_image(open_cv_image)
    cv2.imwrite('processed_image.png', processed_image)  # save to disk

    # 提取卡牌名称
    card_name = extract_card_name()
    time2 = time.time()
    # 输出识别的卡牌名称
    print(f"识别的卡牌名称: {card_name}")
    print(time2 - time1)
    if cv2.waitKey(25) & 0xFF == ord('q'):  # 按q退出，记得输入切成英语再按q
        cv2.destroyAllWindows()
        # break
                         
if __name__ == "__main__":
    while True:
        main()
        if keyboard.is_pressed('esc'):
                break 