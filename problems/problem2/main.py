import cv2
import os


def process(file_path):
    img = cv2.imread(file_path)
    cv2.imshow("Image", img)


def main(image_path):
    assert os.path.exists(image_path) and os.path.isdir(image_path)
    for file in os.listdir(image_path):
        file_path = os.path.join(image_path, file)
        print(file_path)
        assert os.path.isfile(file_path)
        try:
            process(file_path)
        except Exception as e:
            print("can't process image:", file_path)
            print(e)


if __name__ == "__main__":
    main("../../images/problem2")