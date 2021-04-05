import cv2


def convert2RGB(img_path, out_path):
    img = cv2.imread(img_path)
    cv2.imwrite(out_path, img)