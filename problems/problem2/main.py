import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


def filter_compute_impl(
        src, filters, pad=cv2.BORDER_REPLICATE,
        value=0, accum_type=np.float32,
        clip_on_the_fly=False):
    assert len(filters.shape) == 2, "Expect 2D filter"
    R, S = filters.shape
    if len(src.shape) == 2:
        # grey image
        H, W = src.shape
        C = 1
        grey = True
        src = src.reshape([H, W, 1])
    elif len(src.shape) == 3:
        # BGR image
        H, W, C = src.shape
        grey = False
    pad_h = R // 2
    pad_w = S // 2
    res = np.zeros([H, W, C], dtype=accum_type)
    padded = cv2.copyMakeBorder(
                src, pad_h, pad_h, pad_w, pad_w, pad, value=value
            ).astype(accum_type)
    for r in range(R):
        for s in range(S):
            if len(padded.shape) == 2:
                dup = padded[
                    r:H+r,
                    s:W+s].reshape(res.shape)
            else:
                dup = padded[
                    r:H+r,
                    s:W+s, :]
            res += (dup * filters[r, s]).astype(accum_type)
            if clip_on_the_fly:
                res = np.clip(res, 0, 255)
    if grey:
        res = res.reshape([H, W])
    return np.clip(res, 0, 255).astype(src.dtype)


def process(file_path):
    # BGR
    img = cv2.imread(file_path)
    assert img is not None, file_path
    kernel = np.array(
        [[0, -1, 0],
         [-1, 4, -1],
         [0, -1, 0]]
    )

    lap = filter_compute_impl(img, kernel, pad=cv2.BORDER_REPLICATE)

    fig = plt.figure()
    if (img.shape[0] > img.shape[1]):
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        plt.subplots_adjust(wspace=0.5, hspace=0)
    else:
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        plt.subplots_adjust(wspace=0, hspace=0.5)
    # fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle("Image Enhancement")
    ax1.imshow(img[:, :, [2,1,0]])
    ax1.title.set_text("Original")
    ax2.imshow(lap[:, :, [2,1,0]])
    ax2.title.set_text("Enhanced")
    plt.show()


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