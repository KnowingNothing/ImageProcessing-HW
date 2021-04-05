import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math


def scale_image(img, u=0, v=255):
    max_val = np.max(img)
    min_val = np.min(img)
    res = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res[i][j] = u + v * (
                img[i][j] - min_val) / (max_val - min_val + 1e-10)
    return res


def average_filter_compute_impl(
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


def geomean_filter_compute_impl(
        src, filter_size, pad=cv2.BORDER_REPLICATE,
        value=0, accum_type=np.float32,
        clip_on_the_fly=False):
    assert len(filter_size) == 2, "Expect 2D filter"
    R, S = filter_size
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
    for c in range(C):
        for i in range(res.shape[0]):
            def func(j):
                val = 1
                count = 0
                for r in range(R):
                    for s in range(S):
                        if len(padded.shape) == 2:
                            if padded[i + r, j + s] != 0:
                                val *= padded[i + r, j + s]
                                count += 1
                        else:
                            if padded[i + r, j + s, c] != 0:
                                val *= padded[i + r, j + s, c]
                                count += 1
                if count == 0:
                    res[i, j, c] = 0
                else:
                    res[i, j, c] = math.pow(val, 1 / (R + count))
                return j
            list(map(func, range(res.shape[1])))

    if grey:
        res = res.reshape([H, W])
    return np.clip(res, 0, 255).astype(src.dtype)


def adaptive_med_filter_compute_impl(
        src, filter_size, pad=cv2.BORDER_REPLICATE,
        value=0, accum_type=np.float32,
        clip_on_the_fly=False):
    assert len(filter_size) == 2, "Expect 2D filter"
    R, S = filter_size
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
    for c in range(C):
        for i in range(res.shape[0]):
            def func(j):
                RR = 2
                SS = 2
                while RR <= R and SS <= S:
                    if len(padded.shape) == 2:
                        slc = padded[i:i+RR, j:j+SS]
                        cur = padded[i, j]
                    else:
                        slc = padded[i:i+RR, j:j+SS, c]
                        cur = padded[i, j, c]
                    med = np.median(slc)
                    max_v = np.max(slc)
                    min_v = np.min(slc)
                    if med > min_v and med < max_v:
                        if cur > min_v and cur < max_v:
                            res[i, j, c] = cur
                        else:
                            res[i, j, c] = med
                        break
                    else:
                        RR += 1
                        SS += 1
                return j
            list(map(func, range(res.shape[1])))

    if grey:
        res = res.reshape([H, W])
    return np.clip(res, 0, 255).astype(src.dtype)


def another_geomean_filter_compute_impl(
        src, filter_size, pad=cv2.BORDER_REPLICATE,
        value=0, accum_type=np.float32,
        clip_on_the_fly=False):
    assert len(filter_size) == 2, "Expect 2D filter"
    R, S = filter_size
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
    res = np.ones([H, W, C], dtype=accum_type)
    padded = cv2.copyMakeBorder(
                src, pad_h, pad_h, pad_w, pad_w, pad, value=value
            ).astype(accum_type)
    padded = np.maximum(1, padded)
    padded = padded / 255
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
            res = res * (dup).astype(accum_type)
            if clip_on_the_fly:
                res = np.clip(res, 0, 255)

    res = np.power(res, 1 / (R + S))
    res = res * 255
    # res = np.power(math.e, res)
    # res = np.power(res, 1.0 / (R * S))
    # res = scale_image(res, 0, 255)
    if grey:
        res = res.reshape([H, W])
    res = np.clip(res, 0, 255).astype(src.dtype)
    return res


def process(file_path):
    # BGR
    img = cv2.imread(file_path)
    # img = np.random.uniform(0, 255, [8, 8]).astype(np.uint8)
    assert img is not None, file_path
    kernel = np.ones([7, 7]) / (7 * 7)

    avg = average_filter_compute_impl(
        img, kernel, pad=cv2.BORDER_REPLICATE, accum_type=np.float64
    )

    geo = geomean_filter_compute_impl(
        img, [7, 7], pad=cv2.BORDER_REPLICATE, accum_type=np.float64)

    adp = adaptive_med_filter_compute_impl(
        img, [7, 7], pad=cv2.BORDER_REPLICATE, accum_type=np.float64)
        
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    # fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle("Image Reconstruction")
    if len(img.shape) == 2:
        ax1.imshow(img)
    else:
        ax1.imshow(img[:, :, [2,1,0]])
    ax1.title.set_text("Original")
    if len(avg.shape) == 2:
        ax2.imshow(avg)
    else:
        ax2.imshow(avg[:, :, [2,1,0]])
    ax2.title.set_text("Average Filtered")
    if len(geo.shape) == 2:
        ax3.imshow(geo)
    else:
        ax3.imshow(geo[:, :, [2,1,0]])
    ax3.title.set_text("Geomean Filtered")
    if len(adp.shape) == 2:
        ax4.imshow(adp)
    else:
        ax4.imshow(adp[:, :, [2,1,0]])
    ax4.title.set_text("Adaptive Filtered")
    plt.show()


def Image_Restoration_Average_Filter_solve(img):
    kernel = np.ones([7, 7]) / (7 * 7)

    avg = average_filter_compute_impl(
        img, kernel, pad=cv2.BORDER_REPLICATE, accum_type=np.float64
    )

    return avg


def Image_Restoration_Geometirc_Mean_Filter_solve(img):
    geo = geomean_filter_compute_impl(
        img, [7, 7], pad=cv2.BORDER_REPLICATE, accum_type=np.float64)

    return geo


def Image_Restoration_Adaptive_Median_solve(img):
    adp = adaptive_med_filter_compute_impl(
        img, [7, 7], pad=cv2.BORDER_REPLICATE, accum_type=np.float64)

    return adp


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
    main("../../images/problem6")