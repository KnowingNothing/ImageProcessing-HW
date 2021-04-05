import cv2
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import os
import math


def gaussian_function(sigma):
    def func(x, y):
        return (
            math.pow(
                math.e,
                -(x * x + y * y) / (2 * sigma * sigma)
            )
        )
    return func


def ideal_function(sigma):
    def func(x, y):
        if math.pow(x*x + y*y, 1/2) <= sigma:
            return 1
        else:
            return 0
    return func


def scale_image(img, u=0, v=255):
    max_val = np.max(img)
    min_val = np.min(img)
    res = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res[i][j] = u + v * (
                img[i][j] - min_val) / (max_val - min_val + 1e-10)
    return res


def gaussian_scale_image(img, u=0, v=255):
    max_val = np.max(img)
    min_val = np.min(img)
    res = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res[i][j] = u + v * (
                img[i][j] - min_val) / (max_val - min_val + 1e-10)
    return res


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
    
    partial_results = []
    bgr_specs = []
    for c in range(C):
        slc = src[:, :, c]
        max_val = np.max(slc)
        min_val = np.min(slc)
        mean_val = np.mean(slc)
        std_val = np.std(slc)
        f = np.fft.fft2(slc)
        fshift = np.fft.fftshift(f)
        spec = 20 * np.log(np.abs(fshift))
        bgr_specs.append(spec)
        filtered = np.multiply(fshift, filters)
        if_shift = np.fft.ifftshift(filtered)
        im = np.fft.ifft2(if_shift)
        # im = np.clip(
        #     scale_image(
        #         np.abs(im), mean_val - 5 * std_val, mean_val + 5 * std_val),
        #     0, 255).astype(src.dtype)
        im = np.clip(
                np.abs(im),
            0, 255).astype(src.dtype)
        partial_results.append(im)
    res = np.stack(partial_results, axis=2)
    specs = np.stack(bgr_specs, axis=2)
    return res, specs


def gaussian_filter_compute_impl(
        src, radius, pad=cv2.BORDER_REPLICATE,
        value=0, accum_type=np.float32,
        clip_on_the_fly=False
):
    sigma = radius
    radius = int(radius)
    GF = gaussian_function(sigma)
    kernel = np.zeros(src.shape[:2])
    sum_val = 0
    X = kernel.shape[0] / 2
    Y = kernel.shape[1] / 2
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            kernel[i][j] = GF(i - X, j - Y)
    return filter_compute_impl(
        src, kernel, pad=pad,
        value=value, accum_type=accum_type,
        clip_on_the_fly=clip_on_the_fly)


def ideal_filter_compute_impl(
        src, radius, pad=cv2.BORDER_REPLICATE,
        value=0, accum_type=np.float32,
        clip_on_the_fly=False
):
    sigma = radius
    radius = int(radius)
    GF = ideal_function(sigma)
    kernel = np.zeros(src.shape[:2])
    X = kernel.shape[0] / 2
    Y = kernel.shape[1] / 2
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            kernel[i][j] = GF(i - X, j - Y)
    return filter_compute_impl(
        src, kernel, pad=pad,
        value=value, accum_type=accum_type,
        clip_on_the_fly=clip_on_the_fly)


def process(file_path):
    # BGR
    img = cv2.imread(file_path)
    assert img is not None, file_path

    gau, specs = gaussian_filter_compute_impl(img, 10, pad=cv2.BORDER_REPLICATE)

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
    fig.suptitle("Gaussian Filter")
    ax1.imshow(img[:, :, [2,1,0]])
    ax1.title.set_text("Original")
    ax2.imshow(gau[:, :, [2,1,0]])
    ax2.title.set_text("Filtered")
    plt.show()


def Gaussian_Lowpass_solve(img, radius):
    gau, specs = gaussian_filter_compute_impl(img, radius, pad=cv2.BORDER_REPLICATE)
    return gau, specs


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
    main("../../images/problem5")