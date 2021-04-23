# web-app for API image manipulation

from flask import Flask, request, render_template, send_from_directory
import os
from PIL import Image
from problems import (
    Laplacian_Filter_solve, Laplacian_Filter_opencv,
    Gaussian_Lowpass_solve,
    Image_Restoration_Average_Filter_solve,
    Image_Restoration_Geometirc_Mean_Filter_solve,
    Image_Restoration_Improved_Geometirc_Mean_Filter_solve,
    Image_Restoration_Adaptive_Median_solve
)
import numpy as np
import cv2
import time

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# default access page
@app.route("/")
def main():
    return render_template('new_index.html')


# upload selected image and forward to processing page
@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/temp_images/')

    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    # retrieve file from html file-picker
    upload = request.files.getlist("file")[0]
    print("File name: {}".format(upload.filename))
    filename = upload.filename

    # file support verification
    ext = os.path.splitext(filename)[1]
    if (
        (ext == ".jpg")
        or (ext == ".png")
        or (ext == ".bmp")
        or (ext == ".tif")
        or (ext == ".npy")
        or (ext == ".jpeg")
        ):
        print("File accepted")
    else:
        return render_template("new_error.html", message="The selected file is not supported"), 400

    # save file
    local_time = time.localtime(time.time())

    filename= "filename-{}-{}".format(local_time,filename)

    destination = "/".join([target, filename])
    if os.path.isfile(destination):
        os.remove(destination)
    print("File saved to:", destination)
    upload.save(destination)

    # forward to processing page
    return render_template("new_processing.html", image_name=filename)


@app.route("/M_T", methods=["POST"])
def M_T():

    # retrieve parameters from html form
    mode = request.form['mode']
    filename = request.form['image']

    # open and process image
    target = os.path.join(APP_ROOT, 'static/temp_images')
    destination = "/".join([target, filename])

    img = cv2.imread(destination, 1)
    if (img[:, :, 0] == img[:, :, 1]).all() and (img[:, :, 1] == img[:, :, 2]).all() and (img[:, :, 0] == img[:, :, 2]).all():
        img = cv2.imread(destination, 0)

    print(img.shape, img.dtype)

    kernel = np.random.randint(0, 2, (3, 3)).astype(np.uint8)
    print(kernel)

    # check mode
    if mode == 'Erosion':
        img = Morphological_Transformation.MyErosion(img, kernel)
    elif mode == 'Dilation':
        img = Morphological_Transformation.MyDilation(img, kernel)
    elif mode == 'Opening':
        img = Morphological_Transformation.MyOpening(img, kernel)
    elif mode == 'Closing':
        img = Morphological_Transformation.MyClosing(img, kernel)
    else:
        return render_template("new_error.html", message="Invalid mode (vertical or horizontal)"), 400


    local_time = time.localtime(time.time())

    result_name = "{}-{}".format(mode, filename)
    destination = "/".join([target, result_name])
    if os.path.isfile(destination):
        os.remove(destination)
    cv2.imwrite(destination, img)

    opencv_name = "opencv-{}-{}".format(mode, filename)
    destination = "/".join([target, opencv_name])
    if os.path.isfile(destination):
        os.remove(destination)
    cv2.imwrite(destination, img)

    kernel_name = "kernel-{}-{}".format(mode, filename)
    destination = "/".join([target, kernel_name])
    if os.path.isfile(destination):
        os.remove(destination)
    kernel = cv2.resize(kernel*255, (256, 256),
                        interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(destination, kernel)

    return render_template("MT_result.html", original_name=filename, kernel_name=kernel_name, opencv_name=opencv_name, result_name=result_name)


@app.route("/S_F", methods=["POST"])
def S_F():
    # retrieve parameters from html form
    mode = request.form['mode']
    filename = request.form['image']

    # open and process image
    target = os.path.join(APP_ROOT, 'static/temp_images')
    destination = "/".join([target, filename])

    img = cv2.imread(destination, 1)
    if (img[:, :, 0] == img[:, :, 1]).all() and (img[:, :, 1] == img[:, :, 2]).all() and (img[:, :, 0] == img[:, :, 2]).all():
        img = cv2.imread(destination, 0)

    print(img.shape, img.dtype)

    # check mode
    if mode == "Laplacian-5x5":
        # kernel = np.array(
        #     [[-1, -1, -1],
        #      [-1, 8, -1],
        #      [-1, -1, -1]]
        # )
        kernel = np.array(
            [[0, 0, -1, 0, 0],
             [0, -1, -2, -1, 0],
             [-1, -2, 16, -2, -1],
             [0, -1, -2, -1, 0],
             [0, 0, -1, 0, 0]]
        )
        print(kernel)
        grad = Laplacian_Filter_solve(img, kernel)
        # kernel = np.array(
        #     [[-1, -1, -1],
        #      [-1, 9, -1],
        #      [-1, -1, -1]]
        # )
        kernel = np.array(
            [[0, 0, -1, 0, 0],
             [0, -1, -2, -1, 0],
             [-1, -2, 17, -2, -1],
             [0, -1, -2, -1, 0],
             [0, 0, -1, 0, 0]]
        )
        print(kernel)
        enhanced = Laplacian_Filter_solve(img, kernel)
        enhanced_opencv = Laplacian_Filter_opencv(img, kernel)
    elif mode == "Laplacian-3x3":
        kernel = np.array(
            [[-1, -1, -1],
             [-1, 8, -1],
             [-1, -1, -1]]
        )
        print(kernel)
        grad = Laplacian_Filter_solve(img, kernel)
        kernel = np.array(
            [[-1, -1, -1],
             [-1, 9, -1],
             [-1, -1, -1]]
        )
        print(kernel)
        enhanced = Laplacian_Filter_solve(img, kernel)
        enhanced_opencv = Laplacian_Filter_opencv(img, kernel)
    else:
        return render_template("new_error.html", message="Invalid mode (Laplacian)"), 400


    local_time = time.localtime(time.time())

    result_name = "{}-{}".format(mode, filename)
    destination = "/".join([target, result_name])
    if os.path.isfile(destination):
        os.remove(destination)
    cv2.imwrite(destination, enhanced)

    opencv_name = "opencv-{}-{}".format(mode, filename)
    destination = "/".join([target, opencv_name])
    if os.path.isfile(destination):
        os.remove(destination)
    cv2.imwrite(destination, enhanced_opencv)

    grad_name = "grad-{}-{}".format(mode, filename)
    destination = "/".join([target, grad_name])
    if os.path.isfile(destination):
        os.remove(destination)
    cv2.imwrite(destination, grad)

    return render_template("SF_result.html", original_name=filename, kernel_name=grad_name, opencv_name=opencv_name, result_name=result_name)


@app.route("/G_L_F", methods=["POST"])
def G_L_F():
    # retrieve parameters from html form
    mode = request.form['mode']
    filename = request.form['image']

    # open and process image
    target = os.path.join(APP_ROOT, 'static/temp_images')
    destination = "/".join([target, filename])

    img = cv2.imread(destination, 1)
    if (img[:, :, 0] == img[:, :, 1]).all() and (img[:, :, 1] == img[:, :, 2]).all() and (img[:, :, 0] == img[:, :, 2]).all():
        img = cv2.imread(destination, 0)

    print(img.shape, img.dtype)

    # check mode
    radius_lst = [5, 15, 30, 80, 230]
    result_lst = []
    if mode == "Gaussian":
        for radius in radius_lst:
            gau, specs = Gaussian_Lowpass_solve(img, radius)
            result_lst.append(gau)
    else:
        return render_template("new_error.html", message="Invalid mode (Gausian)"), 400


    local_time = time.localtime(time.time())

    filenames = []
    for radius, res in zip(radius_lst, result_lst):
        result_name = "{}-radius-{}-{}".format(mode, radius, filename)
        destination = "/".join([target, result_name])
        if os.path.isfile(destination):
            os.remove(destination)
        cv2.imwrite(destination, res)
        filenames.append(result_name)

    return render_template(
        "GLF_result.html",
        original_name=filename,
        radius_5_name=filenames[0],
        radius_15_name=filenames[1],
        radius_30_name=filenames[2],
        radius_80_name=filenames[3],
        radius_230_name=filenames[4])


@app.route("/I_R", methods=["POST"])
def I_R():
    # retrieve parameters from html form
    mode = request.form['mode']
    filename = request.form['image']

    # open and process image
    target = os.path.join(APP_ROOT, 'static/temp_images')
    destination = "/".join([target, filename])

    img = cv2.imread(destination, 1)
    if (img[:, :, 0] == img[:, :, 1]).all() and (img[:, :, 1] == img[:, :, 2]).all() and (img[:, :, 0] == img[:, :, 2]).all():
        img = cv2.imread(destination, 0)

    print(img.shape, img.dtype)

    # check mode
    if mode == "Average":
        res = Image_Restoration_Average_Filter_solve(img)
    elif mode == "Geomean":
        res = Image_Restoration_Geometirc_Mean_Filter_solve(img)
    elif mode == "Improved_Geomean":
        res = Image_Restoration_Improved_Geometirc_Mean_Filter_solve(img)
    elif mode == "Adaptive":
        res = Image_Restoration_Adaptive_Median_solve(img)
    elif mode == "All":
        avg = Image_Restoration_Average_Filter_solve(img)
        geo = Image_Restoration_Improved_Geometirc_Mean_Filter_solve(img)
        adp = Image_Restoration_Adaptive_Median_solve(img)

        local_time = time.localtime(time.time())

        average_name = "average-{}-{}".format(mode, filename)
        destination = "/".join([target, average_name])
        if os.path.isfile(destination):
            os.remove(destination)
        cv2.imwrite(destination, avg)

        geomean_name = "geomean-{}-{}".format(mode, filename)
        destination = "/".join([target, geomean_name])
        if os.path.isfile(destination):
            os.remove(destination)
        cv2.imwrite(destination, geo)

        adaptive_name = "adaptive-{}-{}".format(mode, filename)
        destination = "/".join([target, adaptive_name])
        if os.path.isfile(destination):
            os.remove(destination)
        cv2.imwrite(destination, adp)

        return render_template(
            "IR_All_result.html",
            original_name=filename,
            average_name=average_name,
            geomean_name=geomean_name,
            adaptive_name=adaptive_name)
    else:
        return render_template("new_error.html", message="Invalid mode (Average, Geomean, Adaptive, or All)"), 400


    local_time = time.localtime(time.time())

    result_name = "{}-{}".format(mode, filename)
    destination = "/".join([target, result_name])
    if os.path.isfile(destination):
        os.remove(destination)
    cv2.imwrite(destination, res)

    return render_template(
        "IR_result.html", original_name=filename, result_name=result_name)


# retrieve file from 'static/temp_images' directory
@app.route('/static/temp_images/<filename>')
def send_image(filename):
    parts = filename.split(".")
    if parts[-1] not in ["jpg", "jpeg"]:
        img = cv2.imread(os.path.join("static/temp_images", filename))
        parts[-1] = "jpg"
        filename = ".".join(parts)
        cv2.imwrite(os.path.join("static/temp_images", filename), img)
    return send_from_directory("static/temp_images", filename)


if __name__ == "__main__":
    app.run(debug = True)