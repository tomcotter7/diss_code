import numpy as np
import cv2 as cv
from PIL import Image
import tensorflow as tf
import numpy as np
import os


# function to preprocess an image to run through the gradcam++ algorithm
def preprocess_data_for_grad_cam(img_path):
    img = cv.imread(img_path)
    img = crop_image_otsu(img)
    resized_img = cv.resize(img, (512, 512), interpolation=cv.INTER_LINEAR)
    cv.imwrite(img_path, resized_img)
    return resized_img


# function to crop the black background from 2-d fundus images
def crop_image_otsu(img):

    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresholded = cv.threshold(grayscale, 0, 255, cv.THRESH_OTSU)
    bbox = cv.boundingRect(thresholded)
    x, y, w, h = bbox
    foreground = img[y:y+h, x:x+w]
    return foreground


# function that takes output of gradcam++ and overlays it over the correct part of the image
def overlap_heatmap(img_path, heatmap, alpha):
    heatmap = np.asarray(heatmap)
    img = cv.imread(img_path)
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    heatmap = cv.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = createBoxes(heatmap)

    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    # heatmap = cv.cvtColor(heatmap, cv.COLOR_BGR2RGB)
    # heatmap = createBoxes(heatmap)
    # boxes_img = boxes * 0.25 + img
    # superimposed_img_boxes = np.clip(boxes_img, 0, 255).astype("uint8")
    # superimposed_img_boxes = cv.cvtColor(superimposed_img_boxes, cv.COLOR_BGR2RGB)
    # imgwithboxes = Image.fromarray(superimposed_img_boxes)
    # imgwithboxes = imgwithboxes.resize((256, 256), Image.BILINEAR)

    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    cv.imshow("sii", superimposed_img)
    cv.waitKey(0)

    superimposed_img = cv.cvtColor(superimposed_img, cv.COLOR_BGR2RGB)

    imgwithheat = Image.fromarray(superimposed_img)
    imgwithheat = imgwithheat.resize((256, 256), Image.BILINEAR)

    # redheatmap = cv.cvtColor(heatmap, cv.COLOR_BGR2RGB)

    return imgwithheat  # , imgwithboxes


# function to run gradcam++ algorithm
# @tf.function()
def gradCAMplusplus(model, image_path, layer_name):

    img = np.asarray(preprocess_data_for_grad_cam(image_path))
    print(img.shape)
    img_tensor = np.expand_dims(img, axis=0)
    conv_layer = model.get_layer(layer_name)  # get last conv_layer
    # modify model to let you check the last conv_layer output
    gradModel = tf.keras.Model([model.inputs], [
                               conv_layer.output, model.output])

    with tf.GradientTape(persistent=True) as t:
        conv_output, preds = gradModel(img_tensor)
        class_id = np.argmax(preds[0])
        output = preds[:, class_id]
        conv_first_grad = t.gradient(output, conv_output)
        conv_second_grad = t.gradient(conv_first_grad, conv_output)
        conv_third_grad = t.gradient(conv_second_grad, conv_output)

    # calculate weights for gradCAM++
    global_sum = np.sum(conv_output, axis=(0, 1, 2))
    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0] * 2.0 + global_sum * conv_third_grad[0]
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)

    alphas = alpha_num/alpha_denom
    alpha_normalization_constant = np.sum(alphas, axis=(0, 1))
    alphas /= alpha_normalization_constant

    relu_conv = np.maximum(conv_first_grad[0], 0.0)

    weights = np.sum(alphas*relu_conv, axis=(0, 1))
    forward_activation_maps = np.sum(weights*conv_output[0], axis=2)
    cam = np.maximum(forward_activation_maps, 0.0)  # passing through RELU
    max_cam = np.max(cam)
    if max_cam == 0:
        max_cam = 1e-10
    cam /= max_cam
    cam = (cam*-1.0) + 1.0
    return cam, output


def createBoxes(heatmap):
    new_heatmap = []
    for row in heatmap:
        new_row = []
        for pixel in row:
            new_pixel = abs(pixel - 1)
            new_row.append(new_pixel)
        new_heatmap.append(new_row)
    new_heatmap = np.asarray(new_heatmap)
    heatmap = (new_heatmap*255).astype("uint8")

    # maybe create some bounding boxes to show around decision making pixels

    _, thresh = cv.threshold(heatmap, 35, 255, cv.THRESH_BINARY_INV)

    cv.imshow("thresh", thresh)

    # Find the contour of the figure
    contours, hierarchy = cv.findContours(image=thresh,
                                          mode=cv.RETR_EXTERNAL,
                                          method=cv.CHAIN_APPROX_SIMPLE)
    new_heatmap = []
    for row in heatmap:
        new_row = []
        for pixel in row:
            new_pixel = abs(pixel - 1)
            new_row.append(new_pixel)
        new_heatmap.append(new_row)
    new_heatmap = np.asarray(new_heatmap)
    heatmap = (new_heatmap*255).astype("uint8")
    cv.imshow("blank heatmap", heatmap)

    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(heatmap, (x, y), (x+w, y+h), (255, 36, 15), 2)

    cv.imshow("bounding box heatmap", heatmap)

    height, width = heatmap.shape
    mask = np.zeros((height, width), np.uint8)

    circle_img = cv.circle(mask, (256, 256), 256, (255, 255, 255), thickness=-1)
    masked_data = cv.bitwise_and(heatmap, heatmap, mask=circle_img)
    _, thresh = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)
    mask_contours, hierarcy = cv.findContours(
        thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv.boundingRect(mask_contours[0])
    crop = masked_data[y:y+h, x:x+w]

    cv.imshow("crop", crop)
    cv.waitKey(0)

    return crop


def gradcam(model, image_path, layer_name):
    img = np.asarray(preprocess_data_for_grad_cam(image_path))
    img_tensor = np.expand_dims(img, axis=0)
    conv_layer = model.get_layer(layer_name)  # get last conv_layer
    # modify model to let you check the last conv_layer output
    gradModel = tf.keras.Model([model.inputs], [
                               conv_layer.output, model.output])

    with tf.GradientTape as t:
        conv_output, preds = gradModel(img_tensor)
        class_id = np.argmax(preds[0])
        output = preds[:, class_id]
        gradient = t.gradient(output, conv_output)

    weights = weights = tf.reduce_mean(gradient, axis=(0, 1))

    cam = tf.math.reduce_sum(weights*conv_output[0], axis=2)
    max_cam = tf.reduce_max(cam)
    if max_cam == 0:
        max_cam = 1e-10
    cam /= max_cam
    cam = (cam*-1.0) + 1.0
    return cam
    return cam
