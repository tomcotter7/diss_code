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
    return img


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
    img = cv.imread(img_path)
    heatmap = cv.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (heatmap*255).astype("uint8")
    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_HOT)
    boxes = createBoxes(heatmap)
    boxes_img = boxes * 0.25 + img
    superimposed_img_boxes = np.clip(boxes_img, 0, 255).astype("uint8")
    superimposed_img_boxes = cv.cvtColor(superimposed_img_boxes, cv.COLOR_BGR2RGB)
    imgwithboxes = Image.fromarray(superimposed_img_boxes)
    imgwithboxes = imgwithboxes.resize((256, 256), Image.BILINEAR)

    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    superimposed_img = cv.cvtColor(superimposed_img, cv.COLOR_BGR2RGB)
    imgwithheat = Image.fromarray(superimposed_img)
    imgwithheat = imgwithheat.resize((256, 256), Image.BILINEAR)

    return imgwithheat, imgwithboxes


@tf.function
def getGradientsDerivatives(model, img_tensor):
    with tf.GradientTape() as t1:
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t3:
                conv_output, preds = model(img_tensor)
                class_id = tf.math.argmax(preds[0])
                output = preds[:, class_id]
                conv_first_grad = t3.gradient(output, conv_output)
            conv_second_grad = t2.gradient(conv_first_grad, conv_output)
        conv_third_grad = t1.gradient(conv_second_grad, conv_output)

    return conv_output, preds, class_id, conv_first_grad, conv_second_grad, conv_third_grad


# function to run gradcam++ algorithm
def gradCAMplusplus(image_path, model, layer_name):
    img = np.asarray(preprocess_data_for_grad_cam(image_path))
    img_tensor = np.expand_dims(img, axis=0)
    conv_layer = model.get_layer(layer_name)  # get last conv_layer
    # modify model to let you check the last conv_layer output
    gradModel = tf.keras.Model([model.inputs], [conv_layer.output, model.output])

    # get gradient of final classification score
    # with respect to output of final conv layer
    """
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape3:
                conv_output, preds = gradModel(img_tensor)
                class_id = np.argmax(preds[0])
                print(class_id)
                # if class_id == 0:
                # return None, class_id
                output = preds[:, class_id]
                conv_first_grad = tape3.gradient(output, conv_output)
            conv_second_grad = tape2.gradient(conv_first_grad, conv_output)
        conv_third_grad = tape1.gradient(conv_second_grad, conv_output)
    """

    # measure time to see if this actually improved time or not - also we could probably show the time taken?
    conv_output, preds, class_id, conv_first_grad, conv_second_grad, conv_third_grad = getGradientsDerivatives(gradModel, img_tensor)

    # calculate weights for gradCAM++
    global_sum = np.sum(conv_output, axis=(0, 1, 2))
    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0] * 2.0 + global_sum * conv_third_grad[0]
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)

    alphas = alpha_num/alpha_denom

    relu_conv = np.maximum(conv_first_grad[0], 0.0)

    weights = np.sum(alphas*relu_conv, axis=(0, 1))
    forward_activation_maps = np.sum(weights*conv_output[0], axis=2)
    cam = np.maximum(forward_activation_maps, 0.0)  # passing through RELU
    max_cam = np.max(cam)
    if max_cam == 0:
        max_cam = 1e-10
    cam /= max_cam
    cam = (cam*-1.0) + 1.0

    return cam, class_id


def gradCAM(image_path, model, layer_name):
    # might be worth building a vanilla gradcam implementation to test differences
    pass


def createBoxes(heatmap):
    # maybe create some bounding boxes to show around decision making pixels
    grayheatmap = cv.cvtColor(heatmap, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(grayheatmap, 40, 255, cv.THRESH_BINARY_INV)

    # Find the contour of the figure
    contours, hierarchy = cv.findContours(image=thresh,
                                          mode=cv.RETR_EXTERNAL,
                                          method=cv.CHAIN_APPROX_SIMPLE)

    height, width, _ = heatmap.shape
    boxes = np.full_like(heatmap, 255)
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(boxes, (x, y), (x+w, y+h), (255, 36, 15), 2)

    return boxes
