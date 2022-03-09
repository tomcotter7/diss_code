import tensorflow as tf
import numpy as np
import cv2 as cv
from PIL import Image

# run gradCAM++ on an input image and trained model


def preprocess_data_for_grad_cam(img_path):
    img = cv.imread(img_path)
    img = crop_image_otsu(img)
    resized_img = cv.resize(img, (512, 512), interpolation=cv.INTER_LINEAR)
    cv.imwrite(img_path, resized_img)
    return img


def crop_image_otsu(img):

    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresholded = cv.threshold(grayscale, 0, 255, cv.THRESH_OTSU)
    bbox = cv.boundingRect(thresholded)
    x, y, w, h = bbox
    foreground = img[y:y+h, x:x+w]
    return foreground


def overlap_heatmap(img_path, heatmap, alpha):
    img = cv.imread(img_path)
    heatmap = cv.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (heatmap*255).astype("uint8")
    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    superimposed_img = cv.cvtColor(superimposed_img, cv.COLOR_BGR2RGB)

    imgwithheat = Image.fromarray(superimposed_img)
    return imgwithheat


def gradCAMplusplus(image_path, model, layer_name):
    img = np.asarray(preprocess_data_for_grad_cam(image_path))
    img_tensor = np.expand_dims(img, axis=0)
    conv_layer = model.get_layer(layer_name)  # get last conv_layer
    # modify model to let you check the last conv_layer output
    gradModel = tf.keras.Model([model.inputs], [conv_layer.output, model.output])

    # get gradient of final classification score
    # with respect to output of final conv layer
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape3:
                conv_output, preds = gradModel(img_tensor)
                class_id = np.argmax(preds[0])
                output = preds[:, class_id]
                conv_first_grad = tape3.gradient(output, conv_output)
            conv_second_grad = tape2.gradient(conv_first_grad, conv_output)
        conv_third_grad = tape1.gradient(conv_second_grad, conv_output)

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
    cam_heatmap = np.array(cv.applyColorMap(np.uint8(255*cam), cv.COLORMAP_JET))

    return cam_heatmap


def gradCAM(image_path, model, layer_name):
    # might be worth building a vanilla gradcam implementation to test differences
    pass


def createBoxes(heatmap):
    # maybe create some bounding boxes to show around decision making pixels
    pass
