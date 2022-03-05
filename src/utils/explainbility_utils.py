import tensorflow as tf
import numpy as np

# run gradCAM++ on an input image and trained model


def gradCAMplusplus(img, model, layer_name):
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
    return cam


def gradCAM(img, model, layer_name):
    # might be worth building a vanilla gradcam implementation to test differences
    pass


def guided_backprop(img, model):
    # although bad, might be worth including just for the visualization element of it
    pass
