from model.IncResNetV2 import IncResNetV2
from utils.explainbility_utils import gradCAMplusplus, guidedBackProp

WEIGHTS = "../weights/test_weights.cpkt"


def main():
    # get img file path
    # create an instance of model
    image_path = "../images/test.jpg"
    model = IncResNetV2(WEIGHTS)
    predictions = model.run_image(image_path)
    return predictions
    # run model with img
    # should return a gradcam++ instance
    # visualize gradcam++ instance with heatmap and guided backprop


main()
# also think about here some kind of UI to pass in images
