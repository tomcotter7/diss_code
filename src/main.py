from model.PreTrainedIncResNetV2 import IncResNetV2
from utils.explainbility_utils import gradCAMplusplus, guidedBackProp

WEIGHTS = "../weights/test_weights.cpkt"


def main():

    image_path = "../images/test.jpg"
    model = IncResNetV2(WEIGHTS)
    predictions = model.run_image(image_path)

    return predictions


main()
# also think about here some kind of UI to pass in images
