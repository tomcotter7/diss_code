from model.IncResNetV2 import IncResNetV2

WEIGHTS = "../weights/test_weights.cpkt"


def main():

    image_path = "../images/test.jpg"
    model = IncResNetV2(WEIGHTS, False)
    cam = model.run_image(image_path)
    # visualize heatmap
    # show next to input image
    return cam


main()
# also think about here some kind of UI to pass in images
