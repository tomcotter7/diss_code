from model.IncResNetV2 import IncResNetV2
from frontend import App


WEIGHTS = "/home/tcotts/Documents/uni-work/year3/dissertation/diss_code/weights/dbest_save_at_34.ckpt.index"


def main():

    model = IncResNetV2(WEIGHTS, True)
    app = App(model)
    app.run()

    # visualize heatmap
    # show next to input image


main()
# also think about here some kind of UI to pass in images
