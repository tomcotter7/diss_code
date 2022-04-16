
from model.IncResNetV2 import IncResNetV2
from frontend import App


WEIGHTS = "/home/tcotts/Documents/uni-work/year3/dissertation/diss_code/weights/full_model_ckpt"

# Python application to build a previously trained model, and then produce a Tkinter
# window that allows the user to input an image and test for Diabetic Retinopathy
# on said image.


def main():

    model = IncResNetV2(WEIGHTS, False)
    app = App(model)
    app.run()


main()
