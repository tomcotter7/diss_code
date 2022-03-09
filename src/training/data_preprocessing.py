import cv2 as cv
import os
import csv
import random


def crop_image_otsu(img_path):
    img = cv.imread(img_path)
    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresholded = cv.threshold(grayscale, 0, 255, cv.THRESH_OTSU)
    bbox = cv.boundingRect(thresholded)
    x, y, w, h = bbox
    foreground = img[y:y+h, x:x+w]
    return foreground


def get_green_channel(img):
    r, g, b = cv.split(img)
    return g


def resize(src_path, dst_path):
    # takes any image, crops it and resizes it to 512x512
    for filename in os.listdir(src_path):
        if not os.path.isfile(dst_path+filename):
            img = crop_image_otsu(src_path+filename)
            # uncomment this if you want to make your images green_channel only
            # slightly decreases model accuray but improves training time
            # img = get_green_channel(img)
            resized_down = cv.resize(img, (512, 512), interpolation=cv.INTER_NEAREST)
            cv.imwrite(dst_path + filename, resized_down)


# re-arranges directory structure to image_dataset_from_directory() to work
def sort_directory(dst_directory, label_file):
    # needs to take the labels, move each image into the 'class' subfolder.
    with open(label_file) as labels:
        labels = csv.reader(labels, delimiter=",")
        for row in labels:
            try:
                image_name = row[0]
                severity_level = row[1]
                img = cv.imread(dst_directory+image_name+".jpeg")
                dirt = dst_directory+severity_level+"/"+image_name+".jpeg"
                cv.imwrite(dirt, img)
            except Exception as e:
                print(e)


# reduces the number of 0 class images in my dataset to be 3500
# rather than the original 25,000.
# which was causing problems in training
def make_balanced_dataset():
    file_list = os.listdir('/content/drive/MyDrive/year3/diss/512-train/0/')
    for i in range(3500):
        filename = random.choice(file_list)
        img = cv.imread("/content/drive/MyDrive/year3/diss/512-train/0/" + filename)
        cv.imwrite("/content/drive/MyDrive/year3/diss/512-train/reduced-0/" + filename, img)


def run_pp(src_path, dst_path, src_labels):

    resize(src_path, dst_path)
    print("pre-processed all images")
    sort_directory(dst_path, src_labels)
    print("sorted into folders")
    make_balanced_dataset()
    print("made balanced")
