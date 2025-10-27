import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import tensorflow as tf
import random

DATASET_PATH = "Dataset/"

num_classes = len(os.listdir(DATASET_PATH))
class_mode = "binary" if num_classes == 2 else "categorical"

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File not found at path: {image_path}")
        return
    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
    except (OSError, IOError):
        print(f"Error: File is corrupted - {image_path}")
    model = tf.keras.models.load_model("image_classifier.h5")

    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Failed to read image - {image_path}")
        return

    img = cv2.resize(img, (128,128))
    img = img/255
    img = tf.expand_dims(img, axis=0)

    prediction = model.predict(img)

    class_names = os.listdir(DATASET_PATH)

    if class_mode == "binary":
        predicted_class = class_names[int(bool(prediction[0]>0.5))]
    else:
        predicted_class = class_names[tf.argmax(prediction,axis=-1).numpy()[0]]

    print(prediction)
    print(image_path)
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f"The model has detected: '{predicted_class}'")
    print(f"The model has detected: '{predicted_class}'")
    plt.axis("off")
    plt.show()

#subfolder = ["cat", "dog", "other"]

#i=1

#while i < 26:
    #img_class = random.choice(subfolder)

    #if img_class == "other":
    #    img_num = random.randint(1,4)
    #else:
    #    img_num = random.randint(1,10)

    #img_img = f"{img_class}{img_num}.jpg"

    #print(img_img)

    #predict_image(f"Dataset/{img_class}/{img_img}")
    #i = i + 1

img_num = 1

while img_num < 11:
    predict_image(f"Dataset/cat/cat{img_num}.jpg")
    img_num = img_num + 1
    if img_num == 10:
        img_num = 1
        while img_num < 11:
            predict_image(f"Dataset/dog/dog{img_num}.jpg")
            img_num = img_num + 1