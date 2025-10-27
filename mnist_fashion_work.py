import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
import random

model = tf.keras.models.load_model("fashion_mnist_model.h5")

(_,_), (x_test, y_test) = fashion_mnist.load_data()
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28*28)

class_names = ["t-shirt/top",
               "trouser",
               "pullover",
               "dress",
               "coat",
               "sandal",
               "shirt",
               "sneaker",
               "bag",
               "ankle boot"]

def predict_image(index):
    img = x_test[index]
    img_expanded = img.reshape(1, 28, 28)
    predictions = model.predict(img_expanded)
    predicted_class = predictions[0].argmax()
    plt.imshow(img, cmap="grey")
    plt.title(f"Predicted: {class_names[predicted_class]}/n"
              f"True class: {class_names[y_test[index]]}")
    plt.axis("off")
    plt.show()

random_index = random.randint(0, len(x_test) - 1)
predict_image(random_index)