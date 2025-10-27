import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

x_income = tf.constant([46, 60, 34, 30, 110], dtype=tf.float32)
x_price = tf.constant([70, 67, 50, 65, 80], dtype=tf.float32)
x_stations = tf.constant([6, 10, 3, 1, 12], dtype=tf.float32)
x_train = tf.stack([x_income, x_price, x_stations], axis=1)

y_train = tf.constant([7, 9, 3, 1, 3], dtype=tf.float32)

model = keras.Sequential([
    layers.Dense(200, activation="relu", input_shape=(3,)),
    layers.Dense(100),
    layers.Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

model.fit(x_train, y_train, epochs=700, verbose=1)
print("Training Complete")

x_income_test = tf.constant([120, 70, 80, 120, 70], dtype=tf.float32)
x_price_test = tf.constant([70, 70, 70, 70, 70], dtype=tf.float32)
x_stations_test = tf.constant([15, 8, 8, 15, 8], dtype=tf.float32)
x_test = tf.stack([x_income_test, x_price_test, x_stations_test], axis=1)

predictions = model.predict(x_test)
print("Input data:", x_test)
tf.print("Demand in new regions:", predictions.flatten())