import tensorflow as tf
from numpy1 import weighted_input


def sigmoid (x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    return np.tanh(x)

# /// WINTER SEVERITY ///
# inputs_tensor = tf.constant([-10,80], dtype=tf.float32)
# weights_tensor = tf.constant([-0.4, 0.3], dtype=tf.float32)
#
# weighted_weather = tf.multiply(inputs_tensor, weights_tensor)
# result = tf.reduce_sum(weighted_weather)
# winter_severity_probability = tf.sigmoid(result )
# tf.print("Probability of winter severity:", winter_severity_probability)

# /// FARM YIELD INDEX ///
# inputs_tensor = tf.constant ([7, 15], dtype=tf.float32)
# weights_tensor = tf.constant([0.6, 0.4], dtype=tf.float32)
#
# weighted_inputs = tf.multiply(inputs_tensor, weights_tensor)
# yield_index = tf.reduce_sum(weighted_inputs)
#
# result = tf.nn.relu(yield_index)
# tf.print("Yield Index:", result)

# /// SNACK APP AI ///
# inputs_tensor = tf.constant ([6, 5, 7, 4, 8], dtype=tf.float32)
# weights_tensor = tf.constant ([0.3, 0.4, 0.2, 0.1, 0.1], dtype=tf.float32)
#
# weighted_inputs = tf.multiply(inputs_tensor, weights_tensor)
# sum_features = tf.reduce_sum(weighted_inputs) #reduce to a sum
#
# activated_features = tf.nn.relu(sum_features)
# snack_rating = tf.round(activated_features) #round to the nearest whole number
#
# tf.print("Precise Rating:", sum_features)
# tf.print("Attractiveness Rating:", snack_rating)

# /// MOVIE POPULARITY APP ///
inputs_tensor = tf.constant ([9, 8.7, 5, 8], dtype=tf.float32)
weights_tensor = tf.constant ([0.1, 0.3, 0.25, 0.35])

weighted_inputs = tf.multiply(inputs_tensor, weights_tensor)
sum_factors = tf.reduce_sum(weighted_inputs)

result = tf.nn.relu(sum_factors)
tf.print("Movie popularity:", result)
