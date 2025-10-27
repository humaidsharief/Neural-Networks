from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os

DATASET_PATH = "Dataset/"

num_classes = len(os.listdir(DATASET_PATH))

optimizer = Adam(learning_rate=1e-3)

class_mode = "binary" if num_classes == 2 else "categorical"
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128,128),
    batch_size=32,
    class_mode=class_mode,
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128,128),
    batch_size=32,
    class_mode=class_mode,
    subset="validation"
)

model = Sequential([
    Input(shape=(128, 128, 3)),

    Conv2D(32, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dropout(0.5),
    Dense(256, activation="relu"),
    Dropout(0.3),

    Dense(num_classes, activation="softmax" if class_mode == "categorical" else "sigmoid")
])

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

loss_function = "binary_crossentropy" if class_mode == "binary" \
                                      else "categorical_crossentropy"
model.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])

model.fit(train_data, validation_data=val_data, epochs=10)
test_loss, test_accuracy = model.evaluate(val_data)
print(f"Model accuracy on validation data: {test_accuracy:.2f}")

model.save("image_classifier.h5")