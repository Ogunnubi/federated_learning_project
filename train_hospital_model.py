import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import sys

# Read hospital name from command line
hospital_name = sys.argv[1]

DATASET = f"federated_hospitals/{hospital_name}"

# Image generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

# Same CNN architecture as central model
model = models.Sequential([
    layers.Input(shape=(224,224,3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)

model.save(f"{hospital_name}_model.h5")
print(f"{hospital_name} model trained and saved!")