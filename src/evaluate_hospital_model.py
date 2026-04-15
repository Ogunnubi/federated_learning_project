import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys

hospital_name = sys.argv[1]

DATASET = f"federated_hospitals/{hospital_name}"

# Load the trained model
model = tf.keras.models.load_model(f"{hospital_name}_model.h5")

# Image generator (same split as training)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

val_gen = datagen.flow_from_directory(
    DATASET,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Evaluate the model
loss, accuracy = model.evaluate(val_gen)

print(f"Results for {hospital_name}:")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Validation Loss: {loss:.4f}")