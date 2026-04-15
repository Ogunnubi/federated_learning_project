import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys

# Read hospital name from command line
hospital_name = sys.argv[1]

DATASET = f"federated_hospitals/{hospital_name}"

# Load the trained model
model = tf.keras.models.load_model(f"{hospital_name}_model.h5")

# Image generator (same split as training)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training subset
train_gen = datagen.flow_from_directory(
    DATASET,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='training',
    shuffle=False
)

# Evaluate on training data
loss, accuracy = model.evaluate(train_gen)

print(f"Training Accuracy for {hospital_name}: {accuracy * 100:.2f}%")
print(f"Training Loss: {loss:.4f}")