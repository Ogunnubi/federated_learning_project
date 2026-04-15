import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to the central dataset
DATASET = "central_dataset"

# Load the central model
model = tf.keras.models.load_model("central_model.h5")

# Image generator (same settings used during training)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Validation subset
val_gen = datagen.flow_from_directory(
    DATASET,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Evaluate the central model
loss, accuracy = model.evaluate(val_gen)

print("Central Model Results:")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Validation Loss: {loss:.4f}")