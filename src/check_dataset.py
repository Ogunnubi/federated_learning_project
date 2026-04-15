from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET = "central_dataset"

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATASET,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='training'
)