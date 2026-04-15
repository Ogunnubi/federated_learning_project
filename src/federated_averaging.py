import tensorflow as tf
import numpy as np

# Load the 3 hospital models
model_A = tf.keras.models.load_model("hospital_A_model.h5")
model_B = tf.keras.models.load_model("hospital_B_model.h5")
model_C = tf.keras.models.load_model("hospital_C_model.h5")

# Extract weights
weights_A = model_A.get_weights()
weights_B = model_B.get_weights()
weights_C = model_C.get_weights()

# Average the weights
new_weights = []
for wA, wB, wC in zip(weights_A, weights_B, weights_C):
    new_weights.append((wA + wB + wC) / 3.0)

# Create a new model with the same architecture
federated_model = tf.keras.models.clone_model(model_A)
federated_model.set_weights(new_weights)

# Save the federated model
federated_model.save("federated_model.h5")

print("Federated model created and saved!")