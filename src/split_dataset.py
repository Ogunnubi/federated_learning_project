import os
import shutil
import random

# Path to your downloaded dataset
DATASET_PATH = r"C:\Users\user\Downloads\train_model\chest_xray\train"

# Output folders for hospitals
OUTPUT_BASE = "federated_hospitals"
hospitals = ["hospital_A", "hospital_B", "hospital_C"]

# Create hospital folders
for h in hospitals:
    os.makedirs(f"{OUTPUT_BASE}/{h}/NORMAL", exist_ok=True)
    os.makedirs(f"{OUTPUT_BASE}/{h}/PNEUMONIA", exist_ok=True)

# Load image paths
normal_images = [os.path.join(DATASET_PATH, "NORMAL", f) 
                 for f in os.listdir(os.path.join(DATASET_PATH, "NORMAL"))]

pneumonia_images = [os.path.join(DATASET_PATH, "PNEUMONIA", f) 
                    for f in os.listdir(os.path.join(DATASET_PATH, "PNEUMONIA"))]

# Pick 750 from each class
normal_selected = random.sample(normal_images, 750)
pneumonia_selected = random.sample(pneumonia_images, 750)

# Combine and shuffle
all_images = [(img, "NORMAL") for img in normal_selected] + \
             [(img, "PNEUMONIA") for img in pneumonia_selected]

random.shuffle(all_images)

# Split into 3 hospitals (500 each)
for i, (img_path, label) in enumerate(all_images):
    hospital_index = i // 500  # 0, 1, or 2
    if hospital_index > 2:
        break  # Only need 1500 images total

    hospital_name = hospitals[hospital_index]
    dest_folder = f"{OUTPUT_BASE}/{hospital_name}/{label}"

    shutil.copy(img_path, dest_folder)

print("Dataset successfully split into 3 hospitals!")