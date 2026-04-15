import os
import shutil

# Paths to the hospitals
BASE = "federated_hospitals"
hospitals = ["hospital_A", "hospital_B", "hospital_C"]

# Output folder for centralised dataset
OUTPUT = "central_dataset"

os.makedirs(f"{OUTPUT}/NORMAL", exist_ok=True)
os.makedirs(f"{OUTPUT}/PNEUMONIA", exist_ok=True)

# Copy all images from all hospitals into central_dataset
for h in hospitals:
    for label in ["NORMAL", "PNEUMONIA"]:
        src_folder = f"{BASE}/{h}/{label}"
        for img in os.listdir(src_folder):
            src = os.path.join(src_folder, img)
            dst = os.path.join(OUTPUT, label, img)
            shutil.copy(src, dst)

print("Central dataset created successfully!")