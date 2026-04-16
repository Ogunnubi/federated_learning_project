![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Federated Learning](https://img.shields.io/badge/Federated-Learning-green)


Federated Learning for Pneumonia Detection
A privacy‑preserving deep learning system for medical image classification

🚀 Overview
This project implements a Federated Learning framework for pneumonia detection using chest X‑ray images.
Instead of centralizing sensitive medical data, multiple hospitals train models locally, and only the model weights are shared with a central server. 
This preserves patient privacy while still enabling strong global performance.

The project includes:

Local hospital model training

Centralized baseline model

Federated Averaging (FedAvg)

Evaluation scripts

Dataset splitting utilities

Visualization of model performance

🧠 Key Features
Federated Learning (FedAvg) implementation

Local hospital models trained independently

Centralized model for comparison

Evaluation metrics: accuracy, loss, and comparison charts

Dataset preprocessing and splitting

Clean modular code structure

Results visualizations included

## 📂 Project Structure  

Code
federated_learning_project/
│
├── src/
│   ├── split_dataset.py
│   ├── create_central_dataset.py
│   ├── train_central_model.py
│   ├── train_hospital_model.py
│   ├── training_hospital_model.py
│   ├── federated_averaging.py
│   ├── evaluate_central_model.py
│   ├── evaluate_federated_model.py
│   ├── evaluate_hospital_model.py
│   └── check_dataset.py
│
├── results/
│   ├── central_model_accuracy.png
│   ├── central_model_loss.png
│   └── model_comparison.png
│
├── .gitignore
├── requirements.txt
└── README.md
🏥 Federated Learning Workflow
Each hospital trains a local CNN model on its own X‑ray dataset.

Only the model weights (not the data) are sent to the central server.

The server performs Federated Averaging (FedAvg) to combine the weights.

The updated global model is redistributed to hospitals.

The cycle repeats until convergence.

This approach ensures:

Patient data never leaves the hospital

Improved performance through collaboration

Compliance with privacy regulations

🛠️ Installation
1. Clone the repository
Code
git clone https://github.com/Ogunnubi/federated_learning_project.git
cd federated_learning_project
2. Install dependencies
Code
pip install -r requirements.txt
▶️ How to Run the Project
1. Prepare the dataset
Split the dataset into hospital partitions:

Code
python src/split_dataset.py
2. Train hospital models
Code
python src/train_hospital_model.py
3. Train the central model (baseline)
Code
python src/train_central_model.py
4. Perform Federated Averaging
Code
python src/federated_averaging.py
5. Evaluate models
Code
python src/evaluate_central_model.py
python src/evaluate_hospital_model.py
python src/evaluate_federated_model.py
📊 Results
The results/ folder contains:

Accuracy curves

Loss curves

Model comparison chart

These visualizations help compare:

Local hospital models

Centralized model

Federated global model

🔮 Future Improvements
Add differential privacy

Implement secure aggregation

Use more advanced architectures (ResNet, EfficientNet)

Add cross‑hospital validation

Deploy global model as an API


🙌 Acknowledgements
Chest X‑ray dataset from publicly available sources

Federated Learning concept from McMahan et al. (2017)

TensorFlow/Keras for model development
