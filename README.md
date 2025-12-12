# 🐶🐱 Cats vs Dogs Image Classification

A machine learning project comparing classical ML models and a neural network to classify grayscale images of cats and dogs.  
This project explores how model complexity affects performance on visual data and highlights the differences between linear models and deep learning.

---

## 📸 Project Preview
### **Model Predictions (Sample Output)**
<img width="1356" height="373" alt="image" src="https://github.com/user-attachments/assets/91bca465-49b6-4123-8733-bd5706edd195" />

### **Accuracy Comparison Chart**
<img width="1220" height="767" alt="image" src="https://github.com/user-attachments/assets/fe4919f2-1ec7-4615-91d3-d6c858e24cb7" />

---

# 🧠 Project Overview
This project builds and evaluates **three models**:

1. **Logistic Regression** – baseline linear classifier  
2. **Perceptron** – single-layer neural classifier  
3. **Multi-Layer Neural Network (MLP)** – nonlinear deep learning model  

The objective is to classify **64×64 grayscale images** as either **Cat** or **Dog**, and understand:

- Why neural networks outperform simple models  
- How image preprocessing affects model performance  
- What changes when using grayscale vs. RGB images  
- How to visualize and interpret classification results  

---

# 📂 Dataset Description
- **Type:** Grayscale images of cats and dogs  
- **Image Size:** 64×64 pixels  
- **Classes:**  
  - `0` → Cat  
  - `1` → Dog  
- **Split:** 80% training, 20% testing  

### **Preprocessing Steps**
- Resizing images to 64×64  
- Normalizing pixel values (0–1 scale)  
- Flattening images for classical ML models  
- Reshaping images for neural network input layers  

---

# 🧰 Tech Stack
### **Programming Language**
- Python 3.x

### **Libraries Used**
- NumPy  
- Pandas  
- Scikit-learn  
- TensorFlow / Keras  
- Matplotlib  
- Seaborn  

### **Tools**
- Jupyter Notebook / VS Code  
- Git & GitHub  

---

# 🤖 Models Implemented
## **1. Logistic Regression**
- Linear model  
- Struggles to learn spatial relationships  
- Serves as a baseline

**Accuracy:** ~52.4%

---

## **2. Perceptron**
- Single-layer neural network  
- Limited ability to learn complex patterns  
- Slightly better than Logistic Regression

**Accuracy:** ~52.9%

---

## **3. Multi-Layer Neural Network (Deep Learning)**
- Architecture: Dense(256) → Dense(128) → Dense(1, sigmoid)  
- Learns hierarchical and nonlinear representations  
- Best-performing model

**Accuracy:** ~58.1% ⭐

---

# 📊 Results Summary
| Model | Accuracy |
|-------|----------|
| Logistic Regression | **52.4%** |
| Perceptron | **52.9%** |
| Neural Network | **58.1%** |

---

# 🎯 Why Did the Neural Network Perform Best?
- It learns **non-linear** patterns using activation functions (ReLU)  
- More layers allow extraction of **spatial and abstract features**  
- Better modeling of relationships between pixel groups  

Classical ML models cannot capture these deep patterns due to their linear nature.

---

# 🎨 What if We Used RGB Images?
Switching from grayscale to RGB would:

- Increase input size from **64×64 → 64×64×3**  
- Provide richer color information → potential accuracy improvement  
- Increase training time and computational cost  
- Require stronger regularization  

---

# 🧩 Visualizations Included
This repository includes:

- ✔ Accuracy comparison bar chart  
- ✔ Model predictions vs actual labels grid  
- ✔ Training logs for the neural network  

These visualizations make performance analysis more intuitive.

---

# 💡 Key Learnings
This project demonstrates:

### ✔ How image data is preprocessed for ML/DL  
Flattening, normalization, reshaping.

### ✔ Differences between ML and DL for images  
Linear vs non-linear pattern recognition.

### ✔ How to evaluate models using  
- Accuracy  
- Precision  
- Recall  
- F1-score  

### ✔ How to visualize model performance  
Charts and prediction samples.

### ✔ Why deep learning outperforms classical ML for images  

---

# 🚀 How to Run This Project

### **1. Clone the repository**
```
git clone https://github.com/yourusername/cats-vs-dogs-classifier.git
cd cats-vs-dogs-classifier
```
### **2. Install dependencies**
```
pip install -r requirements.txt
```
### **3. Open the notebook**
```
Cats-vs-Dogs-Classifier.ipynb
```

# Conclusion
This project illustrates how different machine learning models perform on image classification tasks and why deep learning is generally more effective for visual datasets.
It provides hands-on experience with preprocessing, modeling, evaluation, and visualization—making it a strong portfolio project for AI/ML or Data Science roles.
