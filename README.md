# 🩺 Pneumonia Detection Using CNN & Transfer Learning

A deep learning project to detect pneumonia from chest X-ray images using Convolutional Neural Networks (CNN) with Transfer Learning and Grad-CAM visualization.

## 📁 Dataset

The dataset used for this project is from the [Kaggle Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). It includes:

- **Train folder**: Images labeled as `NORMAL` or `PNEUMONIA`
- **Test folder**: Separate unseen samples for evaluation
- **Validation folder**: (Optional) for fine-tuning

💡 You can also explore your own preprocessed version with 2-class binary labels (PNEUMONIA = 1, NORMAL = 0) if needed.

---

## 🎯 Objective

Build a binary image classifier to distinguish between normal lungs and those affected by pneumonia using chest X-ray scans.

---

## 🧠 Model Architecture

- **Transfer Learning Base**: `EfficientNetB0` (you can also try ResNet50, VGG16, etc.)
- **Custom Head**:
  - `GlobalAveragePooling2D`
  - `Dense(128, activation='relu')`
  - `Dropout(0.5)`
  - `Dense(1, activation='sigmoid')`

---

## ⚙️ Steps Followed

1. **Data Preprocessing**
   - Image resizing (e.g., 224x224)
   - ImageDataGenerator for real-time augmentation:
     - Rotation
     - Zoom
     - Horizontal Flip
     - Rescale (1./255)

2. **Model Compilation**
   - Loss: `binary_crossentropy`
   - Optimizer: `Adam` or `RMSprop`
   - Metrics: `accuracy`, `precision`, `recall`

3. **Training**
   - Trained for ~10–30 epochs
   - Validation loss/accuracy tracked
   - Early stopping to prevent overfitting

4. **Evaluation**
   - Accuracy on test set
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1-score)
   - 💡 *[ROC-AUC was not used but can be added in future versions]*

---

## 📈 Results

- ✅ Achieved ~95% Accuracy  
- ✅ High Recall (93%) for detecting pneumonia  
- ⚠️ Class imbalance handled by using appropriate augmentation and batch strategies

---

## 🔍 Grad-CAM Visualization

To make the model more explainable:

- **Used Grad-CAM** to visualize **which part of the X-ray** the model focuses on when predicting pneumonia.
- Heatmaps overlaid on original X-rays help doctors trust the model's decisions.

---

## 📦 Output

- Trained model saved as `.h5`  
- Grad-CAM results exported for selected predictions  
- Clean and reusable pipeline for future medical imaging projects

---

## 🧰 Tech Stack

- Python, TensorFlow, Keras
- NumPy, Pandas, Matplotlib, Seaborn
- Google Colab (GPU)
- Grad-CAM Visualization

---

## 📧 Contact

**Akram Moiz**  
[LinkedIn](https://www.linkedin.com/in/akram-moiz) | [GitHub](https://github.com/akrammoiz35)

---

## 📜 License

This project is for educational purposes and not intended for real clinical diagnosis. Use with caution and medical guidance.
