# Fraud Detection with Machine Learning

This repository focuses on detecting fraudulent transactions using the **Credit Card Fraud Detection Dataset (2023)**. It showcases two implementations:  
1. **GPU-Accelerated Training (cuML)**: Optimized for large datasets with faster training times using GPU.  
2. **CPU-Based Training**: A simpler implementation for environments without GPU support.  

## Overview
Fraud detection in financial transactions is critical for security and risk management. This project applies machine learning models to classify transactions as fraudulent or legitimate. Models were evaluated based on various metrics like Accuracy, Precision, Recall, F1-Score, and ROC AUC.

## Key Features
- **Models Used**:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (Linear Kernel)
  - Support Vector Machine (RBF Kernel)
- **Performance Metrics**:
  - Accuracy, Precision, Recall, F1-Score, ROC AUC
  - Confusion Matrix
  - ROC Curve
  - Training Time Measurement
- **Dataset**:
  - Credit Card Fraud Detection Dataset (2023)
  - Highly balanced dataset with a large number of transactions.
- **Tools**:
  - GPU-Accelerated Training: Utilized **cuML** for faster model training.
  - CPU-Based Training: For environments with limited hardware resources.

## Training Performance Summary
| Model                | Accuracy | Precision | Recall  | F1-Score | ROC AUC  | Training Time (GPU) | Training Time (CPU) |
|----------------------|----------|-----------|---------|----------|----------|----------------------|----------------------|
| Logistic Regression  | 0.964854 | 0.977061  | 0.952060| 0.964399 | 0.993506 | 3.04 seconds         | 3.80 seconds         | 
| **Random Forest**      |**0.999543** | **0.999420**  | **0.999666**| **0.999543** | **0.999962** | **2.42 seconds**        | **711.89 seconds**       | 
| SVM (Linear Kernel)  | 0.965584 | 0.978527  | 0.952060| 0.965112 | 0.993143 | 58.29 seconds        | ---                  |
| SVM (RBF Kernel)     | 0.965584 | 0.978527  | 0.952060| 0.965112 | 0.999780 | 14.54 seconds        | ---                  |

Due to the large size of the dataset, only **Logistic Regression** and **Random Forest** were implemented in the CPU-based workflow to reduce training time.

## Implementation
### GPU-Based Training
Leverages **cuML** to accelerate model training. This is particularly useful for large datasets where training times can otherwise be prohibitively long.

### CPU-Based Training
A simpler implementation for environments without access to GPUs. Includes Logistic Regression and Random Forest models for basic benchmarking.

## Evaluation Techniques
- **Confusion Matrix**: Visual representation of true positives, false positives, true negatives, and false negatives.
- **ROC Curve**: Evaluates the trade-off between true positive and false positive rates.
- **Training Time Measurement**: Highlights the computational efficiency of each model.

## Results
The Random Forest model achieved the best performance with:

- Accuracy: **99.95%**
- ROC AUC: **0.999962 (GPU-based training)**

## Future Work
- Experiment with additional algorithms like Gradient Boosting or Neural Networks.
- Explore real-time fraud detection applications.

## Credits
- Dataset: [Credit Card Fraud Detection Dataset (2023)](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
- Tools: cuML for GPU acceleration, scikit-learn for CPU-based models.
