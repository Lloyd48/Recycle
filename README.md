# ♻️ Recycling Classifier (CNN)

This project is a Deep Learning model that analyzes images of household waste (Glass, Plastic, Paper, Metal, Cardboard, Trash) and predicts the correct recycling category. 

## 🚀 About The Project
A custom 5-block Convolutional Neural Network (CNN) with Residual (Skip) connections was designed from scratch and trained using PyTorch. The training pipeline was migrated from Windows to a Linux (Ubuntu) environment, utilizing CUDA optimizations on an RTX 4050 to reduce training time from 5 hours to just 30 minutes.

## 📊 Model Results
*   **Test Accuracy:** 95.56%
*   **Test Loss:** 0.1396
*   **Image Size:** 224x224
*   **Architecture:** Custom Residual CNN (5 Blocks)
*   **Dataset Size:** ~13,200 images (6 Classes, ~2200 images per class)

### Techniques Used
*   **Data Augmentation:** RandomCrop, RandomHorizontalFlip, RandomRotation, ColorJitter
*   **Optimization:** Adam Optimizer, Learning Rate Scheduler (StepLR / ReduceLROnPlateau)
*   **Regularization:** Batch Normalization, Dropout

## 🛠️ Installation & Usage

### Prerequisites
Install the required libraries:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install seaborn scikit-learn tqdm
