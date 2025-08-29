# CNN Model for Poultry disease MRI Dataset

This project implements a Convolutional Neural Network (CNN) for classifying excreta into various stages of Chicken Disease using the [Best Chicken disease Dataset](<kaggle url>).

## Dataset

The dataset comprises high-quality Chicken excreta labeled for Chicken Disease progression. It can be downloaded from [Kaggle](<kaggle url>).

### Problem Statement

The task is a **classification** problem to predict the stage of Chicken Disease.

### Classes

The dataset includes the following classes:

- **Cocci**
- **Healthy**
- **NCD**
- **Salmo**

## Model Architecture

The CNN model comprises two convolutional blocks followed by a fully connected classifier; influenced by TinyVGG16 architecture.

## Performance Metrics

The model was evaluated on both training and test datasets. Below are the key metrics:

### Training Metrics

- **Loss:** 0.0039
- **Accuracy:** 99.90%
- **Precision:** 0.9990
- **Recall:** 0.9990
- **F1-Score:** 0.9990

### Testing Metrics

- **Loss:** 0.1574
- **Accuracy:** 95.47%
- **Precision:** 0.9563
- **Recall:** 0.9547
- **F1-Score:** 0.9548

### Train-Test Loss Over Epochs:

![Train Test Loss as observed over 20 epochs](outputs/train-test-loss-over-epochs.png "Train Test Loss")

### Train Data Confusion Matrix:

![Train Data Confusion Matrix](outputs/train_data_cnf_mat.png "Train Data Confusion Matrix")

### Test Data Confusion Matrix:

![Test Data Confusion Matrix](outputs/test_data_cnf_mat.png "Test Data Confusion Matrix")


###Project Structure

Smart-Poultry-System/
├── 📂 Notebooks/                 
│   ├── ChickenDisease_CNN.ipynb  
│   ├── predict_using_cockDiseaseCNN.ipynb
│   └── helper_functions.py       
├── 📂 PREDICT HUMIINDEX/        
│   ├── temp_model.ipynb        
│   └── helper_functions.py
├── 📂 models/                    
├── 📂 dataset/                   
├── 📂 split_dataset/           
├── 📂 Thesis/                
├── 🐍 app.py                    
├── 🏗️ model_arch.py          
├── 📋 requirements.txt          
└── 📖 README.md                 


