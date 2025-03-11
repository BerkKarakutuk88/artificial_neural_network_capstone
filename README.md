# ANN-Based Classification for Dyslexia Detection

## Overview
This project uses an Artificial Neural Network (ANN) to classify data related to dyslexia detection. The model is trained on a dataset containing features that help distinguish between dyslexic and non-dyslexic individuals.

## Features
- Utilizes a Multi-Layer Perceptron (MLP) architecture.
- Data preprocessing includes feature selection and normalization.
- Implements evaluation metrics such as accuracy, precision, recall, and F1-score.

## Installation
To run this project, install the necessary dependencies:
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

## Dataset
The dataset `disleksinormalhepsi.csv` contains labeled data used for classification. The preprocessing step removes unnecessary columns (`name`, `session`, `Unnamed: 73`, `Unnamed: 74`) and extracts features (`X`) and labels (`y`).

## Usage
1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-folder>
```
2. Run the Jupyter Notebook:
```bash
jupyter notebook artificial_neural_network_capstone.ipynb
```

## Model Architecture
- Input Layer: Extracted features from the dataset.
- Hidden Layers: Fully connected layers using activation functions like ReLU.
- Output Layer: Uses softmax/sigmoid activation for classification.

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Example Results
The model achieves an accuracy of approximately **XX%** on the test dataset. Example predictions and confusion matrix results are included in the notebook.

## Future Improvements
- Implement hyperparameter tuning.
- Experiment with different optimizers and activation functions.
- Apply cross-validation for better generalization.


