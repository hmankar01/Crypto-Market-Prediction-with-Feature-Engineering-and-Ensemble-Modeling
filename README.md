# Crypto-Market-Prediction-with-Feature-Engineering-and-Ensemble-Modeling
This repository contains a comprehensive pipeline for predicting cryptocurrency market movements. The project leverages feature engineering, multiple machine learning models (LightGBM and a Multi-Layer Perceptron), and an optimized ensembling strategy to maximize predictive accuracy, as measured by the Pearson correlation coefficient.
##🚀 Project Overview
The core objective of this project is to predict a target value associated with cryptocurrency market data. The approach is structured into a clear, sequential pipeline, with each stage handled by a dedicated Jupyter notebook:

Feature Engineering: Raw market data is transformed into a rich feature set to better capture underlying market dynamics.

Model Training: Multiple models, including a Gradient Boosting Machine (LightGBM) and a Neural Network (MLP), are trained on the engineered features.

Ensembling: The predictions from the individual models are combined using an optimized weighting scheme to produce a final, more robust prediction.
## 💾 Data
The dataset for this project is hosted on Google Drive. You will need to download it and place it in the appropriate directory to run the notebooks.

Download Link: [DRW Crypto Market Prediction Dataset](https://drive.google.com/drive/folders/1axYIArBqH54ZvPXsV6q35hTgFYUGSDdG?usp=drive_link)

Before running the notebooks, please ensure you have the train.parquet and test.parquet files available in a location that the notebooks can access. The notebooks are configured to look for the data in an /input/ directory, so a structure like this is recommended:

/input/drw-crypto-market-prediction/ <br>
├── train.parquet<br>
└── test.parquet

## 📂 Repository Structure
This repository is organized into a series of Jupyter notebooks, each responsible for a specific part of the machine learning pipeline.

### feature-engineering.ipynb
Purpose: To process the raw training data and create a wide array of informative features.

Process:
Loads the initial train.parquet dataset.
Generates features related to:
Order Book Imbalance: Captures the ratio of buy to sell-side liquidity.
Trade Imbalance: Reflects the pressure from executed market orders.
Spreads and Sizes: Measures the difference in quoted quantities and average trade sizes.
Log-Transformed Features: Reduces the skewness of high-value features like volume.
Statistical Features: Calculates metrics like mean, standard deviation, and skewness over a set of X features.
Saves the engineered features (X_engineered.parquet) and the corresponding target variable (y_engineered.parquet).

### mlp-crypto.ipynb
Purpose: To train a Multi-Layer Perceptron (MLP) model as one of the components for the final ensemble.

Process:
Performs its own feature engineering, which includes many of the features from feature-engineering.ipynb.
Splits the data into training and validation sets.
Trains a LightGBM model first to identify the most important features.
Selects the top 200 features based on the LightGBM model's feature importance.
Trains an MLP model on this reduced, high-importance feature set. The model uses a custom pearson_loss function to directly optimize for the competition's metric.
Conducts hyperparameter tuning using Optuna for the LightGBM model.
Experiments with both simple and weighted ensembling of the LightGBM and MLP models.

### 02-train-lgbm.ipynb
Purpose: To train the primary LightGBM model using a cross-validation strategy.

Process:
Loads the engineered features created by feature-engineering.ipynb.
Employs a 5-fold cross-validation scheme to train the LightGBM model, ensuring robustness and preventing overfitting.
For each fold, it trains a model and saves it to disk.

Generates and saves:
Out-of-fold (OOF) predictions (oof_lgbm.npy): These are the predictions for the training data, which are crucial for the ensembling step.
Test predictions (test_preds_lgbm.npy): The predictions for the final test set.
Top 250 Features (sorted_features.csv): A list of the most influential features.

### 05-ensemble-ipynb.ipynb
Purpose: To find the optimal weights for combining the predictions of the different models.

Process:
Loads the OOF predictions from the various models (LGBM, MLP, and a placeholder for a third model, FTT).
Uses the scipy.optimize.minimize function to find the best weights for each model's predictions.
The objective function for this optimization is the negative Pearson correlation between the weighted OOF predictions and the true target values.
The output shows the individual scores of each model and the improved score of the final ensemble, demonstrating the value of this technique. The results indicate that the FTT model had a negligible contribution and was assigned a weight of zero.

## 🛠️ How to Run the Pipeline
To reproduce the results, execute the notebooks in the following order:
feature-engineering.ipynb: This will generate the necessary feature set for the subsequent models.
02-train-lgbm.ipynb and mlp-crypto.ipynb: These can be run in parallel to train the individual models and generate their respective OOF and test predictions.
05-ensemble-ipynb.ipynb: Run this notebook last to combine the predictions from the previous step and determine the optimal ensemble.

## 📋 Requirements
To run these notebooks, you will need Python 3 and the libraries listed in the requirements.txt file. You can install them using pip:
```Bash
pip install -r requirements.txt
requirements.txt
pandas
numpy
scipy
lightgbm
tensorflow
scikit-learn
matplotlib
seaborn
optuna
```
