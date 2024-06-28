# CS4168

## Project Overview

This project involves exploring, analyzing, and building machine learning models on a dataset related to the steel industry. The primary goal is to gain insights from the data and build predictive models for classification, regression, and clustering tasks.

## Getting Started

### Prerequisites

To run the notebooks, you need to have the following packages installed:

- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

You can install the necessary packages using `pip`:

```sh
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```
### Dataset

The dataset steel_industry_data_excerpt.csv should be placed in the same directory as the notebooks. This dataset contains the data used for exploration and model building.

### Notebooks

### Data Exploration

The Data_Exploration.ipynb notebook includes:

Loading and cleaning the dataset.

### Exploratory data analysis (EDA) including summary statistics and visualizations.

Feature engineering and selection.

### Classification

The Classifier.ipynb notebook includes:

- Data preprocessing for classification tasks.
- Building and evaluating different classification models.
- Hyperparameter tuning and model selection.
- Saving the final model as final_model_class.sav.
  
### Regression

The Regression.ipynb notebook includes:

- Data preprocessing for regression tasks.
- Building and evaluating different regression models.
- Hyperparameter tuning and model selection.
- Saving the final model as final_model_reg.sav.

### Clustering

The Clustering.ipynb notebook includes:

- Data preprocessing for clustering tasks.
- Performing clustering analysis using various algorithms.
- Visualization of clustering results and insights.


### Using the Saved Models

The saved models can be loaded and used for making predictions as follows:

### Loading the Classification Model

```sh
import joblib

# Load the model
model_class = joblib.load('final_model_class.sav')

# Example usage
# predictions = model_class.predict(X_new)
```

### Loading the Regression Model

```sh
import joblib

# Load the model
model_reg = joblib.load('final_model_reg.sav')

# Example usage
# predictions = model_reg.predict(X_new)
```
