# stapedotomy_prediction_code
Prediction of hearing recovery after stapedotomy

## Environment setup 

In project root folder:
- conda update --all
- conda env create --file env.yml
- conda activate st_pred
- conda update --all
  
This installs all the necessary libraries. Jupyter notebooks can be run within this conda environment. 

## File details

The file audiogram_data.xlsx contains the dataset, in which previously empty values were filled with interpolated values. 

Files can be run in alphabetical order. 
This executes the whole machine learning pipeline, including data preprocessing (0preprocessing.ipynb), scaling (1scaling.ipynb), split to train and test datasets (2traintestsplit.ipynb), feature selection (3featselection.ipynb), testing the models (4testing.ipynb). Feature importance computation (5importance.ipynb) and audiogram prediction visualization (6visualization.ipynb) are also included.

Files constants.py and funct.py contain constants and helper functions. 
