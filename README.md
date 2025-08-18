# IDS-Script-using-ML
#Intrusion Detection System (IDS) with Machine Learning Script
___________________

## Project Overview 
This project implements a "Network Intrusion Detection System (IDS)" using machine learning. The IDS is trained on processed network traffic data from the CICIDS2017 database to classify a given record as "normal traffic" or a "cyber attack". 

Timeline: 
- Day 1 : Data collection and preprocessing -  network traffic CSV
- Day 2 Exploratory Data Analysis (EDA) - preparing data for training & deciding best model 
- Day 3 Random Forest Model training & evaluation 
- Day 4 CLI prediction interface
- Day 5 Documentation 
____________________

## Features
- Prediction interface (CLI) where users can input new traffic data and recieve predictions
- Evaluation based on accuracy, precision, recall, F1-score, and confusion matrix
- Model training using Random Forest Classifier
- Data preprocessing with Pandas & NumPy
- Cross-validation 
- Good UX & output messages 
_____________________

## Techicalities 
- Programming Language: Python 3.11
- Libraries:  
  - Pandas, NumPy (data processing) 
  - scitkit-learn (ML model & evaluation)
  - Matplotlib, Seaborn (visualization)
  - Joblib (model loading & saving)
_____________________
## How to run
1. Clone the repository or download the project folder
   bash git clone https://github.com/nk-spec-code/ML_IDS_PROJECT_NETRA.git 
   cd ML_IDS_PROJECT_NETRA   
2. Setup virtual environment
   python -m venv venv
   venv\Scripts\activate      # On Windows
   source venv/bin/activate   # On Mac/Linux
   pip install -r requirements.txt
3. Run scripts accordingly
   python explore_data.py
   python model_eval.py
   python predict_interface.py


