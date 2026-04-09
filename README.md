# employee-burnout-analysis

This project analyzes employee burnout using machine learning tecniques to identify key factors contributing to burnout and to predict burnout levels. The analysis is based on a dataset containing employee workload, job characteristics, and mental fatigue factors. 

Goal of project: 
- Identify key factors that drive burnout
- Compare multiple ML models
- Provide actionable insights

Models: 
- Linear Regression
- Random Forest Regression
- Gradient Boosting Regression

How to run: 
1. Download the dataset from kaggle: https://www.kaggle.com/datasets/blurredmachine/are-your-employees-burning-out
2. Extract the files
3. Place train.csv inside the data/ folder
4. Navigate to src/ foldeer
5. run "data_loading.py" - python data_loading.py
6. run "visualizations.py" - python visualizations.py
7. run models - python model_linear.py, etc...


Key findings: 
- Mental fatigue is the strongest predictor of burnout.
- Resource allocation (Workload) is the second most important factor.
- Gradient Boosting achieved highest performance. 
