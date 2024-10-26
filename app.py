import gradio as gr
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import logging

#  logging
logging.basicConfig(level=logging.INFO)

# Load and preprocess the dataset
try:
    dataset = pd.read_csv('./dataset/credit_score_predict.csv')
    logging.info("Dataset loaded successfully.")
except FileNotFoundError as e:
    logging.error("Dataset file not found. Ensure the path is correct.")
    raise e

dataset_2 = dataset.drop(['Loan ID', 'Customer ID', 'Months since last delinquent'], axis='columns')
dataset_3 = dataset_2.dropna()
dataset_3.loc[:, 'refined_jobYears'] = dataset_3['Years in current job'].apply(
    lambda x: int(''.join(filter(str.isdigit, str(x)))) if pd.notnull(x) else 0
)

# Correct categorical variables
dataset_4 = dataset_3.replace(to_replace=['HaveMortgage'], value=['Home Mortgage'])
dataset_4.rename(columns={
    'Credit Score': 'Credit_Score',
    'Current Credit Balance': 'Current_Credit_Balance',
    'Maximum Open Credit': 'Maximum_Open_Credit',
    'Current Loan Amount': 'Current_Loan_Amount',
    'Annual Income': 'Annual_income',
    'Monthly Debt': 'Monthly_Debt',
    'Years of Credit History': 'Years_of_Credit_History',
    'Number of Open Accounts': 'Number_of_Open_Accounts',
    'Number of Credit Problems': 'Number_of_Credit_Problems',
    'Home Ownership': 'Home_Ownership',
    'Tax Liens': 'Tax_Liens'
}, inplace=True)


dataset_5 = dataset_4.replace(
    to_replace=['other', 'major_purchase', 'Home Improvements', 'vacation', 'wedding', 'Take a Trip', 'moving', 'small_business'],
    value=['Other'] * 7 + ['Business Loan']
)
dummies = pd.get_dummies(dataset_5.Home_Ownership)
dummies2 = pd.get_dummies(dataset_5.Term)
dataset_6 = pd.concat([dataset_5, dummies, dummies2], axis='columns')
dataset_7 = dataset_6.drop(['Years in current job', 'Home_Ownership', 'Purpose', 'Rent', 'Term', 'Short Term'], axis='columns')

# Prepare features and target variable
X = dataset_7.drop(['Maximum_Open_Credit', 'Number_of_Credit_Problems', 'Long Term', 'Credit_Score', 'Current_Loan_Amount', 'Bankruptcies', 'Tax_Liens'], axis='columns')
Y = dataset_7.Maximum_Open_Credit
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=15)

# Train multiple regression models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=15)
}

# Fit all models
for model_name, model in models.items():
    model.fit(X_train, Y_train)
    logging.info(f"{model_name} model trained successfully.")

# Prediction function
def predict_credit_limit(Ownership, Annual_Income, Monthly_Debt, Years_of_Credit_History, Number_of_Open_Accounts, Current_Credit_Balance, refined_jobYears, model_choice):
    x = np.zeros(len(X.columns))
    ownership_index = np.where(X.columns == Ownership)[0]

    # Assign values to feature array
    x[0] = Annual_Income
    x[1] = Monthly_Debt
    x[2] = Years_of_Credit_History
    x[3] = Number_of_Open_Accounts
    x[4] = Current_Credit_Balance
    x[5] = refined_jobYears

    if ownership_index.size > 0:
        x[ownership_index] = 1

    # Predict using the selected model
    try:
        predicted = models[model_choice].predict([x])[0]
        logging.info(f"Prediction successful with {model_choice} model.")
    except Exception as e:
        logging.error(f"Prediction failed with {model_choice} model: {str(e)}")
        return "Error in prediction. Check model and inputs."
    return max(0, predicted)

# Gradio interface function
def credit_calculate(Annual_Income, Monthly_Debt, Current_Credit_Balance, Number_of_Open_Accounts, Years_of_Credit_History, refined_jobYears, Home_ownership, model_choice):
    predicted = predict_credit_limit(
        Home_ownership, float(Annual_Income), float(Monthly_Debt), 
        float(Years_of_Credit_History), int(Number_of_Open_Accounts), 
        int(Current_Credit_Balance), int(refined_jobYears), model_choice
    )
    return f"Estimated Credit Limit: {round(predicted, 2)} $"


interface = gr.Interface(
    fn=credit_calculate,
    inputs=[
        gr.components.Textbox(label="Annual Income ($)", placeholder="Enter your annual income"),
        gr.components.Textbox(label="Monthly Debt ($)", placeholder="Enter your monthly debt"),
        gr.components.Textbox(label="Current Credit Balance ($)", placeholder="Enter your current credit balance"),
        gr.components.Slider(minimum=0, maximum=50, label="Number of Open Accounts"),
        gr.components.Slider(minimum=0, maximum=50, label="Years of Credit History"),
        gr.components.Slider(minimum=0, maximum=50, label="Years of Professional Experience"),
        gr.components.Radio(["Own_Home", "Home_Mortgage", "Rent"], label="Home Ownership"),
        gr.components.Dropdown(list(models.keys()), label="Select Prediction Model", value="Linear Regression")
    ],
    outputs="text",
    title="Maximum Credit Limit and Default Risk Prediction by Kidus",
    description="Provide your financial details to get an estimated credit limit.",
    css="""
        .gradio-container {background-color: #f9f9f9; border-radius: 10px; padding: 20px;}
        .gradio-title {font-size: 24px; color: #3e5b91; font-weight: bold;}
        .gradio-description {font-size: 14px; color: #333;}
        .gr-input {border: 1px solid #ccc; border-radius: 5px;}
        .gr-button {background-color: #3e5b91; color: white; border: none; border-radius: 5px;}
        .gr-button:hover {background-color: #2d3a59;}
    """
)


interface.launch()
