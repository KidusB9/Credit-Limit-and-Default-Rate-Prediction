# Credit Limit Decisions and Default Rate Prediction

Welcome to **Credit Limit Decisions and Default Rate Prediction**, a Machine Learning application designed to assess and predict maximum credit limits and default rates for potential applicants based on comprehensive financial parameters. This project aims to equip banks and lending institutions with a  tool for evaluating creditworthiness, enhancing the efficiency and accuracy of their lending processes.

### Project Overview

In an time where data-driven decisions are important, this application utilizes machine learning algorithms to provide an intelligent estimation of credit limits and default risks. By analyzing critical financial metrics such as income, debt levels, and credit history, the model generates insights that help banks make informed lending decisions. This tool not only streamlines the credit approval process but also aids in risk management by ensuring that lending aligns with an applicant's financial capabilities.

### Technology Stack


- **Backend:** Powered by Python 3, with machine learning models implemented to perform complex calculations and predictions.
- **Frontend:** Developed using the Gradio library in Python 3, offering an intuitive user interface for inputting financial data and displaying predictions.
- **Version Control:** Utilizes Git for efficient version management and collaboration throughout the project development.

### Algorithms Used

This project implements:

- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Random Forest Regressor**

These algorithms are selected for their ability to capture linear and nonlinear relationships in the data, enabling accurate predictions based on the financial parameters provided by users.

### Folder Structure

- **app.py:** This file contains the core functionality of the application and serves as the entry point for executing the credit limit prediction.
- **CreditLimit.ipynb:** This Jupyter notebook includes the raw code used for data processing, model training, and evaluation.
- **Dataset Folder:** Contains the dataset (`credit_score_predict.csv`) that is utilized for training the machine learning models and validating their performance.

### Inputs Required from the User

To generate a credit limit prediction, the user must provide the following information:

| Field                          | Description                                                |
|--------------------------------|------------------------------------------------------------|
| **1. Annual Income**          | User's total income earned annually                        |
| **2. Monthly Debt**           | Current monthly obligations in debt payments               |
| **3. Current Credit Balance**  | Outstanding balance on current credit accounts             |
| **4. Number of Open Accounts** | Total number of active credit accounts                     |
| **5. Years of Credit History** | Duration of the user’s credit accounts                     |
| **6. Years in Current Job**   | Duration at the current place of employment                |
| **7. Home Ownership Status**   | Indicates whether the user owns or rents their home       |
| **8. Number of Bankruptcies**  | Count of previous bankruptcy filings                       |
| **9. Tax Liens**              | Indicates any tax liens on the user's credit report       |
| **10. Additional Factors**     | Any other relevant financial details                       |

### Explanation

For a deeper understanding of how the application works, including a demonstration of its functionality, watch this video: [Credit Limit Prediction Demo](https://www.youtube.com/watch?v=k0A4wmo58GY). In this video, we walk through the code and showcase how the app processes user data to deliver accurate credit limit predictions, highlighting its potential benefits for banks and financial institutions.


### Correlation Heatmaps

To visualize the relationships between different financial parameters used in the model, we have created correlation heatmaps:

- [Correlation Heatmap](https://github.com/KidusB9/Credit-Limit-and-Default-Rate-Prediction/blob/master/Correlation%20Heatmap.png)
- [Enhanced Correlation Heatmap for Final Mapping](https://github.com/KidusB9/Credit-Limit-and-Default-Rate-Prediction/blob/master/Enhanced%20Correlation%20Heatmap%20for%20Final%20Mapping.png)

These visualizations aid in understanding how different factors correlate with credit limit decisions and default rates.

### Requirements

To set up the project and ensure all necessary libraries are installed, run the following command in your terminal:

```bash
pip install -r requirements.txt

Then  run it on http://127.0.0.1:7860/

```bash
python app.py
