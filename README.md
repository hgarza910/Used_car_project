# Used Car Sales Predictor
## Overview
This project aims to build a predictive model for estimating the price of used cars. The
## Motivation
The motivation for this project stems from personal experience of being in the market for a used car. Recognizing the importance of accurate car valuation for negotiation purposes, I made the decision to develop a predictive model that could help me in assessing the value of a used car.
## Data Collection and Cleaning
The dataset used in this project was obtained from Kaggle and originally sourced by scaping TrueCar.com for used car listings. Each row in the dataset represents a single used car listing that includes various attributes such as Year, Make, Model, Price, VIN, City, and State. The dataset comprises a collection of 1.2 mission used cars. The dataset obtained from Kaggle was preprocessed so did not need to be cleaned.

## Exploratory Data Analysis (EDA)
Exploratory data analysis was conducted to gain insights into the dataset. The analysis focused on key attributes including Price, production Year, and Mileage examining distributions, spreads, and outliers. Additionally, a correlation heat map was used to identify the most influential features for training the predictive model. Categorical variables such as State , Make and Model were encoded for integration into the modeling phase.

## Exploratory Data Analysis (EDA) with Tableau
Along with the EDA conduction in jupyter, Tableau was also utilized to explore the data further taking a look at average car prices in relation to the production years. An interactive map was additionally created to visualize the average car prices across the United States. Key performance indicators (KPIs) were developed to display the number of cars sold, average mileage, and average price for each state. These tools were then integrated into a dashboard, which linked one to the other to provide targeted statistics based on user interaction. This analysis provided insights into the average car prices, sales metrics and geographic variations in the used car market.

## Model Exploration and Building
Given the objective of predicting a continuous variable (Price) based on multiple independent variables (Year, Milage, State, Make, Model), a multiple regression model was considered the most appropriate approach.
Several regression models were tested:
* Linear Regression
* Lasso Regression
* Random Forest Regression
* Gradient Boost Regression (XGBoost)

Due to the large size of the dataset, the training times for each model was recorded to consider in assessment of the model performance.

## Model Testing and Evaluation
All models were fitted and tested to assess their accuracy in predicting the used car prices. The evaluation process involved observing the Mean Absolute Error (MAE) of the test sample and collecting the accuracy of each model. The Random Forest Regression model demonstrated the highest performance, achieving a training score of 98% and a testing score of 90%, though it had a considerable training time of ~ 2 hours.
Here is how the other models performed:

| Model                	   | MAE	  | Training Accuracy | Testing Accuracy | Time to Train |
|:-------------------------|:-------|:------------------|:-----------------|:--------------|
| XGBoost              	   | $2,930 |  	89%      	      |  	88%     	     |  ~ 4 minutes  |
| Linear Regression    	   | $2,891 |  	86%      	      |  	86%     	     |  ~ 25 minutes |
| Lasso Regression     	   | $3,337 |  	82%      	      |  	81%     	     |  ~ 19 minutes |
| Random Forest Regression | $2,310 |  	98%       	    |  	90%     	     |  ~ 2 hours	   |

## Putting the Model Into Production
To enable the utilization of the model in a production environment, the Random Forest model was serialized using the Pickle library. The Flask framework was employed to create a production ready interface for utilizing the model locally. It is important to note the the current implementation of the model is intended for personal use only.
* insert pics

## Conclusion, Thoughts and Considerations
This project successfully developed a predictive model for estimating the price of used cars with a 90% accuracy. The model demonstrated high accuracy, with the Random Forest Regression approach outperforming the other tested models. Future considerations could be to look at replacing the Random Forest model with the XGBoost model. The reasons being, the XGBoost model offers similar predictive accuracy with significantly less training time but most importantly a much smaller model size.

