# Housing Prices

## Overview
This housing prices project stems from the Intro to Machine Learning course on Kaggle. In the course, Kaggle walks you through models and the libraries used to create models. Models utilize many features such as LotArea or YearBuilt to figure out the prediciton target, which is the SalePrice(cost of the house) in this case. At the very end of the course all the code I wrote in all of the jupyter notebooks was put together to allow me to create a RandomForestRegressor model that I trained and then utilized to predict housing prices from a data set of houses in Iowa. The final exercise/juptyer notebook consists of my code for the Kaggle Housing Prices Competition.

## Jupyter Notebooks Overview

1) **Explore your Data**

The juptyer notebook familiarized me on how to use the pandas library to read in data from a file path. The data read in
is then used to form a DataFrame. It was made clear to me that the pandas library also provides powerful tools
for data manipulation in the DataFrame.

2) **Your First Machine Learning Model**

In this jupyter notebook, I learned how to use the sklearn library to create models for DataFrames. The sklearn library allowed me to create the DecisionTreeRegressor model based on an input file. Now to train this model, I was taught how to gather features and a create a prediction target, to allow my model to train itself off of the features and prediction target(SalePrice). Then I was able to see if my model could accurately predict the sales price of a house based on its training.

3) **Model Validation**

In this juptyer notebook I utilized the train_test_split function from the sklearn library to seperate the data in the DataFrame into training data and validation data. I was able to utilize the training data to train my DecisionTreeRegressor model and then
test the model's ability to predict housing prices based on how well it tested with the validation data. I learned it was better to test the accuracy of a model on new data called validation data rather than data a model was trained on. The accuracy of the model was then calculated through incorporating the MAE(mean absolute error) which is the mean of all the errors(actual-prediction) in housing costs. The mean_absolute_error function allowed for me to do so.

4) **Underfitting and Overfitting**

In this jupyter notebook I learned about how to find a number of max_leaf_nodes that would have the smallest MAE(mean absolute error) when creating a model. Finding the maximum_leaf_nodes amount will prevent overfitting and underfitting.

5) **Random Forests**

In the jupyter notebook I learned about and utilized the RandomForestRegressor model from the sklearn library. The purpose of utlizing this model was for me to see how the RandomForestRegressor model utilizing multiple trees to formulate averaged out housing prices is more accurate than a DecisionTreeRegressor model at predicting housing costs.

6) **Machine Learning Competition**

In this jupyter notebook I started off by utlizing previous code from the Random Forests notebook. I then trained a new RandomForestRegression model on all of the features and SalePrice data in the train.csv file in order to have an even more accurate and well informed model. I then was able to predict the housing prices of houses in Iowa from the test.csv file. I am currently not completely sure of how well my model did at predicting the housing costs in test.csv as the SalePrice data was not made available to me. A csv file of my final predictions was submitted to the competition.

## Installation
1) **Create a Kaggle Account**

2) **Enroll in Intro to Machine Learning Course**

3) **Access Jupyter Notebooks:**
All jupyter notebooks can be accessed directly on Kaggle

4) **Run Jupyter Notebooks:**
All jupyter notebooks can be run directly on Kaggle

## Data
Data is stored within the csv files. All features such as 'LotArea' or 'YearBuilt'can be accessed in train.csv. The prediction target SalePrice is also in train.csv. Test.csv only contains the features for predicting the prediction target. There is a data description file included as well.

### Files
test.csv

train.csv

data_description.txt

## Conclusion
The project examines the step by step process of building a machine learning model to predict housing prices. Within the process of building a machine learning model, there was data examination, feature filtering, model training and result analysis.

## Acknowledgements
Thank you to Kaggle for providing a course on Machine Learning and a data set on Iowa housing for me to analyze