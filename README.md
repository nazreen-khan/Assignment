# Assignment

Main file to run: classification.ipynb or classification.py

To view EDA: classification.html

We have dataset with aim of building of classification model on the data. The output variable has two classes 0 and 1. We have training set with dimensions 3910 X 58 with no null values and all continuous input variables. Distribution of output classes is not equal with ~60% of data with class 0.

The input features are highly positively skewed. Hence we perform square root tranform to decrease skewness to some level. This tramsform could decrease skewness to 57%.

Then we check for correlation in dataset. Turns out degree of correlation amongst predictors is very less and out of 57 input variables only two predictors have correlation greater than 0.5 with target variable.

Since the number of predictors is very large, we use PCA as dimensionality reduction algorithm. To find out number of components to be used we fit PCA with number of componentssame as number of input features. Then we plot graph of number of components vs variance explained. After 10th component, the degree of variance explained is very less. Hence we fit PCA with 10 number of components.

We split the training data into 4:1 split for train and validation set and used Random Forest Classifierfor as classification model. We tested model using Kfold cross validation. We got accuracy of 0.93 and F1 score of 0.92.

Also, on validation set accuracy was 0.94 and F1 score was 0.92. 

Predictions on test set is in test_data_pred.xlsx file.
