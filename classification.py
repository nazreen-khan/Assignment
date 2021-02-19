# import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from scipy import stats
from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, plot_confusion_matrix, precision_score, recall_score, accuracy_score
import warnings
warnings.simplefilter('ignore')

# Read the training data and drop column 'Unnamed: 0'
data = pd.read_csv("Arya_DataScientist_Assignment/training_set.csv")
data.drop('Unnamed: 0', axis=1, inplace=True)

# Print shape of data
print(data.shape)

print("Percentage of data with class 0 is {0:.2f}".format(data[data['Y']==0].shape[0]*100/data.shape[0]))
print("Percentage of data with class 1 is {0:.2f}".format(data[data['Y']==1].shape[0]*100/data.shape[0]))

# Check correlation with target variable
target_corr = data.corr()['Y']
target_corr = target_corr[(target_corr>=0.5) | (target_corr<=-0.5)]
print("Columns with correlation with target variable: ",target_corr.index.tolist()[:-1])
print("\nCorrelation:")
print(target_corr)

log_transform = np.median(((data.drop(['Y'], axis=1).skew()- np.log1p(data.drop(['Y'], axis=1)).skew())*100/(data.drop('Y', axis=1).skew())).values)
sqrt_transform = np.median(((data.drop('Y', axis=1).skew()- np.sqrt(data.drop(['Y'], axis=1)).skew())*100/(data.drop('Y', axis=1).skew())).values)

print("Percentage reduction in skewness due to log transform is {0:.2f}".format(log_transform))
print("Percentage reduction in skewness due to sqrt transform is {0:.2f}".format(sqrt_transform))

# Hence we use transform data using sqrt transform
feature_col = data.drop('Y', axis=1).columns
data[feature_col] = np.sqrt(data[feature_col])

# Separating out the features
X = data.drop('Y', axis=1)

# Separating out the target
y = data['Y']


# ### 2. Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# ### 3. Standardize dataset
# Standardizing the features
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)


# #### As we had seen there is very less correlation amongst predictor variables. Hence PCA will not perform better on this dataset.
# Fit PCA
n_components = 10
pca = PCA(n_components=n_components)
pca.fit(X_train_std)
print("Percentage variance explained by {0} components is {1:.2f}%".format(n_components, 100*pca.explained_variance_ratio_.sum()))

principalComponents = pca.transform(X_train_std)
principalDf = pd.DataFrame(principalComponents)

principalComponents_test = pca.transform(X_test_std)
principalDf_test = pd.DataFrame(principalComponents_test)

# #### Use Random forest classifier
# evaluate a lda model on the dataset
# define model
model = RandomForestClassifier(random_state=100)

# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate model
scores = cross_validate(model, principalDf, y_train, scoring=['accuracy', 'f1'], cv=cv, n_jobs=-1)

# summarize result
print('Mean Accuracy: %.3f' % (mean(scores['test_accuracy'])))
print('F1 score: %.3f' % (mean(scores['test_f1'])))

model.fit(principalComponents, y_train)
y_pred = model.predict(principalComponents_test)

print("F1 score on validation set is {0:.2f}".format(f1_score(y_test, y_pred)))

# Confusion Matrix
class_names = y.unique()

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
disp = plot_confusion_matrix(model, principalComponents_test, y_test,
                             display_labels=class_names,
                             cmap=plt.cm.Blues,
                             normalize='true')
disp.ax_.set_title("Normalized confusion matrix")
plt.show()

print("Accuracy: {0:.2f}".format(accuracy_score(y_test, y_pred)))
print("F1 Score: {0:.2f}".format(f1_score(y_test, y_pred)))
print("Recall Score: {0:.2f}".format(recall_score(y_test, y_pred)))
print("Precision Score: {0:.2f}".format(precision_score(y_test, y_pred)))


# ### From above results, not only accuracy of model is high but precision and recall scores are also good which indicates that model performance is not biased towards any class and FP and FN are also very less.

# ## Predictions on test data

# In[171]:


# Read test data file
test_data = pd.read_csv("Arya_DataScientist_Assignment/test_set.csv")
test_data.drop('Unnamed: 0', axis=1, inplace=True)

# Transformation for removing skewness
test_data[feature_col] = np.sqrt(test_data[feature_col])

# Standardize the data
test_data_std = scaler.transform(test_data)

# PCA
test_data_principalComponents = pca.transform(test_data_std)
test_data_principalDf = pd.DataFrame(test_data_principalComponents)

# Prediction
test_data_pred = pd.DataFrame(model.predict(test_data_principalDf), columns=['pred'])

# Save results
test_data_pred.to_excel("test_data_pred.xlsx",index=False)

test_data_pred


# In[ ]:




