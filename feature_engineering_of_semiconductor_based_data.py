# feature_engineering_of_semiconductor_based_data.py
# Analyze and sort out the unwanted features 
# Obtain a good model in predicting the pass yield of the semiconductor in inhouse testing

# # Informations of the Dataset

# import essential libraries
import pandas as pd
import sklearn 
import matplotlib.pyplot as plt
import numpy as np

#Load data and check the info of data
data=pd.read_csv('uci-secom.csv') 
print(data.info())

# # Missing Values

#check amount of columns containing null values
print(np.count_nonzero(data.isnull().sum())) 

no_na=data.isnull().sum()
#check which columns contain highest null values
no_na.sort_values(ascending=False) 

# # Modified / Remove Categorical Data

# check datatype for the datasets
typedata={str(k): len(list(v)) for k, v in data.groupby(data.dtypes, axis=1)} 
print(typedata)

# check which columns belong to which datagroup
bygroup=data.columns.to_series().groupby(data.dtypes).groups
print(bygroup)

# drop the categorical data column
data.drop('Time',axis=1,inplace=True) 

# # Balance Datasets check

print(data['Pass/Fail'])
#replace value [-1,1] to [1,0] for [pass,fail]
data['Pass/Fail'].replace(to_replace=[-1,1],value=[1,0],inplace=True) 

#check the ratio of pass to fail 
print("{0:.3f}".format(np.count_nonzero(data['Pass/Fail'])/float(data.shape[0]))) 

# method which split dataset to training and testing dataset
from sklearn.model_selection import train_test_split 

X = data.drop('Pass/Fail', axis=1) 
#target dataset
y = data[['Pass/Fail']] 

#split X and y datasets to both train and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42,stratify=y) 

#convert train, testing datasets into pandas dataframe
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)
y_train = pd.DataFrame(y_train, columns=y.columns)
y_test = pd.DataFrame(y_test, columns=y.columns)

# Standard Scaler
from sklearn.preprocessing import StandardScaler

# logistics regression model
from sklearn.linear_model import LogisticRegression

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# metrics used for evaluation
from sklearn.metrics import f1_score,matthews_corrcoef

# visualizations
from yellowbrick.classifier import ClassPredictionError, ConfusionMatrix
from sklearn.metrics import plot_roc_curve

# KNN imputation
from sklearn.impute import KNNImputer

# Datetime
import datetime

# method used for handling imbalaced data
from imblearn.over_sampling import SMOTE

# impute missing values and save it as a temporary dataset
imputer = KNNImputer()
imputer.fit(X_train)
imputed_train = pd.DataFrame(imputer.transform(X_train),columns = X_train.columns)
imputed_test = pd.DataFrame(imputer.transform(X_test),columns = X_test.columns)

# scale the data
scaler = StandardScaler()
train_std = pd.DataFrame(scaler.fit_transform(imputed_train),columns = imputed_train.columns)
test_std = pd.DataFrame(scaler.transform(imputed_test),columns = imputed_test.columns)

# oversampling the dataset to balance the label '1' and label '0' in training datasets 
oversample = SMOTE(random_state = 2)
over_X, over_y = oversample.fit_resample(train_std, y_train)

# # Selection of Model 

# Train logisticRegression model with training datatsets
logit = LogisticRegression(random_state = 42, class_weight = 'balanced', solver = 'liblinear',max_iter = 1000)
logit.fit(over_X, over_y.values.ravel())

# check training time required
start_time = datetime.datetime.now()
elapsed = datetime.datetime.now() - start_time
time = int(elapsed.total_seconds()*1000)

# check the F1 score and MCCscore of the logisticRegression model
y_predict = logit.predict(test_std)
y_true = y_test.values.ravel()
f1score = f1_score(y_true, y_predict, average = 'micro')
mccscore = matthews_corrcoef(y_true, y_predict)

#visualizations of the result
cpe = ClassPredictionError(logit,classes=['fail','pass']) #by Class
cpe.score(test_std, y_true)
cpe.show()

cm = ConfusionMatrix(logit,classes=['fail','pass']) #by Confusion Matrix
cm.score(test_std, y_true)
cm.show()

print (f' Training time: {time}ms\n F1 Score: {f1score} \n MCC Score: {mccscore}' )

# create an empty list for F1scores, MCCscores and times 
f1scores = []
mccscores =[]
times =[]

# append the result after each time the datasets were modified for comparison
f1scores.append(f1score)
mccscores.append(mccscore)
times.append(time)

# Train RandomForestClassifier model with training datasets
forest = RandomForestClassifier(class_weight = 'balanced',random_state = 42)
forest.fit(over_X, over_y.values.ravel())

# check training time required
start_time = datetime.datetime.now()
elapsed = datetime.datetime.now() - start_time
time = int(elapsed.total_seconds()*1000)

# check the F1 score and MCCscore of the model
y_predict = forest.predict(test_std)
y_true = y_test.values.ravel()
f1score = f1_score(y_true, y_predict, average = 'micro')
mccscore = matthews_corrcoef(y_true, y_predict)

#visualizations of the result
cpe = ClassPredictionError(forest,classes=['fail','pass']) # by Class
cpe.score(test_std, y_true)
cpe.show()

cm = ConfusionMatrix(forest,classes=['fail','pass']) # by ConfusionMatrix
cm.score(test_std, y_true)
cm.show()

print (f' Training time: {time}ms\n F1 Score: {f1score} \n MCC Score: {mccscore}' )

# append result to their corresponding list
f1scores.append(f1score)
mccscores.append(mccscore)
times.append(time)

# pop the result obtained from RandomForest Model as LogisticRegression model is more suitable for this dataset 
f1scores.pop()
mccscores.pop()
times.pop()


# create a function to ease the training and testing process
def evaluate(train_df, test_df, train_target, test_target):
        
        # data scaling for better perfomance of logistic regression
        scaler = StandardScaler()
        scaler.fit(train_df)
        train_std = pd.DataFrame(scaler.transform(train_df), columns=train_df.columns)
        test_std = pd.DataFrame(scaler.transform(test_df), columns = test_df.columns)
        
        # oversampling 
        oversample = SMOTE(random_state = 2)
        over_X, over_y = oversample.fit_resample(train_std, train_target)
        
        # training the model
        logit = LogisticRegression(random_state = 42, class_weight='balanced', solver='liblinear',max_iter = 1000)
        logit.fit(over_X,over_y.values.ravel())
        
        start_time = datetime.datetime.now()
        elapsed = datetime.datetime.now() - start_time
        time = int(elapsed.total_seconds()*1000)
        
        y_predict = logit.predict(test_std)
        y_true = test_target.values.ravel()
        f1score = f1_score(y_true, y_predict, average = 'micro')
        mccscore = matthews_corrcoef(y_true, y_predict)

        #visualizations
        cpe = ClassPredictionError(logit, classes=['fail','pass'])
        cpe.score(test_std, y_true)
        cpe.show()

        cm = ConfusionMatrix(logit,classes=['fail','pass'])
        cm.score(test_std, y_true)
        cm.show()

        return time, f1score, mccscore


# # 1. Eliminate Missing Values

# check the columns which is having missing values greater than threshold
def percentna(dataframe, threshold):
    columns = dataframe.columns[(dataframe.isna().sum()/dataframe.shape[1])>threshold] 
    return columns.tolist()
na_columns = percentna(X_train, 0.5)

# drop the respective columns
X_train_nona = X_train.drop(na_columns,axis = 1)
X_test_nona = X_test.drop(na_columns,axis = 1)
n_features1 = X_train_nona.shape[1]
print(f' There are {n_features1} features left.')


# # 2. Impute Missing Values

# impute missing values for those columns which having missing values less than threshold
imputer = KNNImputer()
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_nona), columns = X_train_nona.columns)
X_test_imp = pd.DataFrame(imputer.transform(X_test_nona), columns = X_test_nona.columns)

# check the training time, F1score and MCCscore again after the missing values were all modified
time,f1score,mccscore = evaluate(train_df = X_train_imp, test_df = X_test_imp, train_target = y_train, test_target = y_test)
print (f' Training time: {time}ms\n F1 Score: {f1score} \n MCC Score: {mccscore}' )

f1scores.append(f1score)
mccscores.append(mccscore)
times.append(time)

# # 3. Eliminate Variation below threshold

# Normalize the training and testing datasets
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
normalizer.fit(X_train_imp)

X_train_nrm = pd.DataFrame(normalizer.transform(X_train_imp), columns = X_train_imp.columns)
X_test_nrm = pd.DataFrame(normalizer.transform(X_test_imp), columns = X_test_imp.columns)

# Set VarianceThreshold and Remove the features which having Variance Threshold = 0.0
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold()
selector.fit(X_train_nrm)

# check which columns having variance threshold > 0.0
mask = selector.get_support()  
selected_cols = X_train_nrm.columns[mask]
print(selected_cols)

# transform the datasets
X_train_var = pd.DataFrame(selector.transform(X_train_imp), columns = selected_cols)
X_test_var = pd.DataFrame(selector.transform(X_test_imp),columns = selected_cols)

#check quantity of remained features
print(f' Remaining Features: {X_train_var.shape[1]}')

# Evaluation of model after normalization
time,f1score,mccscore = evaluate(train_df = X_train_var, test_df = X_test_var, train_target = y_train, test_target = y_test)
print (f' Training time: {time}ms\n F1 Score: {f1score} \n MCC Score: {mccscore}' )

f1scores.append(f1score)
mccscores.append(mccscore)
times.append(time)

# # 4. Pairwise Correlation

# Create a function which return first feature that is correlated with anything other feature
# as example: if A is highly correlated with B, A is removed but B is retained 

def correlation(dataset, threshold):
    col_corr = set()  
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]  
                col_corr.add(colname)
    return col_corr

# drop the selected features
corr_features = correlation(X_train_var, 0.95)
X_train_corr = X_train_var.drop(corr_features, axis = 1)
X_test_corr = X_test_var.drop(corr_features, axis = 1)

print(f'After removing {len(corr_features)}, there is {X_train_corr.shape[1]} remaining features.')

# Evaluation of model after removing features with high pairwise correlation
time,f1score,mccscore = evaluate(train_df = X_train_corr, test_df = X_test_corr, train_target = y_train, test_target = y_test)
print (f' Training time: {time}ms\n F1 Score: {f1score} \n MCC Score: {mccscore}' )

f1scores.append(f1score)
mccscores.append(mccscore)
times.append(time)

# # 5. Correlation with Target

# create a temporary datasets which include target columns
temporary_train = X_train_corr.copy()
temporary_train['target'] = y_train

# check the correlation of features with target
temporary_train.corr()

# Notice that some of the data is not available, and it may possibly caused by the variance of the features is 0.
# Check the data under those columns

# retrieve the columns which having NA correlation to target as null_columns
null = pd.DataFrame(temporary_train.corr())
null_columns = null[null['target'].isna()]
null_columns = null_columns.index.tolist()

# define function for features correlation with target
def corrwith_target(dataframe, target, threshold):
    cor = dataframe.corr()
    # Check the correlation value of each features with output variable
    cor_target = abs(cor[target])
    # Select lowly correlated features as irrelevant_features
    irrelevant_features = cor_target[cor_target < threshold]
    return irrelevant_features.index.tolist()[:-1] # return all irrelevant features but excluded the target itself

wcorrwith_cols = corrwith_target(temporary_train, 'target', 0.05)
# append the null_columns in the list of irrelevant_features
for i in range(len(null_columns)):
    wcorrwith_cols.append(null_columns[i])

# remove all irrelevant_features
X_train_corw = X_train_corr.drop(wcorrwith_cols, axis = 1)
X_test_corw = X_test_corr.drop(wcorrwith_cols, axis = 1)
print(f' After removing {len(wcorrwith_cols)} features, there are {X_train_corw.shape[1]} features left.')

# Evaluation of model after removal of irrelevant features
time,f1score,mccscore = evaluate(train_df = X_train_corr, test_df = X_test_corr, train_target = y_train, test_target = y_test)
print (f' Training time: {time}ms\n F1 Score: {f1score} \n MCC Score: {mccscore}' )

f1scores.append(f1score)
mccscores.append(mccscore)
times.append(time)

# # 6. Recursive Feature Elimination (RFE)

# Recursive feature elimination with cross-validation:
# 
# Method to search for optimum features for model improvement
# Search a subset of features based on their importance and eliminate the features which are considered less significant
# It will stop when a certain number of features are left, or elimination of features no longer help the model.
# 

# Scale the remaining datasets
scaler = StandardScaler()
scaler.fit(X_train_corw)

X_train_std = pd.DataFrame(scaler.transform(X_train_corw), columns = X_train_corw.columns)
X_test_std = pd.DataFrame(scaler.transform(X_test_corw), columns = X_test_corw.columns)

# import methods for Recursive Features Elimination
from sklearn.metrics import make_scorer
from yellowbrick.model_selection import RFECV
from sklearn.model_selection import StratifiedKFold

mcc_scorer = make_scorer(matthews_corrcoef)
rfecv = RFECV(estimator=LogisticRegression(random_state = 42, class_weight='balanced', dual=False, solver='liblinear'),
              cv=StratifiedKFold(2),
              scoring =  mcc_scorer)
rfecv.fit(X_train_std, y_train.values.ravel())
rfecv.show() 

# check the selected optimum columns from Recursive Feature Elimination
mask = rfecv.get_support()
columns = X_train_corw.columns
selected_cols = columns[mask]

X_train_rfe = pd.DataFrame(rfecv.transform(X_train_corw), columns = selected_cols)
X_test_rfe = pd.DataFrame(rfecv.transform(X_test_corw), columns = selected_cols)


# check our evaluation model

time, f1score, mccscore = evaluate(train_df = X_train_rfe, test_df = X_test_rfe, train_target=y_train, test_target=y_test)

print(f' Training time: {time}ms\n F1 Score: {f1score}\n MCC Score: {mccscore}')

f1scores.append(f1score)
mccscores.append(mccscore)
times.append(time)

# # Overall Figure for Scores and Training Time 

# Visualization of Perfomance of model after each steps of features elimination
fig, (ax0, ax1) = plt.subplots(2,1)
ax0.plot(times, label = 'Training Time')
ax0.set(ylabel = 'Training Time (ms)')
ax1.plot(f1scores, label = 'F1 Score', c = 'green')
ax1.plot(mccscores, label = 'MCC Score', c = 'red')
ax1.set(ylabel = 'Score')
ax1.set(xlabel = 'Feature selection step')
ax1.legend()
ax0.legend()
fig.show()

# # Conclusion

# Successfully eliminated about 574 features features from 600 features and training time boost from 1338 ms to 8 ms

