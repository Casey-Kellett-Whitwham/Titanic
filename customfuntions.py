
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
import kagglehub
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
import multiprocessing

def missing(full_data):

    missing = full_data.isnull().sum()
    missing_df = pd.DataFrame(missing, columns=['Missing Values'])
    return missing_df


def load_titanic_data():
    testset=pd.read_csv('data/test.csv')
    trainset=pd.read_csv('data/train.csv')
    testsurvied = pd.read_csv('data/gender_submission.csv')

    print("Train Data Shape : ",trainset.shape)
    print("Test Data Shape : ",testset.shape)
    print("Test y Shape : ",testsurvied.shape)

    testsurvied_drop = testsurvied.drop('PassengerId', axis=1)
    test_sets = [testset, testsurvied_drop]
    test_full = pd.concat(test_sets, axis=1)

    print("Test shape after concat: ",test_full.shape)
    test_full.head()

    y_train = trainset['Survived']
    X_train = trainset.drop('Survived', axis=1)
    y_test = test_full['Survived']
    X_test = test_full.drop('Survived', axis=1)
    allsets = [trainset,test_full]

    full_data=pd.concat(allsets,axis=0)
    print("Full data Shape :", full_data.shape)


    return X_train, X_test, y_train, y_test,full_data


def replace_columns(df):
    df['Parch'] = np.log(df['Parch'] + 1)
    df['SibSp'] = np.log(df['SibSp'] +1)
    df['Age'] = np.log(df['Age'] +1)
    df['Fare'] = np.log(df['Fare'] +1)
    return df


def pipeline_prep(Xtrain, Xtest, ytrain, ytest, 
                      poly_degree = 2):
    
    Xtrain = replace_columns(Xtrain)
    Xtest = replace_columns(Xtest)
    
    num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('polynomial_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()) ])
    
    cat_pipeline = make_pipeline(SimpleImputer(strategy = "most_frequent"),\
                                 OneHotEncoder(handle_unknown='ignore'))

    preprocessing = ColumnTransformer([("num", num_pipeline, make_column_selector(dtype_include=np.number)),\
                                       ("cat", cat_pipeline, make_column_selector(dtype_include=object))])
    
    
    Xtrain_prepared = preprocessing.fit_transform(Xtrain)

    Xtest_prepared = preprocessing.transform(Xtest)
    
    Xtrain_prepared, ytrain_prepared = handle_outlier(Xtrain_prepared, ytrain)

    ytrain_scaler = MinMaxScaler()
    ytrain_prepared = ytrain_scaler.fit_transform(ytrain_prepared)
     
    ytest_scaler = MinMaxScaler()
    ytest_prepared = ytest_scaler.fit_transform(ytest)
     
    return Xtrain_prepared, Xtest_prepared, ytrain_prepared, ytest_prepared

def get_outlier_indices(X):
    model = LocalOutlierFactor()
    return model.fit_predict(X) 

def handle_outlier(X, y):
    outlier_ind = get_outlier_indices(X)
    return X[outlier_ind == 1], y[outlier_ind == 1]