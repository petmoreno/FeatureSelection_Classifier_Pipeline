#Created on Fri Apr 9th 2021

#%%

#Import general libraries
from numpy.core.numeric import NaN
from numpy.lib.function_base import average
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import libraries useful for building the pipeline and join their branches
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


#import modules created for data preparation phase
import my_utils
import missing_val_imput
import feature_select
import preprocessing
import adhoc_transf

#import libraries for data preparation phase
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder


#import libraries from modelling phase
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef

#import classifiers
#import Ensemble Trees Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier,VotingClassifier
import xgboost as xgb

#to save model fit with GridSearchCV and avoid longer waits
import joblib

#%%

#Loading the dataset
col_names=['OVERALL_DIAGNOSIS','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20','F21','F22']
path_data_train=r'datasets\SPECTF Heart\SPECT.train'
path_data_test=r'datasets\SPECTF Heart\SPECT.test'

df_train=pd.read_csv(path_data_train, names=col_names, header=None)
df_train.head()

#%%
df_test=pd.read_csv(path_data_test, names=col_names, header=None)
df_test.head()

#%%Characterizing the data set
target_feature='OVERALL_DIAGNOSIS'
numerical_feats=[]
nominal_feats=['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20','F21','F22']
ordinal_feats=[]

len_numerical_feats=len(numerical_feats)
len_nominal_feats=len(nominal_feats)
len_ordinal_feats=len(ordinal_feats)

#%%
######################################
#Step 0: Perform EDA to detect missing values, imbalanced data, strange characters,etc.
#############################

##Statistical analysis
df_train.describe()
#%%
#Identifying missing values
my_utils.info_adhoc(df_train)
#%%
#Exploring wrong characters
my_utils.df_values(df_train)

##Statistical analysis
df_test.describe()
#%%
#Identifying missing values
my_utils.info_adhoc(df_test)
#%%
#Exploring wrong characters
my_utils.df_values(df_test)


#%%
#############################
#Step 1 Solving wrong characters of dataset
#############################
#Set column id as index


# CKD case does only have misspellingCorrector
# df_content_solver=Pipeline([('fx1', misspellingCorrector()),
#                             ('fx2',function2()),
#                             ('fx3',function3())
# ])


#%%Performing numeric cast for numerical features
df_train.loc[:,numerical_feats]=adhoc_transf.Numeric_Cast_Column().fit_transform(df_train.loc[:,numerical_feats])
df_train[numerical_feats].dtypes
df_test.loc[:,numerical_feats]=adhoc_transf.Numeric_Cast_Column().fit_transform(df_test.loc[:,numerical_feats])
df_test[numerical_feats].dtypes

#%%Performing category cast for nominal features
df_train.loc[:,nominal_feats]=adhoc_transf.Category_Cast_Column().fit_transform(df_train.loc[:,nominal_feats])
df_train[nominal_feats].dtypes
df_test.loc[:,nominal_feats]=adhoc_transf.Category_Cast_Column().fit_transform(df_test.loc[:,nominal_feats])
df_test[nominal_feats].dtypes

#%%Performing category cast for ordinal features
df_train.loc[:,ordinal_feats]=adhoc_transf.Category_Cast_Column().fit_transform(df_train.loc[:,ordinal_feats])
df_train[ordinal_feats].dtypes
df_test.loc[:,ordinal_feats]=adhoc_transf.Category_Cast_Column().fit_transform(df_test.loc[:,ordinal_feats])
df_test[ordinal_feats].dtypes


#%%
#############################
##Step 2 Train-Test splitting
#############################

#Split the dataset into train and test
#test_ratio_split=0.3
#train_set,test_set=train_test_split(df, test_size=test_ratio_split, random_state=42, stratify=df[target_feature])

X_train=df_train.drop(target_feature,axis=1)
y_train=df_train[target_feature].copy()

X_test=df_test.drop(target_feature,axis=1)
y_test=df_test[target_feature].copy()

#%%
########################################
##Step 3 Label Encoding of target value
########################################
le=LabelEncoder()
y_train=le.fit_transform(y_train)
y_test=le.fit_transform(y_test)
le.classes_
#%%
##############################
##Step 2 Building pipelines for data preparation
##############################

#Lets define 3 pipeline mode
#a) parallel approach where feature selection is performed in parallel 
# for numerical, nominal and categorical
#b) general approach where feature selection is performed as a whole for other features
#c) no feature selection is performed

#Before a data preprocessing will take place for each type of feature
pipeline_numeric_feat=Pipeline([ ('data_missing',missing_val_imput.Numeric_Imputer(strategy='median')),
                                 ('scaler', MinMaxScaler())])

pipeline_nominal_feat=Pipeline([('data_missing',missing_val_imput.Category_Imputer(strategy='most_frequent')),                                 
                                 ('encoding', OrdinalEncoder())])#We dont use OneHotEncoder since it enlarges the number of nominal features 

pipeline_ordinal_feat=Pipeline([ ('data_missing',missing_val_imput.Category_Imputer(strategy='most_frequent')),
                                 ('encoding', OrdinalEncoder())])


#option a)
pipe_numeric_featsel=Pipeline([('data_prep',pipeline_numeric_feat),
                                ('feat_sel',feature_select.Feature_Selector(strategy='filter_num') )])
pipe_nominal_featsel=Pipeline([('data_prep',pipeline_nominal_feat),
                                ('feat_sel',feature_select.Feature_Selector(strategy='filter_cat') )])
pipe_ordinal_featsel=Pipeline([('data_prep',pipeline_ordinal_feat),
                                ('feat_sel',feature_select.Feature_Selector(strategy='filter_cat') )])

dataprep_pipe_opta=ColumnTransformer([('numeric_pipe',pipe_numeric_featsel,numerical_feats),
                                    ('nominal_pipe',pipe_nominal_featsel,nominal_feats),
                                    ('ordinal_pipe',pipe_ordinal_featsel,ordinal_feats)
                                ])

#option b)
dataprep_merge_feat=ColumnTransformer([('numeric_pipe',pipeline_numeric_feat,numerical_feats),
                                    ('nominal_pipe',pipeline_nominal_feat, nominal_feats),
                                    ('ordinal_pipe',pipeline_ordinal_feat,ordinal_feats)
                                ])
dataprep_pipe_optb=Pipeline([('data_prep',dataprep_merge_feat),
                                ('feat_sel',feature_select.Feature_Selector(strategy='wrapper_RFECV') )])

#option c)
#dataprep_merge_feat is used
dataprep_merge_feat
#%%
#############################
##Step 3 Classifier initialization
#############################
#Several ensemble classifier with Cross validation will be applied
#we take decision tree as base classifier

#Init the clasfifier
dectree_clf=DecisionTreeClassifier(random_state=42)
rndforest_clf=RandomForestClassifier(random_state=42)
extratree_clf=ExtraTreesClassifier(random_state=42)
ada_clf= AdaBoostClassifier(random_state=42)
xgboost_clf= xgb.XGBClassifier(random_state=42)
gradboost_clf=GradientBoostingClassifier(random_state=42)
voting_clf=VotingClassifier(estimators=[('rdf', rndforest_clf), ('xtra', extratree_clf), ('ada', ada_clf)], voting='soft')
#

#%%
#############################
##Step 4 Scoring initialization
#############################

#Lets define the scoring for the GridSearchCV
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'sensitivity': make_scorer(recall_score),
    'specificity': make_scorer(recall_score,pos_label=0),
    'precision':make_scorer(precision_score),
    'f1':make_scorer(f1_score),
    'roc_auc':make_scorer(roc_auc_score),
    'mcc':make_scorer(matthews_corrcoef)    
}

# scoring = {
#     'accuracy': make_scorer(accuracy_score),
#     'sensitivity': make_scorer(recall_score, average='weighted'),
#     'specificity': make_scorer(recall_score,labels=0, average='weighted'),
#     'precision':make_scorer(precision_score,average='weighted'),
#     'f1':make_scorer(f1_score,average='weighted'),
#     'roc_auc':make_scorer(roc_auc_score,average='weighted'),
#     'mcc':make_scorer(matthews_corrcoef)    
# }

#%%
#################################################
##Step 5 Training the data set with GridSearchCV
#################################################


##5.a Parallel approach
#######################
full_parallel_pipe_opta=Pipeline([('data_prep',dataprep_pipe_opta),('clf',dectree_clf)])

full_parallel_pipe_opta.get_params().keys()



#%% Load the model saved to avoid a new fitting
clf_fpipe_a= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\FeatureSelection_Classifier_Pipeline\GridSearchCV_results\SPECT_case\clf_fpipe_a.pkl')

#%%
param_grid_fpipe_a={'clf':[dectree_clf, rndforest_clf, extratree_clf, ada_clf, xgboost_clf,gradboost_clf,voting_clf ],
                    # 'data_prep__numeric_pipe__data_prep__data_missing__strategy':['mean','median'],
                    # 'data_prep__numeric_pipe__feat_sel__k_out_features':[*range(1,len_numerical_feats+1)],
                    # 'data_prep__numeric_pipe__feat_sel__strategy':['filter_num','filter_mutinf'],
                    'data_prep__nominal_pipe__feat_sel__k_out_features':[*range(1,len_nominal_feats+1)],
                    'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat','filter_mutinf']#,
                    #'data_prep__ordinal_pipe__feat_sel__k_out_features':[0,1],
                    #'data_prep__ordinal_pipe__feat_sel__strategy':['filter_cat','filter_mutinf','wrapper_RFE']
                    }

# param_grid_fpipe_a={'clf':[dectree_clf, rndforest_clf],
#                     'data_prep__numeric_pipe__data_prep__data_missing__strategy':['median'],
#                      'data_prep__numeric_pipe__feat_sel__k_out_features':[1,2,3],
#                      'data_prep__numeric_pipe__feat_sel__strategy':['filter_num'],
#                      'data_prep__nominal_pipe__feat_sel__k_out_features':[1,2,3],
#                      'data_prep__nominal_pipe__feat_sel__strategy':['filter_cat']#,
#                      #'data_prep__ordinal_pipe__feat_sel__k_out_features':[1],
#                      #'data_prep__ordinal_pipe__feat_sel__strategy':['filter_mutinf']
#                     }

clf_fpipe_a=GridSearchCV(full_parallel_pipe_opta,param_grid_fpipe_a,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_fpipe_a.fit(X_train,y_train)

#%% Saving the model
joblib.dump(clf_fpipe_a, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\FeatureSelection_Classifier_Pipeline\GridSearchCV_results\SPECT_case\clf_fpipe_a.pkl', compress=1)

#
#%% Printing the best estimator 
print('Best estimator of clf_fpipe_a:', clf_fpipe_a.best_params_)
print('Params of best estimator of clf_fpipe_a:', clf_fpipe_a.best_params_)
print('Score of best estimator of clf_fpipe_a:', clf_fpipe_a.best_score_)

#%% Saving the training results into dataframe
df_results_clf_fpipe_a=pd.DataFrame(clf_fpipe_a.cv_results_)
# create an excel with the cross val resutls
df_results_clf_fpipe_a.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\FeatureSelection_Classifier_Pipeline\GridSearchCV_results\SPECT_case\train_results_clf_fpipe_a.xlsx',index=True)

#%% Performing test phase with test set 
clf_fpipe_a.refit
y_pred_clf_fpipe_a=clf_fpipe_a.predict(X_test)
test_results_clf_fpipe_a={'clf':['clf_fpipe_a'],
                 'params':[clf_fpipe_a.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_clf_fpipe_a)],
                 'f1_test':[f1_score(y_test, y_pred_clf_fpipe_a)],
                 'precision_test':[precision_score(y_test, y_pred_clf_fpipe_a)],
                 'recall_test':[recall_score(y_test, y_pred_clf_fpipe_a)],
                 'specificity_test':[recall_score(y_test, y_pred_clf_fpipe_a,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_clf_fpipe_a)]    
    }

test_results_y_pred_clf_fpipe_a=pd.DataFrame(data=test_results_clf_fpipe_a)
test_results_y_pred_clf_fpipe_a.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\FeatureSelection_Classifier_Pipeline\GridSearchCV_results\SPECT_case\test_results_y_pred_clf_fpipe_a.xlsx',index=False)
print('Accuracy of test set',accuracy_score(y_test, y_pred_clf_fpipe_a))

#%%
##5.b general approach where feature selection is performed as a whole for other features
#########################################################################################
full_parallel_pipe_optb=Pipeline([('data_prep',dataprep_pipe_optb),('clf',dectree_clf)])

full_parallel_pipe_optb.get_params().keys()

#%% Load the model saved to avoid a new fitting
clf_fpipe_b= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\FeatureSelection_Classifier_Pipeline\GridSearchCV_results\SPECT_case\clf_fpipe_b.pkl')

#%%
param_grid_fpipe_b={'clf':[dectree_clf, rndforest_clf, extratree_clf, ada_clf, xgboost_clf,gradboost_clf,voting_clf ],
                    #'data_prep__data_prep__numeric_pipe__data_missing__strategy':['mean','median'],
                    'data_prep__feat_sel__k_out_features':[*range(1,len_numerical_feats+len_nominal_feats+len_ordinal_feats+1)],
                    'data_prep__feat_sel__strategy':['filter_mutinf','wrapper_RFE']
                    }

# param_grid_fpipe_b={'clf':[dectree_clf, rndforest_clf ],
#                     'data_prep__data_prep__numeric_pipe__data_missing__strategy':['mean','median'],
#                     'data_prep__feat_sel__k_out_features':[1,2,3],
#                     'data_prep__feat_sel__strategy':['filter_mutinf']
#                     }

clf_fpipe_b=GridSearchCV(full_parallel_pipe_optb,param_grid_fpipe_b,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_fpipe_b.fit(X_train,y_train)

#%% Saving the model
joblib.dump(clf_fpipe_b, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\FeatureSelection_Classifier_Pipeline\GridSearchCV_results\SPECT_case\clf_fpipe_b.pkl', compress=1)

#
#%% Printing the best estimator 
print('Best estimator of clf_fpipe_b:', clf_fpipe_b.best_params_)
print('Params of best estimator of clf_fpipe_b:', clf_fpipe_b.best_params_)
print('Score of best estimator of clf_fpipe_b:', clf_fpipe_b.best_score_)

#%% Saving the training results into dataframe
df_results_clf_fpipe_b=pd.DataFrame(clf_fpipe_b.cv_results_)
# create an excel with the cross val resutls
df_results_clf_fpipe_b.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\FeatureSelection_Classifier_Pipeline\GridSearchCV_results\SPECT_case\train_results_clf_fpipe_b.xlsx',index=True)

#%% Performing test phase with test set 
clf_fpipe_b.refit
y_pred_clf_fpipe_b=clf_fpipe_b.predict(X_test)
test_results_clf_fpipe_b={'clf':['clf_fpipe_b'],
                 'params':[clf_fpipe_b.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_clf_fpipe_b)],
                 'f1_test':[f1_score(y_test, y_pred_clf_fpipe_b)],
                 'precision_test':[precision_score(y_test, y_pred_clf_fpipe_b)],
                 'recall_test':[recall_score(y_test, y_pred_clf_fpipe_b)],
                 'specificity_test':[recall_score(y_test, y_pred_clf_fpipe_b,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_clf_fpipe_b)]    
    }

test_results_y_pred_clf_fpipe_b=pd.DataFrame(data=test_results_clf_fpipe_b)
test_results_y_pred_clf_fpipe_b.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\FeatureSelection_Classifier_Pipeline\GridSearchCV_results\SPECT_case\test_results_y_pred_clf_fpipe_b.xlsx',index=False)
print('Accuracy of test set',accuracy_score(y_test, y_pred_clf_fpipe_b))

#%%
##5.c general approach where feature selection is performed as a whole for other features
#########################################################################################
full_parallel_pipe_optc=Pipeline([('data_prep',dataprep_merge_feat),('clf',dectree_clf)])

full_parallel_pipe_optc.get_params().keys()
#%% Load the model saved to avoid a new fitting
clf_fpipe_c= joblib.load(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\FeatureSelection_Classifier_Pipeline\GridSearchCV_results\SPECT_case\clf_fpipe_c.pkl')

#%%
param_grid_fpipe_c={'clf':[dectree_clf, rndforest_clf, extratree_clf, ada_clf, xgboost_clf,gradboost_clf,voting_clf ]
                    }

# param_grid_fpipe_c={'clf':[dectree_clf, rndforest_clf ],
#                     'data_prep__numeric_pipe__data_missing__strategy':['mean','median']
#                     }

clf_fpipe_c=GridSearchCV(full_parallel_pipe_optc,param_grid_fpipe_c,scoring=scoring,refit='accuracy', cv=5,n_jobs=None)
clf_fpipe_c.fit(X_train,y_train)

# %%#%% Saving the model
joblib.dump(clf_fpipe_c, r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\FeatureSelection_Classifier_Pipeline\GridSearchCV_results\SPECT_case\clf_fpipe_c.pkl', compress=1)

#
#%% Printing the best estimator 
print('Best estimator of clf_fpipe_c:', clf_fpipe_c.best_params_)
print('Params of best estimator of clf_fpipe_c:', clf_fpipe_c.best_params_)
print('Score of best estimator of clf_fpipe_a:', clf_fpipe_c.best_score_)

#%% Saving the training results into dataframe
df_results_clf_fpipe_c=pd.DataFrame(clf_fpipe_c.cv_results_)
# create an excel with the cross val resutls
df_results_clf_fpipe_c.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\FeatureSelection_Classifier_Pipeline\GridSearchCV_results\SPECT_case\train_results_clf_fpipe_c.xlsx',index=True)

#%% Performing test phase with test set 
clf_fpipe_c.refit
y_pred_clf_fpipe_c=clf_fpipe_c.predict(X_test)
test_results_clf_fpipe_c={'clf':['clf_fpipe_c'],
                 'params':[clf_fpipe_c.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_clf_fpipe_c)],
                 'f1_test':[f1_score(y_test, y_pred_clf_fpipe_c)],
                 'precision_test':[precision_score(y_test, y_pred_clf_fpipe_c)],
                 'recall_test':[recall_score(y_test, y_pred_clf_fpipe_c)],
                 'specificity_test':[recall_score(y_test, y_pred_clf_fpipe_c,pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_clf_fpipe_c)]    
    }

test_results_y_pred_clf_fpipe_c=pd.DataFrame(data=test_results_clf_fpipe_c)
test_results_y_pred_clf_fpipe_c.to_excel(r'C:\Users\k5000751\OneDrive - Epedu O365\SeAMK\GitHub\FeatureSelection_Classifier_Pipeline\GridSearchCV_results\SPECT_case\test_results_y_pred_clf_fpipe_c.xlsx',index=False)
print('Accuracy of test set',accuracy_score(y_test, y_pred_clf_fpipe_c))


# %%
