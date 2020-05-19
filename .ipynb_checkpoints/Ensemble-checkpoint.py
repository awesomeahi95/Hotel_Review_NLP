from Classification import Classification
import pandas as pd
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, StackingClassifier
from xgboost.sklearn import XGBClassifier

#===============================================================================================#

# Ensemble Models Class

#===============================================================================================#

class Ensemble(Classification):
    
    """
    This class is for performing ensemble algorithms such as voting, adaboost, xgboost, or stacking.
    
    Parameters
    ----------
    ensemble_method: 'Voting', 'AdaBoost', 'XGBoost', 'Stacking'
    the type of ensemble algorithm you would like to apply
    
    estimators: list
    the classifcation models to be used by the ensemble algorithm
    
    x_train: dataframe
    the independant variables of the training data
    
    x_val: dataframe
    the independant variables of the validation data
    
    y_train: series
    the target variable of the training data
    
    y_val: series
    the target variable of the validation data
    
    """
    
    def __init__(self, ensemble_method, estimators, X_train, X_val, y_train, y_val):
        
        self.ensemble_method = ensemble_method
        self.x_train = X_train
        self.y_train = y_train
        self.x_val = X_val
        self.y_val = y_val
        self.model_type = ensemble_method
        self.scores_table = pd.DataFrame()
        
        if self.ensemble_method == "Voting":
            self.technique = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        elif self.ensemble_method == "AdaBoost":
            self.technique = AdaBoostClassifier(estimators, algorithm='SAMME', n_estimators=20)
        elif self.ensemble_method == "XGBoost":
            self.technique = XGBClassifier(n_jobs=-1)
        elif self.ensemble_method == "Stacking":
            self.technique = StackingClassifier(estimators)
