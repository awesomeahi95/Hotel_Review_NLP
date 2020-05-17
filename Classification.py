from IPython.display import display
from IPython.display import Image
import pydotplus
from sklearn.externals.six import StringIO 
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from yellowbrick.model_selection import feature_importances

#===============================================================================================#

# Classification Models Class

#===============================================================================================#

class Classification():
    
    """
    This class is for performing classifcation algorithms such as Logistic Regression, Decision Tree, Random Forest, and SVM.
    
    Parameters
    ----------
    model_type: 'Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM'
    the type of classifcation algorithm you would like to apply 
    
    x_train: dataframe
    the independant variables of the training data
    
    x_val: dataframe
    the independant variables of the validation data
    
    y_train: series
    the target variable of the training data
    
    y_val: series
    the target variable of the validation data
    
    """
    
    def __init__(self,model_type,x_train,x_val,y_train,y_val):

        self.model_type = model_type
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.scores_table = pd.DataFrame()
        self.feature_importances = pd.DataFrame()
        self.name = self
        
        if self.model_type == 'Logistic Regression':
            self.technique = LogisticRegression(fit_intercept=False)
        elif self.model_type == 'Decision Tree':
            self.technique = DecisionTreeClassifier(random_state=42)
        elif self.model_type == 'Random Forest':
            self.technique = RandomForestClassifier(n_estimators=20,n_jobs=-1,random_state=42)
        elif self.model_type == "SVM":
            self.technique = SVC()
            
#===============================================================================================#

# Score Function

#===============================================================================================#

    def scores(self,model,X_train,X_val,y_train,y_val):
        
        """
        Gets a ROC AUC score for given data and creates a dataframe containing scores.

        Parameters
        ----------
        model: 
        the type of classifcation applied

        x_train: dataframe
        the independant variables of the training data

        x_val: dataframe
        the independant variables of the validation data

        y_train: series
        the target variable of the training data

        y_val: series
        the target variable of the validation data
        
        Returns
        ----------
        scores_table: a dataframe with the model used, the train AUC score and the validation AUC score

        """
        
        train_prob = model.predict_proba(X_train)[:,1]

        val_prob = model.predict_proba(X_val)[:,1]

        auc_train = round(roc_auc_score(y_train,train_prob),4)
        auc_val = round(roc_auc_score(y_val,val_prob),4)
        
        self.auc_train = auc_train
        self.auc_val = auc_val
        
        d = {'Model Name': [self.model_type],
             'Train AUC': [self.auc_train], 
             'Validation AUC': [self.auc_val]}
        self.scores_table = pd.DataFrame(data=d)
        
        return self.scores_table
        
#===============================================================================================#

# Threshold Annotation Function

#===============================================================================================#

    def annot(fpr,tpr,thr):
        
        """
        Creates an annotation for ROC plot.

        Parameters
        ----------
        fpr: numpy array 
        false positive rates for ROC 
        
        tpr: numpy array 
        true positive rates for ROC 
        
        thr: numpy array 
        threshold values for ROC 

        """
        
        k=0
        for i,j in zip(fpr,tpr):
            if k % 500 == 0:
                plt.annotate(round(thr[k],2),xy=(i,j), textcoords='data')
            k+=1
        
#===============================================================================================#

# ROC Plot Function

#===============================================================================================#
     
    def roc_plot(self,model,X_train,X_val,y_train,y_val):
        
        """
        Creates a ROC plot.

        Parameters
        ----------
        model: 
        the type of classifcation applied

        x_train: dataframe
        the independant variables of the training data

        x_val: dataframe
        the independant variables of the validation data

        y_train: series
        the target variable of the training data

        y_val: series
        the target variable of the validation data

        """
        
        train_prob = model.predict_proba(X_train)[:,1]
        val_prob = model.predict_proba(X_val)[:,1]
        plt.figure(figsize=(7,7))
        self.threshold_df = pd.DataFrame()
        for data in [[y_train, train_prob],[y_val, val_prob]]: # ,[y_test, test_prob]
            fpr, tpr, threshold = roc_curve(data[0], data[1])
            plt.plot(fpr, tpr)
        self.fpr = fpr
        self.tpr = tpr
        self.thr = threshold
        self.threshold_df["Threshold"] = self.thr
        self.threshold_df["TPR"] = self.tpr
        self.threshold_df["FPR"] = self.fpr
        Classification.annot(fpr, tpr, threshold)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.ylabel('TPR (power)')
        plt.xlabel('FPR (alpha)')
        plt.legend(['train','val'])
        plt.show()

#===============================================================================================#

# Get Scores Function

#===============================================================================================#

    def get_scores(self,param_grid,cv_type):
        
        """
        Performs a gridsearch cross validation with given hyperparameters and data.
        Gets a ROC AUC score for given data and creates a dataframe containing scores.

        Parameters
        ----------
        param_grid: dictionary 
        specified hyperparameters for chosen classification algorithm to be passed through gridsearch cross validation
        
        cv_type: 'skf'
        the type of cross validation split to be used for gridsearch

        """
        
        reg = self.technique
        fit_reg = reg.fit(self.x_train,self.y_train)
        opt_model = GridSearchCV(fit_reg,
                                 param_grid,
                                 cv=cv_type,
                                 scoring='roc_auc',
                                 return_train_score=True,
                                 n_jobs=-1)
        self.opt_model = opt_model.fit(self.x_train,self.y_train) 
        self.best_model = opt_model.best_estimator_
        self.scores = Classification.scores(self,self.best_model,self.x_train,self.x_val,self.y_train,self.y_val)
        self.best_params = opt_model.best_params_
        display(self.scores_table)
        if param_grid == {}:
            pass
        else:
            print("The best hyperparameters are: ", self.best_params,'\n')
        self.roc_plot = Classification.roc_plot(self,self.best_model,self.x_train,self.x_val,self.y_train,self.y_val)
        self.y_validated = self.best_model.predict(self.x_val)
        

#===============================================================================================#

# Optimum Plot Function

#===============================================================================================#

    def opt_plots(self):
        
        """
        Creates an optimum hyperparameter heatmap plot for decision trees and random forests.

        """
        
        if self.model_type == "Decision Tree" or self.model_type == "Random Forest":
            opt = pd.DataFrame(self.opt_model.cv_results_)
            cols = [col for col in opt.columns if ('mean' in col or 'std' in col) and 'time' not in col]
            params = pd.DataFrame(list(opt.params))
            opt = pd.concat([params,opt[cols]],axis=1,sort=False)

            plt.figure(figsize=[15,4])
            plt.subplot(121)
            sns.heatmap(pd.pivot_table(opt,index='max_depth',columns='min_samples_leaf',values='mean_train_score')*100)
            plt.title('ROC_AUC - Training')
            plt.subplot(122)
            sns.heatmap(pd.pivot_table(opt,index='max_depth',columns='min_samples_leaf',values='mean_test_score')*100)
            plt.title('ROC_AUC - Validation')
        else:
            print("This model does not have an optimum hyperparameter plot!")
        
#===============================================================================================#

# Confusion Matrix Function

#===============================================================================================#

    def conf_matrix(self,y_true, y_pred):
        
        """
        Create a confusion matrix.

        Parameters
        ----------
        y_true: series 
        containing the target variable of the validation data
        
        y_pred: series 
        containing the predicted values of the target variable

        Returns
        ----------
        scores_table: a confusion matrix

        """
        
        cm = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

        for ind, label in enumerate(y_true):
            pred = y_pred[ind]
            if label == 1: 
                if label == pred:
                    cm['TP'] += 1
                else:
                    cm['FN'] += 1
            else:
                if label == pred:
                    cm['TN'] += 1
                else:
                    cm['FP'] += 1
            self.cm_values = cm
        return cm
    
#===============================================================================================#

# Display Confusion Matrix Table Function

#===============================================================================================#
    
    def show_conf_matrix(self):
        
        """
        Displays a graphic confusion matrix.

        """
            
        Classification.conf_matrix(self,self.y_val,self.y_validated)
        cnf_matrix = confusion_matrix(self.y_val,self.y_validated)
        self.cnf_matrix = cnf_matrix

        plt.figure(figsize=(7,7))
        plt.imshow(cnf_matrix,  cmap=plt.cm.Reds) 

        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        tick_marks = np.arange(2)
        plt.xticks(tick_marks, rotation=45)
        plt.yticks(tick_marks)

        thresh = cnf_matrix.max() / 2.
        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
                plt.text(j, i, cnf_matrix[i, j],
                         horizontalalignment='center',fontsize=25)
        plt.grid(False)
        plt.colorbar
        
#===============================================================================================#

# Plot Decision Tree Function

#===============================================================================================#

    def plot_decision_tree(self):
               
        """
        Displays a graphic confusion matrix.

        """
        
        if self.model_type == "Decision Tree":
            dot_data = StringIO()
            export_graphviz(self.best_model, out_file=dot_data,  
                            filled=True, rounded=True,
                            special_characters=True)
            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
            
            graph.write_png(f'{self.model_type}.png')
            
            img = mpimg.imread(f'{self.model_type}.png')
            plt.figure(figsize=(40,50))
            plt.imshow(img)
            plt.show()  
        
        else:
            print("This model does not have a decision tree plot!")
            
#===============================================================================================#

# Feature Importance Function

#===============================================================================================#
   
    def get_feature_importances(self):
        
        """
        Create a confusion matrix.

        Parameters
        ----------
        y_true: series 
        containing the target variable of the validation data
        
        y_pred: series 
        containing the predicted values of the target variable

        Returns
        ----------
        scores_table: a confusion matrix

        """
            
        self.feature_importances = pd.DataFrame(self.best_model.feature_importances_,
                                                index = self.x_train.columns,
                                                columns=['Importance']).sort_values('Importance',ascending =False)
        return self.feature_importances

#===============================================================================================#

# Test Score Function

#===============================================================================================#

    def get_test_scores(self,X_test,y_test):
        
        """
        Gets a ROC AUC score for given data and creates a dataframe containing scores.
        Creates a ROC plot.

        Parameters
        ----------
        x_test: dataframe 
        independant variables of the test data
        
        y_test: dataframe 
        target variable of the test data

        """
            
        self.y_test = y_test
        self.x_test = X_test
        self.scores = Classification.scores(self,self.best_model,self.x_train,self.x_test,self.y_train,self.y_test)
        display(self.scores_table)
        self.roc_plot = Classification.roc_plot(self,self.best_model,self.x_train,self.x_test,self.y_train,self.y_test)
        self.y_tested = self.best_model.predict(self.x_test)
        
        return self.roc_plot
    
#===============================================================================================#

# Show Test Confusion Matrix Function

#===============================================================================================#

    def show_test_conf_matrix(self):
       
        """
        Displays a graphic confusion matrix for test data.

        """
        
        Classification.conf_matrix(self,self.y_val,self.y_tested)
        cnf_matrix = confusion_matrix(self.y_test,self.y_tested)
        self.cnf_matrix = cnf_matrix

        plt.figure(figsize=(7,7))
        plt.imshow(cnf_matrix,  cmap=plt.cm.Reds) 

        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        tick_marks = np.arange(2)
        plt.xticks(tick_marks, rotation=45)
        plt.yticks(tick_marks)

        thresh = cnf_matrix.max() / 2.
        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
                plt.text(j, i, cnf_matrix[i, j],
                         horizontalalignment='center',fontsize=25)
        plt.grid(False)
        plt.colorbar
        
#===============================================================================================#

# Optimal Threshold Calculator Function

#===============================================================================================#       
    
    def threshold_calculator(self,CTP,CFP,CTN,CFN):
        
        """
        Calulates the best threshold for given the model and the costs.

        Parameters
        ----------
        CTP: float or int
        cost of true positives
        
        CFP: float or int
        cost of false positives
        
        CTN: float or int
        cost of true negatives
        
        CFN: float or int
        cost of false negatives

        Returns
        ----------
        scores_table: a dataframe with the threshold, FPR, TPR, and fm value (in descending order)

        """
        
        TP = self.cm_values['TP']
        FP = self.cm_values['FP']
        TN = self.cm_values['TN']
        FN = self.cm_values['FN']
        Prevalance = (TP + FP)/(TP+FP+TN+FN)
        m = ((1-Prevalance)/Prevalance) * ( (CFP - CTN) / (CFN - CTP) )
        fm = []
        for i,row in self.threshold_df.iterrows():
            fm.append(row.TPR - (m*(row.FPR)))
        self.threshold_df['fm']  = fm
        return self.threshold_df.sort_values(by='fm',ascending=False).head()
    