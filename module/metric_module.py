import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
import numpy as np



class get_metrics:
    """
    다중분류일때 분기 설정X get_multi_eval 메소드호출
    roc_auc_score 를 위한 proba 값은 직접 전달하도록한다. till now
    추후 binary 활용 proba값만 전달받으면 전체 적용가능하도록(정수형태로변환) 할예정

    ex) model.fit(X_train, y_train) #* 로컬에서 fit하고 pred, proba값 도출
    test_01 = get_metrics(y_test, pred, proba)
    
    
    test_01.get_reg_eval()
    return
    ----------------------------
    
    'MAE': 1.3548387096774193,
    'MSE': 4.740957966764419,
    'RMSE': 2.1773740989468067,
    'MSLE': 0.1379077355609759,
    'RMSLE': 0.37135930789597277,
    'R2': -1.7597632629234146}    
    """
    
    def __init__(self, y_test, pred, proba, threshold=0.5):
        self.y_test = y_test
        self.pred = pred
        self.threshold = threshold
        self.proba = proba
        
        
    def get_clf_eval(self):
        
        accuracy = accuracy_score(self.y_test, self.pred)
        precision = precision_score(self.y_test, self.pred)
        recall = recall_score(self.y_test, self.pred)
        f1_score = f1_score(self.y_test, self.pred)
        auc = roc_auc_score(self.y_test, self.pred)
        
        return {'Accuracy':accuracy, 'Precision':precision, 'Recall':recall, 'F1_Score':f1_score, 'AUC':auc}
        
    
    
    def get_multi_eval(self):
        acc = accuracy_score(self.y_test, self.pred)
        precision = precision_score(self.y_test, self.pred, average='weighted')
        recall = recall_score(self.y_test, self.pred, average='weighted')
        f1 = f1_score(self.y_test, self.pred, average='weighted')
        auc = roc_auc_score(self.y_test, self.proba , multi_class = 'ovr')
        
        return {'Accuracy':acc, 'Precision':precision, 'Recall':recall, 'F1_Score':f1,'AUC':auc}
    
    def get_reg_eval(self):
        MAE = mean_absolute_error(self.y_test, self.pred)
        MSE = mean_squared_error(self.y_test, self.pred)
        RMSE = np.sqrt(MSE)
        MSLE = mean_squared_log_error(self.y_test, self.pred)
        RMSLE = np.sqrt(mean_squared_log_error(self.y_test, self.pred))
        R2 = r2_score(self.y_test, self.pred)
        
        return {'MAE':MAE, "MSE":MSE, "RMSE":RMSE, "MSLE":MSLE, "RMSLE":RMSLE, "R2":R2}
        
        