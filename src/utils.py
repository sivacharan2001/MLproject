import os
import sys
from src.logger import logging
import numpy as np 
import pandas as pd
import dill
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models):

    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]

            #model training
            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)

            y_test_pred=model.predict(X_test)

            #accuray
            train_model_accuracy=r2_score(y_train,y_train_pred)
            test_model_accuracy=r2_score(y_test,y_test_pred)

            logging.info('Test accuracy is done')
            report[list(models.keys())[i]]=test_model_accuracy
        return report

    except Exception as e:
        CustomException(e,sys)