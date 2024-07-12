import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
'''
Python dataclasses are a type of class used for storing data. 
They automatically generate special methods like __init__() and __repr__() that make managing and manipulating data easier. They are part of Python's standard library since Python 3.7
'''

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocess.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            num_features=['reading_score', 'writing_score']
            cat_features=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline=Pipeline(
                steps=[('imputer',SimpleImputer(strategy='median')),
                       ('Scaler',StandardScaler())]
            )
            logging.info('preprocessing on numerical features completed')
            cat_pipeline=Pipeline(
                steps=[('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder',OneHotEncoder(drop='first')),
                    ('scaler',StandardScaler(with_mean=False))]
            )

            logging.info('categorical encoding completed')

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,num_features),
                    ('categorical_pipeline',cat_pipeline,cat_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformation_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            '''
            np.c_ is a shorthand in NumPy for column-wise concatenation. It is used to join arrays along the second axis (columns).
            
            Adding features to datasets: When working with datasets, you might want to add new features (columns) to your existing data. 
            np.c_ allows you to concatenate these new features efficiently.
            '''
            train_arr=np.c_[
                 input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                 input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info(f'saved preprocessing object.')

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
