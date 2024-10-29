import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score, precision_score
from sklearn.preprocessing import OneHotEncoder

params_for_GS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'max_features': ['auto', 'sqrt', 'log2'],
    'criterion': ['gini', 'entropy']
}


def pre_processing(df):
    
    
    df['Age'].interpolate(method='linear', inplace=True)
    df['Age'] = df['Age'].astype(int)
    columns_to_drop = ['Ticket', 'Fare', 'Cabin', 'Embarked', 'Name']
    df = df.drop(columns=columns_to_drop)
    
    encoder_sex = OneHotEncoder(sparse_output=False)
    sex_encoded = encoder_sex.fit_transform(df[['Sex']])
    sex_encoded_df = pd.DataFrame(sex_encoded, columns=encoder_sex.get_feature_names_out(['Sex']))
    df = pd.concat([df, sex_encoded_df], axis=1)
    df.drop('Sex', axis=1, inplace=True)
    
    encoder_pclass = OneHotEncoder(sparse_output=False)
    Pclass_encoded = encoder_pclass.fit_transform(df[['Pclass']])
    Pclass_encoded_df = pd.DataFrame(Pclass_encoded, columns=encoder_pclass.get_feature_names_out(['Pclass']))
    df = pd.concat([df, Pclass_encoded_df], axis=1)
    df.drop('Pclass', axis=1, inplace=True)
    
    if 'Survived' in df.columns:
        survived = df.pop('Survived')
        df['Survived'] = survived

    df = df.set_index('PassengerId')
        
    return df

def get_data(path, pre_processing_enabled=True):
    df = pd.read_csv(path)
    if pre_processing_enabled:
        df = pre_processing(df)
    return df

def get_train_test_data(train_path, test_path, pre_processing_enabled=True):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if pre_processing_enabled:
        train = pre_processing(train)
        test = pre_processing(test)
    return train, test
