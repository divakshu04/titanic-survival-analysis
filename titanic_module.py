import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
sns.set()
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class titanic_model():
    def __init__(self, model_file, scaler_file):
        with open('model', 'rb') as model_file, open('scaler', 'rb') as scaler_file:
            self.model = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None

    def load_and_clean_data(self, data_file):
        df = pd.read_csv(data_file, delimiter=',')

        df['Pclass'] = df['Pclass'].fillna(2)

        df['Sex'] = df['Sex'].fillna('male')

        df['SibSp'] = df['SibSp'].fillna(1)

        df['Parch'] = df['Parch'].fillna(0)

        df['Fare'] = df['Fare'].fillna(32.0)

        df['Age'] = df['Age'].fillna(24.0)

        df['Embarked'] = df['Embarked'].fillna('S')

        q = df['Age'].quantile(0.99)
        df = df[df['Age']<q]

        q = df['Parch'].quantile(0.99)
        df = df[df['Parch']<q]

        q = df['SibSp'].quantile(0.99)
        df = df[df['SibSp']<q]

        q = df['Fare'].quantile(0.99)
        df = df[df['Fare']<q]

        self.cleaned_data = df.copy()

        df = df.drop(['PassengerId', 'Ticket', 'Name', 'Cabin'],axis=1)

        df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

        pclass = pd.get_dummies(df['Pclass'], drop_first=True).astype(int)
        sibsp = pd.get_dummies(df['SibSp'], drop_first=True).astype(int)
        parch = pd.get_dummies(df['Parch'], drop_first=True).astype(int)
        embarked = pd.get_dummies(df['Embarked'], drop_first=True).astype(int)

        sibsp_above_2 = sibsp.loc[:, 3:].max(axis=1)
        sibsp_1 = sibsp.loc[:, [1]].max(axis=1)
        sibsp_2 = sibsp.loc[:, [2]].max(axis=1)

        df = pd.concat([df, pclass, sibsp_1, sibsp_2, sibsp_above_2, parch, embarked], axis=1)
        df = df.drop(['Pclass','SibSp','Parch','Embarked'], axis=1)

        column_name = ['Sex', 'Age', 'Fare', 'Pclass_2', 'Pclass_3', 'SibSp_1', 'SibSp_2', 'SibSp_above_2','Parch_1', 
               'Parch_2', 'Parch_3', 'Embarked_Q', 'Embarked_S']
        df.columns = column_name
        col = ['Pclass_2', 'Pclass_3', 'Sex', 'Age', 'SibSp_1', 'SibSp_2', 'SibSp_above_2','Parch_1', 
               'Parch_2', 'Parch_3','Fare', 'Embarked_Q', 'Embarked_S']
        df = df[col]

        self.preprocessed_data = df.copy()

        scaler = StandardScaler()
        columns_to_scale = ['Age', 'Fare']
        self.preprocessed_data[columns_to_scale] = scaler.fit_transform(self.preprocessed_data[columns_to_scale])
        self.preprocessed_data = self.preprocessed_data.reset_index(drop=True)
        self.data = self.preprocessed_data.copy()
    
    def predict_output(self):
        if self.data is not None:
            predictions = self.model.predict(self.data)
            prediction_labels = predictions.argmax(axis=1)
            self.cleaned_data['Survived'] = prediction_labels
            return self.cleaned_data

