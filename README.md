# Titanic survival analysis

The Titanic Survival dataset is a widely-used dataset in the field of machine learning and data analysis. It contains information about the passengers on board the RMS Titanic, including whether they survived the shipwreck or not.


## Project Structure

titanic_code.ipynb : contains detailed code with step by step process from data cleaning, data preprocessing to model training in 'Jupyter Notebook'.

tit_train.csv : csv file for training the model.

tit_train.csv : csv file to test the model.

titanic_module.py : contains generalized python code of the titanic survival model.

predicted_test_data.ipynb : contains the predicted result of 'tit_test.csv' dataset file.

## Project Description

PassengerId: unique identifier for each passenger

Survived: whether the passenger survived (1) or not (0)

Pclass: passenger class (1 = 1st class, 2 = 2nd class, 3 = 3rd class)

Name: name of the passenger

Sex: gender of the passenger

Age: age of the passenger (in years)

SibSp: number of siblings or spouses aboard the Titanic

Parch: number of parents or children aboard the Titanic

Ticket: ticket number

Fare: passenger fare

Cabin: cabin number

Embarked: port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)


## Installation

Install 'Python' and 'Jupyter Notebook' with the following modules :

```bash
  pip install pandas
  pip install numpy
  pip install matplotlib
  pip install seaborn
  pip install pickle
  pip install sklearn
  pip install tensorflow
```
    
## Model Overview

The dataset contains data of the passengers that are boarded on Titanic in 1912. The model predicts whether a passenger survived or not when it sank on its maiden voyage in 1912.
## Model Result

After predicting whether a passenger is survived or not, we can the result:

1. PCLASS VS SURVIVAL

From the graph, we can conclude that the passenger of 3rd class had the highest probability of not being survived.

2. SEX VS SURVIVAL

From the graph, we can conclude that Male had the mre chances of being dead. The reason could be that they were sacrificing their lives on saving womens and childrens. 

3. AGE VS SURVIVAL

We can conclude that majority of the passengers that were aboard are passengers of age group of around 20-30.

4. SIBSP VS SURVIVAL

From the graph, we can conclude that passengers having no siblings or spouses were more likely to died. The reason could be that the main target was to save passengers with family and childrens.

5. PARCH VS SURVIVAL

We can conclude that passengers with no family or children aboard are more likely to die.

6. FARE VS SURVIVAL

From the graph, we can conclude that passengers those had paid less money were more likely to die. We can see from the graph, the chances of  survival of passengers with higher fare were high.

7. EMBARKED VS SURVIVAL

From the graph, we can conclude that most of the passengers were embarked to be on Southampton port, haence, there probability of dying is higher as compared to others.
