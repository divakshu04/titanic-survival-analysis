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
   
   ![Pclass vs Survival](https://github.com/divakshu04/titanic-survival-analysis/assets/127183494/a0fe291a-d215-41d6-afd3-228e659a8152)

From the graph, we can conclude that the passenger of 3rd class had the highest probability of not being survived.

2. SEX VS SURVIVAL

   ![Sex vs Survival](https://github.com/divakshu04/titanic-survival-analysis/assets/127183494/1b9741e9-f996-4632-ab95-2e728f9d2534)

From the graph, we can conclude that Male had the mre chances of being dead. The reason could be that they were sacrificing their lives on saving womens and childrens. 

3. AGE VS SURVIVAL

   ![Age vs Survival](https://github.com/divakshu04/titanic-survival-analysis/assets/127183494/ea4e347c-4b4e-4d88-8c98-0654c005ea39)

We can conclude that majority of the passengers that were aboard are passengers of age group of around 20-30.

4. SIBSP VS SURVIVAL

   ![SibSp vs Survival](https://github.com/divakshu04/titanic-survival-analysis/assets/127183494/119854a2-c169-4e06-92c7-5a0671edf105)

From the graph, we can conclude that passengers having no siblings or spouses were more likely to died. The reason could be that the main target was to save passengers with family and childrens.

5. PARCH VS SURVIVAL

   ![Parch vs Survival](https://github.com/divakshu04/titanic-survival-analysis/assets/127183494/e02b4e7a-c14d-4906-b324-2d2131c5037e)

We can conclude that passengers with no family or children aboard are more likely to die.

6. FARE VS SURVIVAL

   ![Fare vs Survival](https://github.com/divakshu04/titanic-survival-analysis/assets/127183494/63c683a0-ea0d-4bc2-b6af-1894e7a0cd9f)

From the graph, we can conclude that passengers those had paid less money were more likely to die. We can see from the graph, the chances of  survival of passengers with higher fare were high.

7. EMBARKED VS SURVIVAL

   ![Embarked vs Survival](https://github.com/divakshu04/titanic-survival-analysis/assets/127183494/5b6e449f-9d2b-4cba-a2b3-f058e33c7b19)

From the graph, we can conclude that most of the passengers were embarked to be on Southampton port, haence, there probability of dying is higher as compared to others.
