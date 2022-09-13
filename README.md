# match-outcome-predictor
Use machine learning models and techniques to predict outcomes (win/loss/draw) of football matches

## Project Description

The aim of this football match outcome prediction project is to create a pipeline to systematically clean and perform feature engineering on football matches data samples and upsert the data into a database. ML models will then be trained on this data to predict match outcomes.

## Project Flow

* pipeline.py / pipeline.ipynb

    - initial pipeline for cleaning initial CSV data and generating initial features
    
* Further Engineering For ML Models/classification_dataset_cleaing 

    - this splits dataset into decades and leagues alongside creating average goals scored and conceeded per game for each team in each split dataframe.
    
* Model Training/premier_league 

    - in this repo I have only included the models created and the methods of creation for training classification models on the premier league datasets.
    
* Further Data Collection 

        - This folder includes three notebooks which contain scripts to:
            - scrape besoccer for further ELO ratings and results for all premier league fixture 2021 - 2023
            - Modify, clean and generate basic features for that data
            - Generate further features (average goals per game) and prepare data for input to ML models

* Inference/premier_league

        - In this notebook the trained models and matches for which the result is to be predicted are loaded
        - This data is normalised with the minmax scaler initially used on the training data for ML models
        - The data is then fed into the models for the predicitions to be made
## Technologies Used

To accomplish this I have used pandas and scikit.

## Pipeline

### EDA and Data Cleaning

The pipeline has been desgined to aggregate, clean, and combine multiple CSV files containing result information from a number of leagues and seasons alongside team and match information. 

This combined dataset is then merged with ELO data for each game.

### Feature Engineering

Nested functions are then run across the dataset, breaking it down into League and Seasons to calculate features such as:

* Number of goals scored and conceeded by home/away teams so far that season
* Number of points gained by home/away teams so far that season
* Season win/ loss streak for home and away teams leading into that game
* Season count of number of games in a row where the home/away team has not scored a goal

### Further Engineering for ML Models

During the feature engineering some seasons where found to be incomplete. These are collected into a list (missing_data_information_list) and removed from the combined dataframe.

Further feature were then added to the data using additional nested loops:

* Average home and away team gols scored per game
* Average home and away team gols conceeded per game

To provide more specific and targeted data to the Machine Learning Models, this data was also split in decades as the styles of football were deemed different from 1990 vs 2021.

The data was also split into league, again aiming to capture the difference in top flight football styles from Spain to England.

Further columns are then removed from the dataframe including non-numerical values, target information and any fixtures that don't contain ELO information.

### AWS RDS Upsert

Once cleaned and engineered this data is upserting into a AWS RDS SQL database.

## Creating Classification Predicition Models

### Machine Model Training

A number of ML models were trained on the decade split premier league and primera division data. Again for the purposes of this project the data was assessed on match outcomes from 2011 - 2021.

Following feautre selection to reduce overfitting, KNeighbours Classifier and Random Forest Classifier proved most adept at prediciting outcomes. Using a randomised and grid search CV on the premier league data a RFClassifier and KNeighbouts model was found to have an accuracy of 55.7 % and 56.6%  in prediciting match outcomes (win/lose/draw).


### Further Data Collection

A web scraper module was then created to scrape data from besoccer.com required for the predicitions. Following the feature selection discussed above, this only included home and away goals and home/away ELO ratings.

In order to compute an accurate features, data from 2 seasons before of matches played was collected alongside the unplayed matches.

This scraped data was then run through a modified version of the pipeline, and had appropriate features computed.

### Inference

The most recent unplayed matched were then run through the Random Forest and KNeighbours models to predict match outcomes.




