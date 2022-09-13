# match-outcome-predictor
Use ML techniques to predict score outcomes of football matches

## Project Description

The aim of this football match outcome prediction project is to create a pipeline to systematically clean and perform feature engineering on football matches data samples and upsert the data into a database. ML models will then be trained on this data to predict match outcomes.

## Technologies Used

To accomplish this I have used pandas and sklearn.

## Pipeline

### Cleaning and Aggregating Dataset

The pipeline has been desgined to aggregate, clean, and combine multiple CSV files containing result information from a number of leagues and seasons alongside team and match information. 

This combined dataset is then merged with ELO data for each game.

### Feature Engineering

Nested functions are then run across the dataset, breaking it down into League and Seasons to calculate features such as:

* Number of goals scored and conceeded by home/away teams so far that season
* Number of points gained by home/away teams so far that season
* Season win/ loss streak for home and away teams leading into that game
* Season count of number of games in a row where the home/away team has not scored a goal

### Further cleaning and manipulation

During the feature engineering some seasons where found to be incomplete. These are collected into a list (missing_data_information_list) and removed from the combined dataframe.

Further feature were then added to the data using additional nested loops:

* Average home and away team gols scored per game
* Average home and away team gols conceeded per game

To provide more specific and targeted data to the Machine Learning Models, this data was also split in decades as the styles of football were deemed different from 1990 vs 2021.

The data was also split into league, again aiming to capture the difference in top flight football styles from Spain to England.

Further columns are then removed from the dataframe including non-numerical values, target information and any fixtures that don't contain ELO information.

### AWS RDS Upsert

Once cleaned and engineered this data is upserting into a AWS RDS SQL database.

### Machine Model Training

A number of ML models were trained on the decade split premier league and primera division data. Again for the purposes of this project the data was assessed on match outcomes from 2011 - 2021.

Following feautre selection to reduce overfitting, KNeighbours Classifier and Random Forest Classifier proved most adept at prediciting outcomes. Using a randomised and grid search CV on the premier league data a RFClassifier model was founf to have an accuracy of 56.7 % in prediciting match outcomes (win/lose/draw).

### Inference

A web scraper module was then created to scrape data from besoccer.com required for the predicitions.




