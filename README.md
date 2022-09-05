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

Further columns are then removed from the dataframe including non-numerical values, target information and any fixtures that don't contain ELO information.

### AWS RDS Upsert

Once cleaned and engineered this data is upserting into a AWS RDS SQL database.

