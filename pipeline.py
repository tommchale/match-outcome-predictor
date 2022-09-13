import pandas as pd
import numpy as np
import glob
import os
import pickle
from decouple import Config, RepositoryEnv
from sqlalchemy import create_engine


def load_and_combine_result_data_from_csv(results_path) -> pd.DataFrame:
    '''load_and_combine_result_data_from_csv 
    Load all individual season and league result data and combine into one data frame.

    Arguments:
        results_path -- file path to folder where all the results data is stored

    Returns:
        Dataframe containing merged results for all leagues and seasons within results folder.
    '''

    all_files = glob.glob(os.path.join(results_path, "*"))
    df_list = []

    for path in all_files:
        all_csv_files = glob.glob(os.path.join(path, "*.csv"))
        df_name = (path.split("/")[9])
        df_list.append(f"{df_name}_df")
        globals()[f"{df_name}_df"] = pd.concat((pd.read_csv(f)
                                                for f in all_csv_files), ignore_index=True)

    combined_results_df = pd.concat(
        (globals()[df] for df in df_list), ignore_index=True)

    return combined_results_df


def load_match_and_team_info_from_csv(info_path, team_path) -> pd.DataFrame:
    '''load_match_and_team_info_from_csv 
    Load additional game information such as team stadium capacities and referee names.

    Arguments:
        info_path -- file path to the additional match information file
        team_path -- file path to the additional team information file

    Returns:
        _description_
    '''
    match_info_df = pd.read_csv(info_path)
    team_info_df = pd.read_csv(team_path)

    return match_info_df, team_info_df


def _clean_results_data(combined_results_df) -> pd.DataFrame:
    '''_clean_results_data 
    Caller function to clean results data. This includes:
    1. Removing unnamed values
    2. Several scores in an invalid format so these were re-formatted or removed
    3. Match links standardised to enable merging or separate dataframes
    4. Results data types converted to enable calculations during feature engineering.

    Arguments:
        combined_results_df -- Dataframe containing merged results, match and team information.

    Returns:
        _description_
    '''

    combined_results_unnamed_removed_df = _remove_unnamed_values(
        combined_results_df)
    combined_results_remove_invalid_scores_df = _remove_invalid_scores(
        combined_results_unnamed_removed_df)
    combined_results_standarised_match_links_df = _standardise_match_links(
        combined_results_remove_invalid_scores_df)
    combined_results_data_types_df = _convert_results_data_types(
        combined_results_standarised_match_links_df)

    return combined_results_data_types_df


def _convert_results_data_types(combined_results_df):
    """
    Function to update data types of columns to the erquired formats.
    """
    # Update data types
    combined_results_df["Home_Team"] = combined_results_df["Home_Team"].astype(
        'category')
    combined_results_df["Away_Team"] = combined_results_df["Away_Team"].astype(
        'category')
    combined_results_df["Result"] = combined_results_df["Result"].astype(
        'string')
    combined_results_df["Season"] = combined_results_df["Season"].astype('int')
    combined_results_df["Round"] = combined_results_df["Round"].astype('int')
    combined_results_df["League"] = combined_results_df["League"].astype(
        'string')
    combined_results_df["Link"] = combined_results_df["Link"].astype('string')

    return combined_results_df


def _remove_unnamed_values(combined_results_df):
    """
    Function to remove presence of unknown variable
    """
    # remove unnamed values
    combined_results_df.drop("Unnamed: 0", axis=1, inplace=True)
    return combined_results_df


def _remove_invalid_scores(combined_results_df):
    '''_remove_invalid_scores 
    Function to remove invalid scores.

    A number of scores are in invlid formats. This function modifies wha is known and removes what isn't.

    Arguments:
        combined_results_df -- Combined results dataframe

    Returns:
        updated combined results dataframe
    '''

    # identify and remove invalid scores

    invalid_score_1 = (combined_results_df.loc[:, "Result"].str.contains(":"))
    invalid_score_2 = (combined_results_df.loc[:, "Result"].str.len() > 5)
    invalid_score_3 = (combined_results_df.loc[:, "Result"].str.len() < 2)

    combined_results_df.replace("1 (0-0) 1", np.NaN, inplace=True)
    combined_results_df.replace("0 (0-0) 0", "0-0", inplace=True)
    combined_results_df.replace("3 (3-2) 2", "3-2", inplace=True)
    combined_results_df.replace("0 (0-1) 1", "0-1", inplace=True)

    combined_results_df[invalid_score_1] = np.NaN
    combined_results_df[invalid_score_2] = np.NaN
    combined_results_df[invalid_score_3] = np.NaN

    combined_results_df.dropna(subset=["Result"], inplace=True)

    return combined_results_df


def _standardise_match_links(combined_results_df):
    '''_standardise_match_links 
    Function to standardise the links to enable merging between datasets.
    '''
    combined_results_df["Link"] = combined_results_df["Link"].apply(
        lambda x: "".join(x.replace(x.split("/")[6], "")) + x.split("/")[6][0:4])

    return combined_results_df


def _remove_partial_complete_league(combined_ELO_df):
    '''_remove_partial_complete_league 
    Function to remove presence of leagues that contain only one round of results

    Arguments:
        combined_results_df -- _description_
    '''
    for league in combined_ELO_df["League"].unique():
        for season in combined_ELO_df["Season"].unique():
            if len(combined_ELO_df.loc[(combined_ELO_df.League == league) & (
                    combined_ELO_df.Season == season)]["Round"].unique()) <= 1:
                combined_ELO_df.drop(combined_ELO_df.index[(combined_ELO_df.League == league) & (
                    combined_ELO_df.Season == season)], inplace=True)

    return combined_ELO_df


def _clean_match_info_data(match_info_df) -> pd.DataFrame:
    '''clean_match_info_data 


    Returns:
        _description_
    '''
    match_info_card_df = _replace_missing_card_data(match_info_df)
    match_info_data_types_converted_df = _convert_match_info_data_types(
        match_info_card_df)
    match_info_standardised_df = _standardise_referee_and_links(
        match_info_data_types_converted_df)

    return match_info_standardised_df


def _replace_missing_card_data(match_info_df):
    '''_replace_missing_card_data _summary_

    Arguments:
        match_info_df -- _description_
    '''
    match_info_df["Home_Yellow"].replace(np.NaN, 0, inplace=True)
    match_info_df["Home_Red"].replace(np.NaN, 0, inplace=True)
    match_info_df["Away_Yellow"].replace(np.NaN, 0, inplace=True)
    match_info_df["Away_Red"].replace(np.NaN, 0, inplace=True)

    return match_info_df


def _convert_match_info_data_types(match_info_df):
    '''_convert_data_types 

    Arguments:
        match_info_df -- _description_
    '''
    match_info_df["Date_New"] = match_info_df["Date_New"].astype('datetime64')
    match_info_df["Referee"] = match_info_df["Referee"].astype('string')
    match_info_df["Home_Yellow"] = match_info_df["Home_Yellow"].astype('int')
    match_info_df["Home_Red"] = match_info_df["Home_Red"].astype('int')
    match_info_df["Away_Yellow"] = match_info_df["Away_Yellow"].astype('int')
    match_info_df["Away_Red"] = match_info_df["Away_Red"].astype('int')
    match_info_df["Link"] = match_info_df["Link"].astype('string')

    return match_info_df


def _standardise_referee_and_links(match_info_df):
    '''_standardise_referee_and_links _summary_

    Arguments:
        match_info_df -- _description_
    '''
    match_info_df["Referee"] = match_info_df["Referee"].apply(
        lambda x: (x.replace("\r\n", "")).replace("Referee: ", ""))
    match_info_df["Link"] = match_info_df["Link"].apply(
        lambda x: "https://www.besoccer.com" + x)

    return match_info_df


def _clean_team_info_data(team_info_df):
    '''clean_team_info_data 

    Arguments:
        team_info_df -- _description_
    '''
    team_info_df["Capacity"] = team_info_df["Capacity"].apply(
        lambda x: x.replace(",", ""))
    team_info_df["Capacity"] = team_info_df["Capacity"].astype('int')

    # rename column in team info

    team_info_df = team_info_df.rename(columns={'Team': 'Home_Team'})

    return team_info_df


def clean_all_dataframes(combined_results_df, match_info_df, team_info_df):
    combined_results_cleaned_df = _clean_results_data(combined_results_df)
    match_info_cleaned_df = _clean_match_info_data(match_info_df)
    team_info_cleaned_df = _clean_team_info_data(team_info_df)

    return combined_results_cleaned_df, match_info_cleaned_df, team_info_cleaned_df


def add_features_to_results_df(combined_results_cleaned_df):
    '''add_features_to_results_df _summary_

    Arguments:
        combined_results_cleaned_df -- _description_
    '''
    combined_results_cleaned_df["Home_Goals"] = combined_results_cleaned_df["Result"].apply(
        lambda x: x.split("-")[0])
    combined_results_cleaned_df["Away_Goals"] = combined_results_cleaned_df["Result"].apply(
        lambda x: x.split("-")[1])
    combined_results_cleaned_df["Home_Win"] = combined_results_cleaned_df[
        "Home_Goals"] > combined_results_cleaned_df["Away_Goals"]
    combined_results_cleaned_df["Away_Win"] = combined_results_cleaned_df[
        "Home_Goals"] < combined_results_cleaned_df["Away_Goals"]

    combined_results_cleaned_df["Home_Goals"] = combined_results_cleaned_df["Home_Goals"].astype(
        'int')
    combined_results_cleaned_df["Away_Goals"] = combined_results_cleaned_df["Away_Goals"].astype(
        'int')

    return combined_results_cleaned_df


def combine_dataframes(combined_results_cleaned_df, match_info_cleaned_df, team_info_cleaned_df) -> pd.DataFrame:
    '''combine_dataframes _summary_

    Arguments:
        combined_results_cleaned_df -- _description_
        match_info_cleaned_df -- _description_
        team_info_cleaned_df -- _description_

    Returns:
        _description_
    '''
    # TODO: If these values are not being used in the ML models can consider removal of this option.
    result_match_df = pd.merge(
        combined_results_cleaned_df, match_info_cleaned_df, how='inner', on="Link")
    combined_df = pd.merge(
        result_match_df, team_info_cleaned_df, how='outer', on="Home_Team")

    return combined_df


def _load_ELO_pickle_data() -> pd.DataFrame:
    ELO_dict = pickle.load(open('2. Feature Engineering/elo_dict.pkl', 'rb'))
    ELO_df = pd.DataFrame(ELO_dict.items(), columns=["Link", "ELO_dict"])

    return ELO_df


def _clean_EDA_ELO_data(ELO_df):
    ELO_df["ELO_home"] = (ELO_df["ELO_dict"]).apply(lambda x: x['Elo_home'])
    ELO_df["ELO_away"] = (ELO_df["ELO_dict"]).apply(lambda x: x['Elo_away'])
    ELO_df.drop(columns="ELO_dict", inplace=True)
    ELO_df["Link"] = ELO_df["Link"].apply(lambda x: "".join(
        x.replace(x.split("/")[6], "")) + x.split("/")[6][0:4])

    return ELO_df


def _merge_ELO_combined_df(combined_df, ELO_df):

    combined_ELO_df = pd.merge(combined_df, ELO_df, how='inner', on="Link")

    return combined_ELO_df


def load_clean_merge_ELO_data(combined_df):

    ELO_df = _load_ELO_pickle_data()
    ELO_cleaned_df = _clean_EDA_ELO_data(ELO_df)
    combined_ELO_df = _merge_ELO_combined_df(combined_df, ELO_cleaned_df)

    return combined_ELO_df


def create_new_feature_columns(combined_elo_df):
    '''create_new_feature_columns 
    Function to create columns filled with zeros for each required feature

    Arguments:
        combined_local_elo_df -- _description_
    '''
    combined_elo_df["home_team_total_goals_scored_so_far"] = 0
    combined_elo_df["home_team_total_goals_conceeded_so_far"] = 0
    combined_elo_df["home_team_current_win_streak"] = 0
    combined_elo_df["home_team_current_loss_streak"] = 0
    combined_elo_df["home_team_total_points_so_far"] = 0
    combined_elo_df["home_team_current_goal_drought"] = 0
    combined_elo_df["home_team_total_wins_so_far"] = 0

    combined_elo_df["away_team_total_goals_scored_so_far"] = 0
    combined_elo_df["away_team_total_goals_conceeded_so_far"] = 0
    combined_elo_df["away_team_current_win_streak"] = 0
    combined_elo_df["away_team_current_loss_streak"] = 0
    combined_elo_df["away_team_total_points_so_far"] = 0
    combined_elo_df["away_team_current_goal_drought"] = 0
    combined_elo_df["away_team_total_wins_so_far"] = 0

    return combined_elo_df


def group_by_league_df(combined_elo_df: pd.DataFrame, league) -> pd.DataFrame:
    """
    Function to group full dataframe into only the required league

    """
    grouped_df = combined_elo_df.groupby("League")
    league_df = grouped_df.get_group(league)

    return league_df


def group_by_season_df(league_df: pd.DataFrame, season) -> pd.DataFrame:
    """
    Function to group full dataframe into seasons

    """

    grouped_df = league_df.groupby("Season")
    season_df = grouped_df.get_group(season)
    # Order season by round
    season_df = season_df.sort_values(by=['Round'], ascending=[True])

    return season_df


def create_summary_template(season_df):
    '''create_summary_template 
    Function to create auxiliary template summary per season

    Arguments:
        season_df -- _description_

    Returns:
        _description_
    '''

    summary_df_template = pd.DataFrame(columns=[
        'team_name', 'season', 'league', 'total_goals_scored_so_far', 'total_goals_conceeded_so_far',
        'recent_win_streak', 'recent_loss_streak', 'total_points_so_far', 'recent_goal_drought',
        'total_wins_so_far'
    ])
    team_group = season_df.groupby("Home_Team")

    for home_team_name in season_df["Home_Team"].unique():
        team_df = team_group.get_group(home_team_name)
        team_name = home_team_name
        season = team_df["Season"].unique()[0]
        league = team_df["League"].unique()[0]

        team_stats_dict = {
            'team_name': team_name, 'season': season,
            'league': league, 'total_goals_scored_so_far': 0,
            'total_goals_conceeded_so_far': 0, 'recent_win_streak': 0,
            'recent_loss_streak': 0, 'total_points_so_far': 0,
            'recent_goal_drought': 0, 'total_wins_so_far': 0
        }
        team_stats_dict_df = pd.DataFrame([team_stats_dict])
        summary_df_template = pd.concat(
            [summary_df_template, team_stats_dict_df], ignore_index=True)

    print("Summary Template Complete")
    return summary_df_template


def populate_season_stats(season_df, summary_df_template):
    '''populate_season_stats 
    Function to add to season templte summary per round.

    Arguments:
        season_df -- _description_
        summary_df_template -- _description_

    Returns:
        _description_
    '''
    round_df_grouped = season_df.groupby("Round")

    # TODO: This doesn't look right - need to test!!

    for round in season_df["Round"].unique():
        round_df = round_df_grouped.get_group(round)
        # Set the values of the previous match
        for team in list(summary_df_template["team_name"]):

            populated_season_df = _add_home_team_feature_total_to_df(
                "home_team_total_goals_scored_so_far", "total_goals_scored_so_far", team, season_df, summary_df_template, round)
            populated_season_df = _add_home_team_feature_total_to_df(
                "home_team_total_goals_conceeded_so_far", "total_goals_conceeded_so_far", team, season_df, summary_df_template, round)
            populated_season_df = _add_home_team_feature_total_to_df(
                "home_team_total_points_so_far", "total_points_so_far", team, season_df, summary_df_template, round)
            populated_season_df = _add_home_team_feature_total_to_df(
                "home_team_total_wins_so_far", "total_wins_so_far", team, season_df, summary_df_template, round)

            populated_season_df = _add_away_team_feature_total_to_df(
                "away_team_total_goals_scored_so_far", "total_goals_scored_so_far", team, season_df, summary_df_template, round)
            populated_season_df = _add_away_team_feature_total_to_df(
                "away_team_total_goals_conceeded_so_far", "total_goals_conceeded_so_far", team, season_df, summary_df_template, round)
            populated_season_df = _add_away_team_feature_total_to_df(
                "away_team_total_points_so_far", "total_points_so_far", team, season_df, summary_df_template, round)
            populated_season_df = _add_away_team_feature_total_to_df(
                "away_team_total_wins_so_far", "total_wins_so_far", team, season_df, summary_df_template, round)

        for index, row in round_df.iterrows():

            home_team_name = row["Home_Team"]
            away_team_name = row["Away_Team"]
            season = row["Season"]
            league = row["League"]
            home_goals_scored = row["Home_Goals"]
            away_goals_scored = row["Away_Goals"]
            home_win = row["Home_Win"]
            away_win = row["Away_Win"]

            # Home Team
            summary_df_template = _goals_and_points_totaller(
                summary_df_template, home_team_name, home_goals_scored, away_goals_scored, season, league)
            # Away Team
            summary_df_template = _goals_and_points_totaller(
                summary_df_template, away_team_name, away_goals_scored, home_goals_scored, season, league)

        print(f"Summary DF Template Populated for Round {round}")

    print(f"Summary DF Template Populated for season {season}")

    return summary_df_template, populated_season_df


def _goals_and_points_totaller(summary_df_template, team, goal_option_1, goal_option_2, season, league):

    summary_df_template.loc[((summary_df_template.team_name == team) & (
        summary_df_template.season == season) & (
        summary_df_template.league == league)), "total_goals_scored_so_far"] += goal_option_1

    summary_df_template.loc[((summary_df_template.team_name == team) & (
        summary_df_template.season == season) & (
        summary_df_template.league == league)), "total_goals_conceeded_so_far"] += goal_option_2

    if goal_option_1 > goal_option_2:

        summary_df_template.loc[((summary_df_template.team_name == team) & (
            summary_df_template.season == season) & (
            summary_df_template.league == league)), "total_points_so_far"] += 3
        summary_df_template.loc[((summary_df_template.team_name == team) & (
            summary_df_template.season == season) & (
            summary_df_template.league == league)), "total_wins_so_far"] += 1

    if goal_option_1 == goal_option_2:

        summary_df_template.loc[((summary_df_template.team_name == team) & (
            summary_df_template.season == season) & (
            summary_df_template.league == league)), "total_points_so_far"] += 1

    return summary_df_template


def _add_home_team_feature_total_to_df(feature, summary_feature, team, df, summary_df_template, round):

    df.loc[(df.Home_Team == team) & ((
        df.Round == round)), feature] = summary_df_template.loc[(
            summary_df_template.team_name == team), summary_feature].values[0]
    return df


def _add_away_team_feature_total_to_df(feature, summary_feature, team, df, summary_df_template, round):

    df.loc[((df.Away_Team == team) & (
        df.Round == round)), feature] = summary_df_template.loc[((
            summary_df_template.team_name == team)), summary_feature].values[0]
    return df


def populate_streak_season_stats(summary_df_populated, populated_season_df, season, league, missing_data_information_list):
    '''populate_streak_season_stats 
    Function to calculate the recent form going into game, such as win / loss streaks as well as goal drought.

    Arguments:
        summary_df_populated -- _description_
        populated_season_df -- _description_

    Returns:
        _description_
    '''

    home_team_group = populated_season_df.groupby("Home_Team")
    away_team_group = populated_season_df.groupby("Away_Team")

    for team in list(summary_df_populated["team_name"]):

        try:

            home_team_df = home_team_group.get_group(team)
            away_team_df = away_team_group.get_group(team)
            team_df = pd.concat([home_team_df, away_team_df],
                                axis=0, ignore_index=True)
            team_df = team_df.sort_values(by=['Round'], ascending=[True])
            win_streak = 0
            loss_streak = 0
            drought_streak = 0

            round_df_grouped = team_df.groupby("Round")
            print(f"Completing Streak Calculation for {team}...")
            for round in team_df["Round"].unique():
                round_df = round_df_grouped.get_group(round)
                # Here so that is shows for the round after

                populated_season_df = _add_home_team_feature_total_to_df(
                    "home_team_current_win_streak", "recent_win_streak", team, populated_season_df, summary_df_populated, round)
                populated_season_df = _add_home_team_feature_total_to_df(
                    "home_team_current_loss_streak", "recent_loss_streak", team, populated_season_df, summary_df_populated, round)
                populated_season_df = _add_home_team_feature_total_to_df(
                    "home_team_current_goal_drought", "recent_goal_drought", team, populated_season_df, summary_df_populated, round)

                populated_season_df = _add_away_team_feature_total_to_df(
                    "away_team_current_win_streak", "recent_win_streak", team, populated_season_df, summary_df_populated, round)
                populated_season_df = _add_away_team_feature_total_to_df(
                    "away_team_current_loss_streak", "recent_loss_streak", team, populated_season_df, summary_df_populated, round)
                populated_season_df = _add_away_team_feature_total_to_df(
                    "away_team_current_goal_drought", "recent_goal_drought", team, populated_season_df, summary_df_populated, round)

                for index, row in round_df.iterrows():

                    home_team_name = row["Home_Team"]
                    away_team_name = row["Away_Team"]
                    season = row["Season"]
                    league = row["League"]
                    home_win = row["Home_Win"]
                    away_win = row["Away_Win"]
                    home_goals_scored = row["Home_Goals"]
                    away_goals_scored = row["Away_Goals"]

                    # Winning Streak

                    if (home_team_name == team and home_win == True) or (away_team_name == team and away_win == True):
                        win_streak += 1

                    if (home_team_name == team and home_win == False) or (away_team_name == team and away_win == False):
                        win_streak = 0

                    # Losing Streak
                    if (home_team_name == team and home_goals_scored < away_goals_scored) or (away_team_name == team and home_goals_scored > away_goals_scored):
                        loss_streak += 1
                    if (home_team_name == team and home_goals_scored > away_goals_scored) or (away_team_name == team and home_goals_scored < away_goals_scored):
                        loss_streak = 0

                    # Goal Drought
                    if (home_team_name == team and home_goals_scored == 0) or (away_team_name == team and away_goals_scored == 0):
                        drought_streak += 1
                    if (home_team_name == team and home_goals_scored > 0) or (away_team_name == team and away_goals_scored > 0):
                        drought_streak = 0

                    summary_df_populated.loc[((summary_df_populated.team_name == team) & (
                        summary_df_populated.season == season) & (
                        summary_df_populated.league == league)), "recent_win_streak"] = win_streak

                    summary_df_populated.loc[((summary_df_populated.team_name == team) & (
                        summary_df_populated.season == season) & (
                        summary_df_populated.league == league)), "recent_loss_streak"] = loss_streak

                    summary_df_populated.loc[((summary_df_populated.team_name == team) & (
                        summary_df_populated.season == season) & (
                        summary_df_populated.league == league)), "recent_goal_drought"] = drought_streak
        except KeyError:
            print(f"The league {league} in season {season} appears incomplete")
            missing_data_information_list.append(f"{league}: {season}")

    print(f"Streak Stats Completed for season {season}")

    return summary_df_populated, populated_season_df, missing_data_information_list


def merge_season_df_with_combined_df(combined_ELO_df, populated_season_df):

    combined_ELO_df.update(populated_season_df, overwrite=True)
    return combined_ELO_df


def _remove_incomplete_seasons(missing_information_list, combined_elo_with_features_df):

    missing_info_set = set(missing_information_list)
    missing_info_deduplicated_list = list(missing_info_set)
    missing_info_deduplicated_list
    for item in missing_info_deduplicated_list:
        league = item.split(": ")[0]
        season = item.split(": ")[1]
        combined_elo_with_features_df.drop(combined_elo_with_features_df.index[(combined_elo_with_features_df.League == league) & (
            combined_elo_with_features_df.Season == season)], inplace=True)

    return combined_elo_with_features_df


def load_clean_merge_all_datasets():

    results_path = r'/Users/tom/Documents/Coding/AiCore/Projects/4. Football Match Outcome Predictor /Results'
    info_path = r'/Users/tom/Documents/Coding/AiCore/Projects/4. Football Match Outcome Predictor /Other/Match_Info.csv'
    team_path = r'/Users/tom/Documents/Coding/AiCore/Projects/4. Football Match Outcome Predictor /Other/Team_Info.csv'

    # Caller Functions

    combined_results_df = load_and_combine_result_data_from_csv(results_path)
    match_info_df, team_info_df = load_match_and_team_info_from_csv(
        info_path, team_path)
    combined_results_cleaned_df, match_info_cleaned_df, team_info_cleaned_df = clean_all_dataframes(
        combined_results_df, match_info_df, team_info_df)
    # changed to indlude combined_results_features_df in the combine datafraemes function - check if this has changed anything
    combined_results_features_df = add_features_to_results_df(
        combined_results_cleaned_df)
    combined_df = combine_dataframes(
        combined_results_features_df, match_info_cleaned_df, team_info_cleaned_df)
    combined_ELO_df = load_clean_merge_ELO_data(combined_df)
    combined_ELO_df = _remove_partial_complete_league(combined_ELO_df)

    return combined_ELO_df


def create_new_features(combined_ELO_df):

    missing_data_information_list = []

    combined_elo_with_features_df = create_new_feature_columns(combined_ELO_df)
    for league in combined_elo_with_features_df["League"].unique():
        league_df = group_by_league_df(combined_elo_with_features_df, league)
        for season in league_df["Season"].unique():
            season_df = group_by_season_df(league_df, season)
            summary_df_template = create_summary_template(season_df)
            summary_df_populated, populated_season_df = populate_season_stats(
                season_df, summary_df_template)
            # Why is the summary_df_populated needed?
            summary_df_populated_with_streak, populated_with_streak_season_df, missing_data_information_list = populate_streak_season_stats(
                summary_df_populated, populated_season_df, season, league, missing_data_information_list)
            combined_elo_with_features_df = merge_season_df_with_combined_df(
                combined_elo_with_features_df, populated_with_streak_season_df)

        print(f"League {league} completed")

    return combined_elo_with_features_df, missing_data_information_list


def _connect_to_RDS() -> create_engine:
    '''_connect_to_RDS 
    Method to create connection to RDS database using sqlalchemy

    Returns:
        engine - connection

    '''
    DOTENV_FILE = '/Users/tom/Documents/Coding/AiCore/Projects/4. Football Match Outcome Predictor /match-outcome-predictor/.env'
    env_config = Config(RepositoryEnv(DOTENV_FILE))

    RDS_DATABASE_PASSWORD = env_config.get('RDS_DATABASE_PASSWORD')
    RDS_DATABASE_TYPE = env_config.get('RDS_DATABASE_TYPE')
    RDS_DBAPI = env_config.get('RDS_DBAPI')
    RDS_ENDPOINT = env_config.get('RDS_ENDPOINT')
    RDS_USER = env_config.get('RDS_USER')
    RDS_DATABASE = env_config.get('RDS_DATABASE')
    PORT = 5432

    engine = create_engine(
        f"{RDS_DATABASE_TYPE}+{RDS_DBAPI}://{RDS_USER}:{RDS_DATABASE_PASSWORD}@{RDS_ENDPOINT}:{PORT}/{RDS_DATABASE}")

    return engine


def _upload_dataframes_to_rds(df):
    '''_upload_dataframes_to_rds 
    If dataframe exists, append it to SQL database.
    '''

    engine = _connect_to_RDS()
    df.to_sql('match_outcomes', engine, if_exists='append')


def run_pipeline():

    combined_ELO_df = load_clean_merge_all_datasets()
    combined_elo_with_features_df, missing_data_information_list = create_new_features(
        combined_ELO_df)
    combined_elo_with_partial_seasons_removed_df = _remove_incomplete_seasons(
        missing_data_information_list, combined_elo_with_features_df)
    _upload_dataframes_to_rds(combined_elo_with_partial_seasons_removed_df)
