import pandas as pd
from datetime import datetime
from dateutil import parser
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import re
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

def subset_dataframe_and_retrieve_departure_time_cabin_type(df, features_target_list):
    """
    Function to subset the dataframe to include only the features and a
    target column

    Parameters
    ----------
    df : pandas.DataFrame
        Input Dataframe containing all the data
    features_target_list : list
        list containing all the feature column names and target column name

    Returns
    -------
    features: pd.DataFrame
        features dataframe

    target : pd.DataFrame
        target dataframe containing the total fare
    """
    features = df.copy()
    features = features[features['isNonStop'] == True] 
    features = features[features_target_list]
    # Retrieving the departure time of the flight
    features.loc[:, 'departureTimeRaw'] = features['segmentsDepartureTimeRaw'].str.split('\|\|').str[0]
    # Retrieving the cabin type
    features.loc[:, 'cabinType'] = features['segmentsCabinCode'].str.split('\|\|').str[0]
    # Dropping the segmentsDepartureTimeRaw and segmentsCabinCode columns
    features.drop(columns=['segmentsDepartureTimeRaw', 'segmentsCabinCode'], inplace=True)
    features = features.sort_values(by='flightDate')
    return features

## Building Feature Functions
def remove_utc_offset(datetime_str):
    """
    # Function to remove the UTC offset from datetime strings
    """
    dt = parser.parse(datetime_str)
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')


def get_time_of_day(dt):
    """
    Function to generate time categories
    """
    hour = dt.hour
    if hour >=5 and hour < 8:
        return 'Early Morning'
    elif hour >= 8 and hour < 11:
        return 'Morning'
    elif hour >= 11 and hour < 14:
        return 'Midday'
    elif hour >= 14 and hour < 17:
        return 'Afternoon'
    elif hour >= 17 and hour < 20:
        return 'Evening'
    elif hour >= 20 and hour < 23:
        return 'Night'
    else:
        return 'Late Night'
 
def remove_utc_offset(datetime_str):
    """
    # Function to remove the UTC offset from datetime strings
    """
    dt = parser.parse(datetime_str)
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')

def generate_adjusted_date_and_time_categories(df):
    """
    Function to generate time categories and a adjusted date that sets the date
    for a time between midnight and 2 am to the day before (midnight to 2am taken
    as late night)
    """
    df_copy = df.copy() 
    # Apply the function to the DataFrame column
    df_copy['departuretime'] = df_copy['departureTimeRaw'].apply(remove_utc_offset)
    
    df_copy['departuretime'] = pd.to_datetime(df_copy['departuretime'], utc=False)
    
    df_copy['time_category'] = df_copy['departuretime'].apply(get_time_of_day)
    
    df_copy['adjusted_date'] = (df_copy['departuretime'] - pd.Timedelta(hours=2)).dt.date
    return df_copy

def split_dataset(features, target, split_date):
    """
    Function to split the dataset based on a split date.
    Rows with departure date earlier than the split date to be taken
    in the train set
    Rows with departure date greater or equal to the split date to be
    taken in the test set
    Parameters
    ----------
    features : pd.DataFrame
        Input dataframe

    target : pd.Series
        Target column

    split_date : str
        Cut-off date
    """
    features['departure_date'] = pd.to_datetime(features['departure_date'])

    split_date = pd.to_datetime(split_date)
    x_train = features[features['departure_date'] < split_date]
    x_test = features[features['departure_date'] >= split_date]
    y_train = target.loc[x_train.index]
    y_test = target.loc[x_test.index]

    x_train_cpy = x_train.copy()
    x_test_cpy = x_test.copy()

    x_train_cpy['departure_date'] = x_train_cpy['departure_date'].astype(str)
    x_test_cpy['departure_date'] = x_test_cpy['departure_date'].astype(str)

    # print the shapes of the resultant dataframes
    print(x_train_cpy.shape)
    print(y_train.shape)
    print(x_test_cpy.shape)
    print(y_test.shape)
    
    return x_train_cpy, y_train, x_test_cpy, y_test


def extract_time(input_string):
    """
    Function to extract time from input string
    """
    parsed_datetime = datetime.fromisoformat(input_string[:-6])
    return parsed_datetime.time()


def extract_hour(input_string):
    parsed_datetime = datetime.fromisoformat(input_string[:-6])
    return parsed_datetime.time().hour


def generate_date_related_features(df):
    """
    Function to generate features like day_name
    from date column and days_from_flight based on the 
    difference between search date and flight date
    """
    df_copy = df.copy()
    df_copy['days_from_flight'] = df_copy['adjusted_date'] - pd.to_datetime(df_copy['searchDate']).dt.date
    df_copy['day_of_week'] = pd.to_datetime(df_copy['adjusted_date']).dt.dayofweek
    day_mapping = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }
    df_copy['day_name'] = df_copy['day_of_week'].map(day_mapping)
    return df_copy


def rename_cols_to_appropriate_names(df):
    """
    Function to rename all the columns to appropriate names
    """
    df_copy = df.copy()
    mapping = {
        'startingAirport': 'origin_airport',
        'destinationAirport': 'destination_airport',
        'adjusted_date': 'departure_date',
        'cabinType': 'cabin_type',
        'time_category': 'time_category',
        'days_from_flight': 'days_from_flight',
        'day_name': 'day_name'
    }
    df_copy.rename(columns=mapping, inplace=True)
    return df_copy


def generate_features_target_dataframes(df):
    """
    Function to generate features and target dataframe
    Determines the target column, which is the median fare for 
                                   'startingAirport', 
                                   'destinationAirport', 
                                   'adjusted_date', 
                                   'cabinType', 
                                   'time_category', 
                                   'days_from_flight',
                                   'day_name'
    """
    df_copy = df.copy()
    modal_fares = df_copy.groupby(['origin_airport', 
                                 'destination_airport', 
                                 'departure_date', 
                                 'cabin_type', 
                                 'time_category', 
                                 'days_from_flight',
                                 'day_name'])['totalFare'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None).reset_index(name='modal_fare')

    modal_fares['days_from_flight'] = modal_fares['days_from_flight'].astype(str)
    modal_fares = modal_fares.sort_values(by='departure_date')
    target = pd.DataFrame(modal_fares.pop('modal_fare'))
    return modal_fares, target


def column_transformer_preprocessor(low_cardinal_cat, high_cardinal_cat):
    """
    Function to build a column transformer pipeline which would transform all the 
    columns to make it ready for modelling

    Parameters
    ----------
    low_cardinal_cat : list
        list of column names with low cardinality

    high_cardinal_cat : list
        list of column names with high cardinality
    """

    # Pipeline for low-cardinal categorical columns
    cat_ohe_transformer = Pipeline(
        steps = [
            ('constant_imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
            ('one_hot_encoder', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'))
        ]
    )

    # Pipeline for high-cardinal categorical columns
    cat_target_transformer = Pipeline(
        steps = [
            ('constant_imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
            ('target_encoder', TargetEncoder())
        ]
    )

    # Creating a ColumnTransformer pipeline for the data
    preprocessor = ColumnTransformer(
        transformers = [
            ('low_cardinal_cats', cat_ohe_transformer, low_cardinal_cat),
            ('high_cardinal_cats', cat_target_transformer, high_cardinal_cat)
        ]
    )

    return preprocessor
    
# function to get features
def time_category_features(df):

    df['departuretime'] = df['segmentsDepartureTimeRaw'].apply(remove_utc_offset) 
    
    df['departuretime'] = pd.to_datetime(df['departuretime'], utc=False)
    
    # time category
    df['time_category'] = df['departuretime'].apply(get_time_of_day)
    
    # departure date
    df['date'] = (df['departuretime'] - pd.Timedelta(hours=2)).dt.date

    # no. of days from flight
    df['days_from_flight'] = (df['date'] - pd.to_datetime(df['searchDate']).dt.date)
      
    return df

def date_category_features(df):
    
    df["day_of_week"] = pd.to_datetime(df['date']).dt.day_name()
    
    df["year"] = pd.to_datetime(df['date']).dt.year
    
    df["month"] = pd.to_datetime(df['date']).dt.month
    
    df['day'] = pd.to_datetime(df['date']).dt.day
    
    return df


def extract_days(duration_string):
    days = re.search(r'(\d+) days', duration_string)
    return int(days.group(1)) if days else None


def create_transformer(categories):
    return Pipeline(steps=[('ordinal_encoder', OrdinalEncoder(categories=[categories]))])
