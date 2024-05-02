import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
from joblib import load

def print_rmse_scores(y_train, train_pred, y_test, test_pred):
    """
    Function to print RMSE scores for train, validation and test sets
    """
    # Root mean squared error for train, val and test sets
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    print(f'Training RMSE: {train_rmse}')
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    print(f'Test RMSE: {test_rmse}')

def print_mae_scores(y_train, train_pred, y_test, test_pred):
    """
    Function to print MAE scores for train, validation and test sets
    """
    # Root mean squared error for train, val and test sets
    train_rmse = mean_absolute_error(y_train, train_pred)
    print(f'Training MAE: {train_rmse}')
    test_rmse = mean_absolute_error(y_test, test_pred)
    print(f'Test MAE: {test_rmse}')


def calculate_mape(actual, predicted):
    errors = np.abs((actual - predicted) / actual)
    mape = np.mean(errors) * 100
    return mape

def print_mape_scores(y_train, train_pred, y_test, test_pred):
    # Calculate MAPE for training set
    mape_train = calculate_mape(y_train, train_pred)
    print(f'Training MAPE: {mape_train:}%')

    # Calculate MAPE for testing set
    mape_test = calculate_mape(y_test, test_pred)
    print(f'Testing MAPE: {mape_test}%')

def print_r2_scores(y_train, train_pred, y_test, test_pred):
    """
    Function to print R2 scores for train, validation and test sets
    """
    # Root mean squared error for train, val and test sets
    train_r2 = r2_score(y_train, train_pred)
    print(f'Training R2: {train_r2}')
    test_r2 = r2_score(y_test, test_pred)
    print(f'Test R2: {test_r2}')


def generate_data_for_prediction(origin_airport, 
                                 destination_airport,
                                 departure_date,
                                 departure_time_category,
                                 cabin_type):
    """
    Function to generate a dataframe for prediction
    """
    date_object = datetime.strptime(departure_date, '%Y-%m-%d').date()
    day_of_week = date_object.weekday()
    # day_of_week = pd.to_datetime(departure_date).dt.dayofweek
    day_mapping = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }
    day_name = day_mapping[day_of_week]
    search_date = datetime.now().date()
    days_from_flight = date_object - search_date
    df = pd.DataFrame({
        'origin_airport': [origin_airport],
        'destination_airport': [destination_airport],
        'departure_date': [departure_date],
        'cabin_type': [cabin_type],
        'time_category': [departure_time_category],
        'days_from_flight': [days_from_flight],
        'day_name': [day_name]
    })
    return df

def predict(origin_airport, 
            destination_airport,
            departure_date,
            departure_time_category,
            cabin_type):
    """
    Function to predict airticket fare
    """
    xgb_pipe = load('../models/xgb_pipe.joblib')
    data_obs_features = generate_data_for_prediction(
        origin_airport=origin_airport, 
        destination_airport=destination_airport,
        departure_date=departure_date,
        departure_time_category=departure_time_category,
        cabin_type=cabin_type)
    pred = xgb_pipe.predict(data_obs_features)
    return pred

def calculate_regression_metrics(y_test, pred):
    rmse = mean_squared_error(y_test, pred, squared=False)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    metrics = {
        "RMSE": rmse,
        "MSE": mse,
        "R2": r2,
        "MAE": mae
    }
    return metrics

def create_comparison_dataframe(y_test, predicted_values, model_name):
    df = pd.DataFrame({
        'Actual Value (y_test)': y_test,
        f'Predicted Value ({model_name})': predicted_values,
        'Difference': abs(y_test - predicted_values)
    })
    return df
