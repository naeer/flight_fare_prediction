import os
import json
import joblib
import catboost
import xgboost
import calendar
import pandas as pd
from datetime import datetime
import gradio as gr

# min_model = joblib.load('/models/catboost.pkl')
median_model = joblib.load('/models/catboost.pkl')
modal_model = joblib.load('/models/xgb_pipe.joblib')
mean_model = joblib.load('/models/XGB_tuned.joblib')

# Possible flight Routes
with open('/models/flightroutes.json', 'r') as file:
    routes = json.load(file)

# Airport lists
origin_airports = ['OAK', 'IAD', 'DEN', 'LGA', 'LAX', 'ONT', 'ATL', 'DFW', 'FLL', 
'CLT', 'PHL', 'TTN', 'DTW', 'JFK', 'DAL', 'BOS', 'EWR', 'SFO', 'ORD', 'MIA']

destination_airports = ['DEN', 'LAX', 'PHL', 'DTW', 'ORD', 'SFO', 'ATL', 'BOS', 'CLT',
'DFW', 'EWR', 'IAD', 'JFK', 'LGA', 'MIA', 'OAK', 'ONT', 'DAL', 'TTN', 'FLL']

# Time categories
departure_times = ['Early Morning', 'Morning', 'Midday', 'Afternoon', 'Evening', 'Night', 'Late Night']


def getfarepredictions(origin, destination, day_of_month, mm, yr, time_category, cabin):
    
    # checking that route exists
    if (origin in routes) and (destination in routes[origin]):

        # generating date features

        mm = {month: index for index, month in enumerate(calendar.month_name) if month}.get(mm, None)

        date = pd.to_datetime(f'{int(yr)}-{mm}-{int(day_of_month)}')
        today = pd.to_datetime('today')

        day_of_week = pd.Series(date).dt.dayofweek
        days_from_flight = (date - today).days

        day_mapping = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
        }

        day_name = day_mapping[date.weekday()]

        # Minimum fare prediction

        #min_df = pd.DataFrame({})

        min_fare = 0 #round(min_model.predict(min_df)[0], 2)

        # Median fare prediction
        median_df = pd.DataFrame({'segmentsDepartureAirportCode': [origin],
        'segmentsArrivalAirportCode': [destination],
        'day_of_month': [day_of_month],
        'day_of_week': [day_of_week],
        'month': [mm],
        'year': [yr],
        'time_category': [time_category],
        'segmentsCabinCode': [cabin],
        'days_from_flight': [days_from_flight]
        })

        median_fare = round(median_model.predict(median_df)[0], 2)

        # Mean Fare Prediction
        mean_df = pd.DataFrame({'startingAirport': [origin],
        'destinationAirport': [destination],
        'date': [date.strftime("%Y-%m-%d")],
        'day_of_week': [day_name],                        
        'time_category': [time_category],
        'cabin_type': [cabin],
        'days_from_flight': [days_from_flight]                       
        })

        
        mean_fare = round(mean_model.predict(mean_df)[0], 2)

        # Modal fare prediction
        modal_df = pd.DataFrame({'origin_airport': [origin],
        'destination_airport': [destination],
        'departure_date': [date.strftime("%Y-%m-%d")],
        'cabin_type': [cabin],
        'time_category': [time_category],
        'days_from_flight': [days_from_flight],
        'day_name': [day_name]
        })

        modal_fare = round(modal_model.predict(modal_df)[0], 2)

        return median_fare, mean_fare, modal_fare

    else:
        raise gr.Error('Flight route not found! Please try different airports')

# Gradio interface

with gr.Blocks() as demo:

    gr.Markdown("# Flight Fare Prediction for US flights")

    with gr.Row():
        origin = gr.Dropdown(choices=origin_airports, label="Origin Airport")
        destination = gr.Dropdown(choices=destination_airports, label="Destination Airport")
        cabin = gr.Radio(['coach', 'premium coach', 'first', 'business'], label="Cabin")
        
    with gr.Row():
        day_of_month = gr.Number(label="Day", minimum=1, maximum=31)
        mm = gr.Dropdown(calendar.month_name, label="Month")
        yr = gr.Number(label="Year", minimum=2020, maximum=2030)
        time_category = gr.Dropdown(choices=departure_times, label="Departure Time")

    btn = gr.Button("Submit")

    with gr.Row():
        # min_fare = gr.Textbox(value="", label="Minimum Prediction")
        median_fare = gr.Textbox(value="", label="Median Prediction")
        mean_fare = gr.Textbox(value="", label="Mean Prediction")
        modal_fare = gr.Textbox(value="", label="Modal Prediction")

    btn.click(fn=getfarepredictions, inputs=[origin, destination, day_of_month, mm, yr, time_category, cabin], outputs=[median_fare, mean_fare, modal_fare])


demo.launch(server_name="0.0.0.0", server_port=7860)


