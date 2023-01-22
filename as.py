import requests
import json
import pandas as pd
import datetime as dt
from tqdm import tqdm
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

COUNTRY = "The Netherlands"
KEY = "51692ad112e439586cfe21f4fb436f50"

START_STRING, END_STRING = "1985-01-01", "2020-01-01"
START_DATE,END_DATE = dt.datetime.strptime(START_STRING,"%Y-%m-%d"),dt.datetime.strptime(END_STRING,"%Y-%m-%d")
TOTAL_OBSERVATIONS = (END_DATE.year - START_DATE.year) *12 +1

def main():
    series_ids_list = get_all_series([])
    keep_monthly_data(series_ids_list)
    dataframe = pd.DataFrame()
    for element in tqdm(series_ids_list):
        vals = get_data_series(element['id'])
        title = element['title']
        if vals and len(vals) == TOTAL_OBSERVATIONS:
            dataframe[title] = vals
    
    with open('data.pkl', 'wb') as f:
        pickle.dump(dataframe, f)


def get_series_ids(offset):
    """
    Call the API and fetch data.
    """
    response = requests.get(f"https://api.stlouisfed.org/fred/series/search",
                        headers = {'Accept': 'application/json'},
                        params = {"search_text": COUNTRY,
                                "api_key": KEY,
                                "file_type": "json",
                                "offset": offset})

    data = json.loads(response.content)['seriess']
    return data

def get_all_series(data):
    """
    recursive call to offset the API
    """
    new_data = get_series_ids(len(data))
    
    
    if len(new_data) < 1000:
        merged_data = data + new_data
        return merged_data
    else:
        merged_data = data + new_data
        return get_all_series(merged_data)

def keep_monthly_data(data):
    for idx,element in enumerate(data):
        if element['frequency'] != "Monthly":
            data.pop(idx)

def get_data_series(id):
    response = requests.get(f"https://api.stlouisfed.org/fred/series/observations",
                        headers = {'Accept': 'application/json'},
                        params = {"series_id": id,
                                "api_key": KEY,
                                "file_type": "json",
                                "observation_start": START_STRING,
                                "observation_end": END_STRING})

    try:
        data = json.loads(response.content)['observations']
        vals = []
        for element in data:
            vals.append(element['value'])
        return vals
    except KeyError:
        return None
    

if __name__ == "__main__":
    main()

with open("data.pkl", 'rb') as f:
    dataframe = pickle.load(f)