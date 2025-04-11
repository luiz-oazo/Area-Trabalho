import pandas as pd
import numpy as np

def process(df):
    no_results = [
        'Nenhum resultado disponível',
        'Os resultados para este evento podem ser acessados através da sua Página Oficial.'
    ]
    
    # transpose data frame
    data = df.T
    
    # rename & drop unnecessary columns 
    data = data.drop(columns = "url_details")
    data = data.rename(columns = {
        "data": "date",
        "categories": "json"
    })
    
    # remove entries where race results aren't available
    data = data[~data["json"].isin(no_results)]
    
    # extract the race distance and explode dataframe
    data["race_distance"] = data.apply(lambda df: list(df["json"].keys()), axis = 1)
    data = data.explode("race_distance")
    
    # select race info for each distance
    data["json"] = data.apply(lambda df: df["json"][df["race_distance"]], axis = 1)
    
    # extract group category and explode dataframe
    data["category"] = data.apply(lambda df: list(df["json"].keys()), axis = 1)
    data = data.explode("category")
    
    # select race info for each distance + category
    data["json"] = data.apply(lambda df: df["json"][df["category"]], axis = 1)
    
    # extract information about each athlete
    def extract_athletes(df):
        athletes = []
        try:
            for athlete in df["json"]["athletes"]:
                athletes.append(list(athlete.values()))

            return athletes
        except:
            return np.NaN
        
    data["athletes"] = data.apply(extract_athletes, axis = 1)
    data = data.explode("athletes")
    
    # unpack information into corresponding columns
    def unpack(index):
        def func(df):
            try:
                return df["athletes"][index]
            except:
                return np.NaN
        return func
    
    cols = ["highlight_place", "number", "name", "gender", "age", "age_group", "team", "timing", "liquid_timing", "pace (rhythm)"]
    for index, col in enumerate(cols):
        data[col] = data.apply(unpack(index), axis = 1)
        
    # drop unnecessary columns & set numerical index
    data = data.drop(columns = ["json", "athletes"]).reset_index()
    data = data.rename(columns = {"index": "race_name"})

    # change name and race name columns to have title casing
    data["name"] = data["name"].str.title()
    data["race_name"] = data["race_name"].str.title()

    return data