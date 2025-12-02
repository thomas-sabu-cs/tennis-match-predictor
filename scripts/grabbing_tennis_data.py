import pandas as pd
import requests
from io import StringIO

years = [2020, 2021, 2022, 2023, 2024]
dfs = []

for year in years:
    url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
    df = pd.read_csv(url)
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined.to_csv("atp_matches_2020_2024.csv", index=False)
