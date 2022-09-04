import pandas as pd

path = r"dataset/"
file_name2 = "ALL_DATA_0713_SINGLE_FILE.json"

df2 = pd.read_json(path + file_name2, lines=True)
for index, row in df2.iterrows():
    url = df2.at[index, 'url']
    if url.endswith(','):
        df2.at[index, 'url'] = url[:len(url) - 1]

df2.to_json(path + "All2.json", lines=True, orient='records')
