import pandas as pd

df = pd.read_csv("data/games.csv")
print("Columns:", df.columns.tolist())
print("\nFirst row:")
print(df.iloc[0])
