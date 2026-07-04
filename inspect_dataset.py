import pandas as pd

df = pd.read_csv(
    'data/paysim_data.csv'
)

# print(df.head())

# print("\nColumns:\n")
# print(df.columns)

# print("\nDataset Shape:\n")
# print(df.shape)
# print(df.isnull().sum())

# print(df.info())
# print(df['isFraud'].value_counts())

print(df["isFraud"].value_counts())

print(df["type"].value_counts())

print(
    pd.crosstab(
        df["type"],
        df["isFraud"]
    )
)