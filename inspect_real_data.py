import pandas as pd


DATA_FILE = "data/real/creditcard.csv"


df = pd.read_csv(DATA_FILE)


print("=== DATASET OVERVIEW ===")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nClass distribution:")
print(df["Class"].value_counts())

print("\nClass percentages:")
print(
    df["Class"]
    .value_counts(normalize=True)
    .mul(100)
)


print("\nAmount statistics:")
print(df["Amount"].describe())


print("\nFirst 5 transactions:")
print(df.head())


print("\nFraud transactions:")
print(
    df[df["Class"] == 1].head()
)