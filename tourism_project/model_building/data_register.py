import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

df = pd.read_csv(RAW_PATH, on_bad_lines="skip")

expected_columns = [
    "CustomerID","ProdTaken","Age","TypeofContact","CityTier","Occupation","Gender",
    "NumberOfPersonVisiting","PreferredPropertyStar","MaritalStatus","NumberOfTrips",
    "Passport","OwnCar","NumberOfChildrenVisiting","Designation","MonthlyIncome",
    "PitchSatisfactionScore","ProductPitched","NumberOfFollowups","DurationOfPitch"
]

missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset missing columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Target column preview (ProdTaken):")
print(df["ProdTaken"].value_counts())
