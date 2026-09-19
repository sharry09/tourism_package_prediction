import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = "tourism_project/data/tourism.csv"
df = pd.read_csv(DATASET_PATH, on_bad_lines="skip")

df.drop(columns=["CustomerID"], inplace=True)

categorical_cols = ["TypeofContact","Occupation","Gender","MaritalStatus","Designation","ProductPitched"]
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

target_col = "ProdTaken"
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
