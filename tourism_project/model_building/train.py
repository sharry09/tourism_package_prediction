import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import mlflow
import joblib
import os

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").values.ravel()
ytest = pd.read_csv("ytest.csv").values.ravel()

numeric_features = ["Age","NumberOfPersonVisiting","PreferredPropertyStar","NumberOfTrips",
                    "MonthlyIncome","PitchSatisfactionScore","NumberOfFollowups","DurationOfPitch"]
categorical_features = ["TypeofContact","CityTier","Occupation","Gender","MaritalStatus",
                        "Passport","OwnCar","NumberOfChildrenVisiting","Designation","ProductPitched"]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)

param_grid = {
    "xgbclassifier__n_estimators":[50,100],
    "xgbclassifier__max_depth":[3,5],
    "xgbclassifier__learning_rate":[0.01,0.1],
    "xgbclassifier__subsample":[0.8,1.0],
    "xgbclassifier__colsample_bytree":[0.8,1.0]
}

pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    grid.fit(Xtrain, ytrain)

    best_model = grid.best_estimator_
    ypred = best_model.predict(Xtest)

    acc = accuracy_score(ytest, ypred)
    f1 = f1_score(ytest, ypred)

    mlflow.log_params(grid.best_params_)
    mlflow.log_metrics({"accuracy":acc,"f1_score":f1})

    print("Best Params:", grid.best_params_)
    print("Accuracy:", acc)
    print("F1 Score:", f1)

    os.makedirs("tourism_project/deployment", exist_ok=True)
    joblib.dump(best_model, "tourism_project/deployment/best_model.pkl")
    print("Best model saved to tourism_project/deployment/best_model.pkl")
