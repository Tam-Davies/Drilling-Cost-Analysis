import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv(
    "data/processed/drilling_cost_clean.csv"
)

X = sm.add_constant(df["Depth"])
y = df["Cost"]

models = {
    "Direct": sm.OLS(y, X).fit(),
    "Log": sm.OLS(np.log(y), X).fit()
}

diagnostics = []

for name, model in models.items():

    bp = het_breuschpagan(
        model.resid,
        model.model.exog
    )

    sw = shapiro(model.resid)

    diagnostics.append({
        "Model": name,
        "Breusch_Pagan_p": bp[1],
        "Shapiro_p": sw.pvalue
    })

pd.DataFrame(diagnostics).to_csv(
    "outputs/tables/formal_diagnostics.csv",
    index=False
)

# LOOCV
loo = LeaveOneOut()
cv_results = []

for name in models:

    actual = []
    predicted = []

    for train, test in loo.split(df):

        train_df = df.iloc[train]
        test_df = df.iloc[test]

        X_train = sm.add_constant(
            train_df["Depth"]
        )
        X_test = sm.add_constant(
            test_df["Depth"],
            has_constant="add"
        )

        if name == "Direct":

            model = sm.OLS(
                train_df["Cost"],
                X_train
            ).fit()

            pred = model.predict(X_test).iloc[0]

        else:

            model = sm.OLS(
                np.log(train_df["Cost"]),
                X_train
            ).fit()

            pred = np.exp(
                model.predict(X_test).iloc[0]
            )

        actual.append(test_df["Cost"].iloc[0])
        predicted.append(pred)

    actual = np.array(actual)
    predicted = np.array(predicted)

    cv_results.append({
        "Model": name,
        "MAE": mean_absolute_error(
            actual, predicted
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                actual, predicted
            )
        ),
        "MAPE_%": np.mean(
            np.abs(
                (actual - predicted) / actual
            )
        ) * 100
    })

pd.DataFrame(cv_results).to_csv(
    "outputs/tables/loocv_results.csv",
    index=False
)

print("Diagnostics complete.")