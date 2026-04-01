from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd


from data_loading import load_data, split_data
from data_cleaning import get_preprocessor


df = load_data()
X_train, X_test, y_train, y_test = split_data(df)
preprocess = get_preprocessor()

#simple pipeline
lin_model = Pipeline([
    ("prep", preprocess),
    ("reg", LinearRegression())
])

#train 
lin_model.fit(X_train, y_train)

#predict
pred = lin_model.predict(X_test)

#evaluation
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test,  pred))
r2 = r2_score(y_test,  pred)


#feature importance info 
feature_names = lin_model.named_steps["prep"].get_feature_names_out()
coefficients = lin_model.named_steps["reg"].coef_



print("Linear Regression Results: ")
print("RMSE: ", round(rmse, 4))
print("R2: ", round(r2, 4))

print("MAE: ", round(mae, 4))

#feature importance df
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
}).sort_values(by="coefficient", key=abs, ascending=False)



print(coef_df.head())