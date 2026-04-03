from sklearn.ensemble import RandomForestRegressor
from data_loading import load_data, split_data
from data_cleaning import get_preprocessor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error,  mean_absolute_error
import pandas as pd

def random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return rf, y_test, y_pred

def feature_importance(rf, preprocessor):
    feature_names = preprocessor.get_feature_names_out()
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': rf.feature_importances_})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("Feature Importances:")
    print(importance_df)

def plot_results(y_test, y_pred):
    plt.scatter(y_test, y_pred, color='purple')
    plt.xlabel('Actual Burn Rate')
    plt.ylabel('Predicted Burn Rate')
    plt.title("Random Forest: Actual vs Predicted")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.show()

def metrics(y_test, y_pred, rf):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print("Random Forest Results:")
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))
    print("MSE:", round(mse, 4))
    print("OOB:", round(rf.oob_score_, 4))
    print("MAE:", round(mae, 4))


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = get_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    rf, y_test, y_pred = random_forest(X_train_processed, y_train, X_test_processed, y_test)
    metrics(y_test, y_pred, rf)
    feature_importance(rf, preprocessor)
    #plot_results(y_test, y_pred)

if __name__ == "__main__":
    main()
