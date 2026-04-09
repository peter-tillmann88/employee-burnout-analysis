import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from data_cleaning import get_preprocessor
from data_loading import load_data, split_data


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = get_preprocessor()

    # Preprocessing and model pipeline
    pipeline = Pipeline(
        [("prep", preprocessor), ("gbr", GradientBoostingRegressor(random_state=42))]
    )

    param_grid = {
        "gbr__n_estimators": [100, 300, 500],
        "gbr__learning_rate": [0.01, 0.05, 0.1],
        "gbr__max_depth": [2, 3, 5],
        "gbr__min_samples_split": [2, 5, 10],
        "gbr__min_samples_leaf": [1, 3, 5],
        "gbr__subsample": [0.6, 0.8, 1.0],
    }

    # Hyperparameter tuning with RandomizedSearchCV
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=50,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        random_state=42,
    )

    # Train the model
    random_search.fit(X_train, y_train)

    # Best model evaluation
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Metrics
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("Gradient Boosting Results (Tuned):")
    print("Best Params:", random_search.best_params_)
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))
    print("MAE:", round(mae, 4))

    # Feature Importance
    model = best_model.named_steps["gbr"]
    prep = best_model.named_steps["prep"]

    feature_names = prep.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    print("\nFeature Importances:")
    print(importance_df)


if __name__ == "__main__":
    main()
