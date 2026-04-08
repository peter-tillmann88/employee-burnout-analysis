import pandas as pd
from sklearn.model_selection import train_test_split

Data_p = "../data/train.csv"

def load_data():

    df = pd.read_csv(Data_p)

    #clean up naming 
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    #drop uselsss featrues
    df = df.drop(columns=["employee_id"], errors="ignore")

    df = df.dropna(subset=["burn_rate"])
    return df


def split_data(df):
    X = df.drop("burn_rate", axis = 1)
    y = df["burn_rate"]

    X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data()
    
    print("Number of instances: ", df.shape[0])
    print("Number of features :",df.shape[1])