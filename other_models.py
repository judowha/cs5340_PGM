import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression


def oversample(train_data):
    total_rows = len(train_data)
    risk_count = (train_data["GOOD_REPUTATION"] == 0).sum()
    diff = risk_count - (total_rows - risk_count)

    if diff > 0:
        risk_0_rows = train_data[train_data['GOOD_REPUTATION'] == 1]
    else:
        risk_0_rows = train_data[train_data['GOOD_REPUTATION'] == 0]
    duplicated_rows = risk_0_rows.sample(n=abs(diff), replace=True)
    train_data = pd.concat([train_data, duplicated_rows], ignore_index=True)
    return train_data, train_data["GOOD_REPUTATION"]

def train_logistic_regression():
    print("train with logistic regression")
    data_df = pd.read_csv("./dataset/preprocessed_data.csv")
    used_features = ["AMT_INCOME_TOTAL", "NAME_EDUCATION_TYPE", "FLAG_OWN_REALTY", "DAYS_EMPLOYED", "CNT_CHILDREN",
                     "NAME_HOUSING_TYPE", "FLAG_OWN_CAR", "DAYS_BIRTH", "GOOD_REPUTATION"]
    data_df = data_df[used_features]
    X_train, X_test, y_train, y_test = train_test_split(data_df, data_df["GOOD_REPUTATION"], test_size=0.2,
                                                        random_state=42)

    model = LogisticRegression()
    x_train, y_train = oversample(X_train)
    model.fit(x_train.drop(columns=["GOOD_REPUTATION"]), y_train)

    predictions = model.predict(X_test.drop(columns=["GOOD_REPUTATION"]))

    accuracy = accuracy_score(y_test, predictions)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"f1: {f1}")


if __name__ == "__main__":
    train_logistic_regression()