import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression


def train_logistic_regression():
    print("train with logistic regression")
    data_df = pd.read_csv("./dataset/oversampling_records.csv")
    #data_df = data_df.drop(columns=["ID", "FLAG_WORK_PHONE", "FLAG_MOBIL", "FLAG_PHONE", "FLAG_EMAIL", "STATUS", "CODE_GENDER", "Unnamed: 0", "OCCUPATION_TYPE"])
    used_features = ["AMT_INCOME_TOTAL", "NAME_EDUCATION_TYPE", "FLAG_OWN_REALTY", "DAYS_EMPLOYED", "CNT_CHILDREN",
                     "NAME_HOUSING_TYPE", "FLAG_OWN_CAR", "RISK"]
    data_df = data_df[used_features]
    X_train, X_test, y_train, y_test = train_test_split(data_df.drop(columns=["RISK"]), data_df["RISK"], test_size=0.2,
                                                        random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(conf_matrix)


if __name__ == "__main__":
    train_logistic_regression()