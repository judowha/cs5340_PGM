import pandas as pd
from sklearn.model_selection import train_test_split
from pgmpy.estimators import TreeSearch
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianNetwork
from sklearn.metrics import confusion_matrix, accuracy_score


def train_others():
    print("train with expectation maximization")
    data_df = pd.read_csv("./dataset/oversampling_records.csv")
    data_df.dropna(inplace=True)
    data_df = data_df.drop(columns=["ID", "FLAG_WORK_PHONE", "FLAG_MOBIL", "FLAG_PHONE", "FLAG_EMAIL"])
    print(data_df["RISK"].unique())
    X_train, X_test, y_train, y_test = train_test_split(data_df, data_df["RISK"], test_size=0.1, random_state=42)
    risk_num = (X_train["RISK"] == 1).sum()
    print(risk_num / len(X_train))
    used_features = ["AMT_INCOME_TOTAL", "NAME_EDUCATION_TYPE", "FLAG_OWN_REALTY", "DAYS_EMPLOYED", "CNT_CHILDREN",
                     "NAME_HOUSING_TYPE", "FLAG_OWN_CAR"]
    # model = MarkovModel([('DAYS_EMPLOYED', 'RISK'), ('DAYS_EMPLOYED', 'AMT_INCOME_TOTAL'), ('AMT_INCOME_TOTAL', 'RISK'),
    #                     ("NAME_EDUCATION_TYPE", "RISK"), ("FLAG_OWN_REALTY", "RISK"), ("CNT_CHILDREN", "RISK"),
    #                     ("NAME_HOUSING_TYPE", "RISK"), ("FLAG_OWN_CAR", "RISK")])

    est = TreeSearch(X_train, root_node="AMT_INCOME_TOTAL")
    dag = est.estimate()
    model = BayesianNetwork(dag)
    model.fit(X_train)
    bp = BeliefPropagation(model)
    bp.calibrate()
    predictions = model.predict(X_test[used_features])
    accuracy = accuracy_score(y_test, predictions["RISK"].to_numpy())

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions["RISK"].to_numpy())

    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(conf_matrix)

train_others()