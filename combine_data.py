import pandas as pd
import numpy as np
import pomegranate.bayesian_network
from pgmpy.factors.discrete import DiscreteFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score, f1_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianNetwork, MarkovNetwork, MarkovModel
from pgmpy.estimators import MaximumLikelihoodEstimator, BicScore, BayesianEstimator, MmhcEstimator, ExpectationMaximization
from pgmpy.inference import VariableElimination, BeliefPropagation
from joblib import dump

def combine_data():
    application = pd.read_csv("./dataset/application_record.csv")
    credit = pd.read_csv("./dataset/credit_record.csv")
    join_df = pd.merge(application, credit, on="ID", how='inner')
    columns_to_replace = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY", "FLAG_WORK_PHONE", "FLAG_MOBIL", "FLAG_PHONE", "FLAG_EMAIL"]
    join_df[columns_to_replace] = join_df[columns_to_replace].replace({'Y': 1, 'N': 0})
    outdue_status = ["1", "2", "3", "4", "5"]
    not_safe = join_df["STATUS"].isin(outdue_status)
    safe_value = np.where(not_safe, 0, 1)
    join_df['SAFE'] = safe_value
    print(join_df.columns)

    safe_by_id = join_df.groupby('ID')['SAFE'].transform('max') == 0
    join_df.loc[safe_by_id, 'SAFE'] = 0
    join_df = join_df.drop_duplicates(subset='ID', keep='first')
    print(join_df.columns)

    unique_income_types = join_df["NAME_INCOME_TYPE"].unique()
    unique_education_types = join_df["NAME_EDUCATION_TYPE"].unique()
    unique_family_status = join_df["NAME_FAMILY_STATUS"].unique()
    unique_housing_types = join_df["NAME_HOUSING_TYPE"].unique()
    unique_status_types = join_df["STATUS"].unique()

    print("unique income types: ", unique_income_types)
    print("unique education types: ", unique_education_types)
    print("unique family status: ", unique_family_status)
    print("unique housing types: ", unique_housing_types)
    print("unique status types: ", unique_status_types)

    join_df.to_csv("./dataset/complete_credit_record.csv")


def data_preprocess():
    credit_data = pd.read_csv("./dataset/complete_credit_record.csv")

    columns_to_normalize = ["AMT_INCOME_TOTAL", "DAYS_EMPLOYED", "DAYS_BIRTH"]
    # scaler = StandardScaler()
    # credit_data[columns_to_normalize] = scaler.fit_transform(credit_data[columns_to_normalize])
    for col in columns_to_normalize:
        credit_data[col] = pd.qcut(credit_data[col], q=10, labels=False, duplicates='drop') 

    income_type_dict = {"Working": 5, "Commercial associate": 3, "Pensioner": 3, "State servant": 5, "Student": 1}
    education_type_dict = {"Higher education": 4, "Secondary / secondary special": 2, "Incomplete higher": 3, "Lower secondary": 1, "Academic degree": 5}
    family_type_dict = {"Civil marriage": 3, "Married": 3, "Single / not married": 1, "Separated": 2, "Widow": 2}
    housing_type_dict = {"House / apartment": 5, "Municipal apartment": 4, "Co-op apartment": 3, "Office apartment": 2, "Rented apartment": 1, "With parents": 0}
    credit_data["NAME_INCOME_TYPE"] = credit_data["NAME_INCOME_TYPE"].replace(income_type_dict)
    credit_data["NAME_EDUCATION_TYPE"] = credit_data["NAME_EDUCATION_TYPE"].replace(education_type_dict)
    credit_data["NAME_FAMILY_STATUS"] = credit_data["NAME_FAMILY_STATUS"].replace(family_type_dict)
    credit_data["NAME_HOUSING_TYPE"] = credit_data["NAME_HOUSING_TYPE"].replace(housing_type_dict)

    Train_data, Test_data, _, _ = train_test_split(credit_data, credit_data["SAFE"], test_size=0.2, random_state=42)

    total_rows = len(Train_data)
    safe_count = (Train_data["SAFE"] == 1).sum()
    diff = safe_count - (total_rows - safe_count)
    if diff > 0:
        safe_0_rows = Train_data[Train_data['SAFE'] == 0]
        duplicated_rows = safe_0_rows.sample(n=diff, replace=True)
        Train_data = pd.concat([Train_data, duplicated_rows], ignore_index=True)
    if diff < 0:
        safe_1_rows = Train_data[Train_data['SAFE'] == 1]
        duplicated_rows = safe_1_rows.sample(n=-diff, replace=True)
        Train_data = pd.concat([Train_data, duplicated_rows], ignore_index=True)

    Train_data.to_csv("./dataset/train.csv")
    Test_data.to_csv("./dataset/test.csv")


def analyze_data():
    credit_data = pd.read_csv("./dataset/oversampling_records.csv")
    binary_features = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY", "FLAG_WORK_PHONE", "FLAG_MOBIL", "FLAG_PHONE", "FLAG_EMAIL"]
    for feat in binary_features:
        sim = matthews_corrcoef(credit_data[feat], credit_data["RISK"])
        print(f"{feat} correlation coefficient {sim}")
    numerical_features = ["CNT_CHILDREN", "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS", "DAYS_EMPLOYED", "DAYS_BIRTH"]
    for feat in numerical_features:
        correlation_coefficient, p_value = pearsonr(credit_data[feat], credit_data["RISK"])
        print(f"{feat} correlation coefficient {correlation_coefficient}, p-value {p_value}")
    categorical_features = ["NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"]
    for feat in categorical_features:
        correlation_coefficient, p_value = pearsonr(credit_data[feat], credit_data["RISK"])
        print(f"{feat} correlation coefficient {correlation_coefficient}, p-value {p_value}")


def train_model():
    data_df = pd.read_csv("./dataset/oversampling_records.csv")
    data_df = data_df.drop(columns=["ID", "FLAG_WORK_PHONE", "FLAG_MOBIL", "FLAG_PHONE", "FLAG_EMAIL"])
    X_train, X_test, y_train, y_test = train_test_split(data_df, data_df["RISK"], test_size=0.2, random_state=42)
    
    hc = MaximumLikelihoodEstimator(X_train)
    best_model = hc.estimate(scoring_method=BicScore(X_train))
    edges = list(best_model.edges())
    model = BayesianNetwork(edges)
        
    # model = MarkovNetwork([('AMT_INCOME_TOTAL', 'RISK'),('AMT_INCOME_TOTAL', 'NAME_HOUSING_TYPE'), ('NAME_EDUCATION_TYPE', 'RISK'), 
    #                          ('NAME_EDUCATION_TYPE', 'AMT_INCOME_TOTAL'),("FLAG_OWN_CAR", "RISK"),
    #                          ("DAYS_EMPLOYED", "AMT_INCOME_TOTAL"), ("NAME_INCOME_TYPE", "AMT_INCOME_TOTAL"),
    #                          ("NAME_HOUSING_TYPE", "RISK"), ("CNT_CHILDREN", "RISK"), ("FLAG_OWN_REALTY", "RISK"), ])
    model.fit(X_train, estimator=BayesianEstimator)
    # Perform inference
    infer = VariableElimination(model)
    test_dict = X_test[["AMT_INCOME_TOTAL", "NAME_EDUCATION_TYPE", "FLAG_OWN_CAR", "DAYS_EMPLOYED", "NAME_INCOME_TYPE",
                        "NAME_HOUSING_TYPE", "CNT_CHILDREN", "FLAG_OWN_REALTY"]].to_dict("records")
    result = []
    for i in test_dict:
        query_result = infer.query(variables=['RISK'], evidence=i)
        result.append(np.argmax(query_result.values))
    matrix = confusion_matrix(y_test, result, labels=[0, 1])
    print(str(matrix))

def train_customized():
    print("train with MLE")
    data_df = pd.read_csv("./dataset/oversampling_records.csv")
    data_df = data_df.drop(columns=["ID", "FLAG_WORK_PHONE", "FLAG_MOBIL", "FLAG_PHONE", "FLAG_EMAIL"])
    print(data_df["RISK"].unique())
    X_train, X_test, y_train, y_test = train_test_split(data_df, data_df["RISK"], test_size=0.1, random_state=42)
    risk_num = (X_train["RISK"] == 1).sum()
    print(risk_num / len(X_train))
    # used_features = ['DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', "NAME_EDUCATION_TYPE", "FLAG_OWN_REALTY"]
    used_features = ["AMT_INCOME_TOTAL", "NAME_EDUCATION_TYPE", "FLAG_OWN_REALTY", "DAYS_EMPLOYED", "CNT_CHILDREN",
                     "NAME_HOUSING_TYPE", "FLAG_OWN_CAR", "DAYS_BIRTH"]
    # model = BayesianNetwork([('DAYS_EMPLOYED', 'AMT_INCOME_TOTAL'), ('NAME_INCOME_TYPE', 'AMT_INCOME_TOTAL'),
    #                          ('AMT_INCOME_TOTAL', 'RISK'), ("NAME_EDUCATION_TYPE", "RISK"), ("FLAG_OWN_REALTY", "RISK")])
    # estimator = MaximumLikelihoodEstimator(model, X_train)
    model = BayesianNetwork([('DAYS_EMPLOYED', 'RISK'), ('DAYS_EMPLOYED', 'AMT_INCOME_TOTAL'), ('AMT_INCOME_TOTAL', 'RISK'), 
                             ("NAME_EDUCATION_TYPE", "RISK"),("FLAG_OWN_REALTY", "RISK"), ("CNT_CHILDREN", "RISK"), 
                             ("NAME_HOUSING_TYPE", "RISK"), ("FLAG_OWN_CAR", "RISK"), ("DAYS_BIRTH", "RISK")])
    model.fit(X_train, estimator=MaximumLikelihoodEstimator)
    dump(model, "./BayesianNetwork_MLE.joblib")
    predictions = model.predict(X_test[used_features])
    print(predictions["RISK"].unique())
    print(len(y_test))
    print(len(predictions))
    accuracy = accuracy_score(y_test, predictions["RISK"].to_numpy())

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions["RISK"].to_numpy())

    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(conf_matrix)


def train_others():
    print("train with expectation maximization")
    data_df = pd.read_csv("./dataset/oversampling_records.csv")
    data_df = data_df.drop(columns=["ID", "FLAG_WORK_PHONE", "FLAG_MOBIL", "FLAG_PHONE", "FLAG_EMAIL"])
    print(data_df["RISK"].unique())
    X_train, X_test, y_train, y_test = train_test_split(data_df, data_df["RISK"], test_size=0.1, random_state=42)
    risk_num = (X_train["RISK"] == 1).sum()
    print(risk_num / len(X_train))
    used_features = ["AMT_INCOME_TOTAL", "NAME_EDUCATION_TYPE", "FLAG_OWN_REALTY", "DAYS_EMPLOYED", "CNT_CHILDREN",
                     "NAME_HOUSING_TYPE", "FLAG_OWN_CAR"]
    model = MarkovModel([('DAYS_EMPLOYED', 'RISK'), ('DAYS_EMPLOYED', 'AMT_INCOME_TOTAL'), ('AMT_INCOME_TOTAL', 'RISK'),
                        ("NAME_EDUCATION_TYPE", "RISK"), ("FLAG_OWN_REALTY", "RISK"), ("CNT_CHILDREN", "RISK"),
                        ("NAME_HOUSING_TYPE", "RISK"), ("FLAG_OWN_CAR", "RISK")])
    # ten_to_two_array = []
    # for i in range(10):
    #     for j in range(2):
    #         ten_to_two_array.append([i, j])
    #
    # ten_to_ten_array = []
    # for i in range(10):
    #     for j in range(10):
    #         ten_to_two_array.append([i, j])
    #
    # six_to_two_array = []
    # for i in range(6):
    #     for j in range(2):
    #         six_to_two_array.append([i, j])
    #
    # five_to_two_array = []
    # for i in range(1, 6):
    #     for j in range(2):
    #         five_to_two_array.append([i, j])
    #
    # two_to_two_array = []
    # for i in range(2):
    #     for j in range(2):
    #         two_to_two_array.append([i, j])

    # factor_daysEmp_risk = DiscreteFactor(['DAYS_EMPLOYED', 'RISK'], cardinality=[10, 2], values=np.array(ten_to_two_array))
    # factor_daysEmp_income = DiscreteFactor(['DAYS_EMPLOYED', 'AMT_INCOME_TOTAL'], cardinality=[10, 10], values=np.array(ten_to_ten_array))
    # factor_income_risk = DiscreteFactor(['AMT_INCOME_TOTAL', 'RISK'], cardinality=[10, 2], values=np.array(ten_to_two_array))
    # factor_education_risk = DiscreteFactor(["NAME_EDUCATION_TYPE", "RISK"], cardinality=[5, 2], values=np.array(five_to_two_array))
    # factor_realty_risk = DiscreteFactor(["FLAG_OWN_REALTY", "RISK"], cardinality=[2, 2], values=np.array(two_to_two_array))
    # factor_children_risk = DiscreteFactor(["CNT_CHILDREN", "RISK"], cardinality=[10, 2], values=np.array(ten_to_two_array))
    # factor_housing_risk = DiscreteFactor(["NAME_HOUSING_TYPE", "RISK"], cardinality=[6, 2], values=np.array(six_to_two_array))
    # factor_car_risk = DiscreteFactor(["FLAG_OWN_CAR", "RISK"], cardinality=[2, 2], values=np.array(two_to_two_array))
    # model.add_factors(factor_daysEmp_risk, factor_car_risk, factor_housing_risk, factor_daysEmp_income, factor_children_risk, factor_realty_risk, factor_income_risk, factor_education_risk)
    mle = MaximumLikelihoodEstimator(model, X_train)

    factors = []
    for node, cpt in mle.get_parameters().items():
        cardinality = [len(X_train[node].unique())]
        factor = DiscreteFactor([node], cardinality, values=cpt.values)
        factors.append(factor)

    model.add_factors(*factors)

    inference = VariableElimination(model)

    prediction = inference.map_query(variables=["RISK"], evidence={"DAYS_EMPLOYED": X_test["DAYS_EMPLOYED"], "AMT_INCOME_TOTAL": X_test["AMT_INCOME_TOTAL"],
                                                                   "NAME_EDUCATION_TYPE": X_test["NAME_EDUCATION_TYPE"], "FLAG_OWN_REALTY": X_test["FLAG_OWN_REALTY"],
                                                                   "CNT_CHILDREN": X_test["CNT_CHILDREN"], "NAME_HOUSING_TYPE": X_test["NAME_HOUSING_TYPE"],
                                                                   "FLAG_OWN_CAR": X_test["FLAG_OWN_CAR"]})
    accuracy = accuracy_score(y_test, prediction['RISK'])
    conf_matrix = confusion_matrix(y_test, prediction['RISK'])

    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(conf_matrix)


def train_new():
    print("train with MLE")
    train_df = pd.read_csv("./dataset/train.csv")
    # used_features = ['DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', "NAME_EDUCATION_TYPE", "FLAG_OWN_REALTY"]
    used_features = ["AMT_INCOME_TOTAL", "NAME_EDUCATION_TYPE", "FLAG_OWN_REALTY", "DAYS_EMPLOYED", "CNT_CHILDREN",
                     "NAME_HOUSING_TYPE", "FLAG_OWN_CAR", "DAYS_BIRTH"]
    # model = BayesianNetwork([('DAYS_EMPLOYED', 'AMT_INCOME_TOTAL'), ('NAME_INCOME_TYPE', 'AMT_INCOME_TOTAL'),
    #                          ('AMT_INCOME_TOTAL', 'RISK'), ("NAME_EDUCATION_TYPE", "RISK"), ("FLAG_OWN_REALTY", "RISK")])
    # estimator = MaximumLikelihoodEstimator(model, X_train)
    model = BayesianNetwork([('DAYS_EMPLOYED', 'SAFE'), ('DAYS_EMPLOYED', 'AMT_INCOME_TOTAL'), ('AMT_INCOME_TOTAL', 'SAFE'),
                             ("NAME_EDUCATION_TYPE", "SAFE"), ("FLAG_OWN_REALTY", "SAFE"), ("CNT_CHILDREN", "SAFE"),
                             ("NAME_HOUSING_TYPE", "SAFE"), ("FLAG_OWN_CAR", "SAFE"), ("DAYS_BIRTH", "SAFE")])

    train_features = ["AMT_INCOME_TOTAL", "NAME_EDUCATION_TYPE", "FLAG_OWN_REALTY", "DAYS_EMPLOYED", "CNT_CHILDREN",
                     "NAME_HOUSING_TYPE", "FLAG_OWN_CAR", "DAYS_BIRTH", "SAFE"]
    model.fit(train_df[train_features], estimator=MaximumLikelihoodEstimator)

    dump(model, "./BayesianNetwork_MLE_new.joblib")
    test_df = pd.read_csv("./dataset/test.csv")

    predictions = model.predict(test_df[used_features])
    y_test = test_df["SAFE"]
    accuracy = accuracy_score(y_test, predictions["SAFE"].to_numpy())
    f1 = f1_score(y_test, predictions["SAFE"].to_numpy(), average='binary')

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions["SAFE"].to_numpy())

    print("Accuracy:", accuracy)
    print("f1 score", f1)
    print("Confusion Matrix:")
    print(conf_matrix)

    possibilities = model.predict_probability(test_df[used_features])
    possibilities.to_csv("./dataset/possibilities.csv")


def try_thresholds():
    test_df = pd.read_csv("./dataset/test.csv")
    y_test = test_df["SAFE"]
    possibility = pd.read_csv("./dataset/possibilities.csv")
    possibility['SAFE'] = possibility['SAFE_0'].apply(lambda x: 0 if x > 0.0 else 1)
    accuracy = accuracy_score(y_test, possibility["SAFE"].to_numpy())

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, possibility["SAFE"].to_numpy())
    f1 = f1_score(y_test, possibility["SAFE"].to_numpy(), average='binary')

    print("Accuracy:", accuracy)
    print("f1 score", f1)
    print("Confusion Matrix:")
    print(conf_matrix)


if __name__ == "__main__":
    combine_data()
    data_preprocess()
    # analyze_data()
    # train_model()
    # train_customized()
    # train_others()
    train_new()
    # try_thresholds()
