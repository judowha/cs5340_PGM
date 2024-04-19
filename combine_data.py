import pandas as pd
import numpy as np
# import pomegranate.bayesian_network
from pgmpy.factors.discrete import DiscreteFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianNetwork, MarkovNetwork, MarkovModel
from pgmpy.estimators import MaximumLikelihoodEstimator, BicScore, BayesianEstimator, MmhcEstimator, \
    ExpectationMaximization, PC
from pgmpy.inference import VariableElimination, BeliefPropagation
from sklearn.metrics import f1_score
from joblib import dump

def combine_data():
    application = pd.read_csv("./dataset/application_record.csv", index_col= None)
    credit = pd.read_csv("./dataset/credit_record.csv", index_col= None)
    join_df = pd.merge(application, credit, on="ID", how='inner')
    columns_to_replace = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY", "FLAG_WORK_PHONE", "FLAG_MOBIL", "FLAG_PHONE", "FLAG_EMAIL"]
    join_df[columns_to_replace] = join_df[columns_to_replace].replace({'Y': 1, 'N': 0})
    indue_status = ["X", "C", "0", ]
    risk = join_df["STATUS"].isin(indue_status)
    risk_value = np.where(risk, 1, 0)
    join_df['GOOD_REPUTATION'] = risk_value
    print(join_df.columns)

    risk_by_id = join_df.groupby('ID')['GOOD_REPUTATION'].transform('min') == 0
    join_df.loc[risk_by_id, 'GOOD_REPUTATION'] = 0
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

    join_df.to_csv("./dataset/combined_data.csv", index=None)


def data_preprocess():
    credit_data = pd.read_csv("./dataset/combined_data.csv")

    columns_to_normalize = ["AMT_INCOME_TOTAL", "DAYS_EMPLOYED", "DAYS_BIRTH"]
    for col in columns_to_normalize:
        credit_data[col] = pd.qcut(credit_data[col], q=10, labels=False, duplicates='drop') 
    
    credit_data["OCCUPATION_TYPE"] = credit_data["OCCUPATION_TYPE"].fillna("UNKOWN")

    income_type_dict = {"Working": 5, "Commercial associate": 3, "Pensioner": 3, "State servant": 5, "Student": 1}
    education_type_dict = {"Higher education": 4, "Secondary / secondary special": 2, "Incomplete higher": 3, "Lower secondary": 1, "Academic degree": 5}
    family_type_dict = {"Civil marriage": 3, "Married": 3, "Single / not married": 1, "Separated": 2, "Widow": 2}
    housing_type_dict = {"House / apartment": 5, "Municipal apartment": 4, "Co-op apartment": 3, "Office apartment": 2, "Rented apartment": 1, "With parents": 0}
    credit_data["NAME_INCOME_TYPE"] = credit_data["NAME_INCOME_TYPE"].replace(income_type_dict)
    credit_data["NAME_EDUCATION_TYPE"] = credit_data["NAME_EDUCATION_TYPE"].replace(education_type_dict)
    credit_data["NAME_FAMILY_STATUS"] = credit_data["NAME_FAMILY_STATUS"].replace(family_type_dict)
    credit_data["NAME_HOUSING_TYPE"] = credit_data["NAME_HOUSING_TYPE"].replace(housing_type_dict)
    credit_data.to_csv("./dataset/preprocessed_data.csv", index=None)

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

def analyze_data():
    credit_data = pd.read_csv("./dataset/preprocessed_data.csv")
    binary_features = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY", "FLAG_WORK_PHONE", "FLAG_MOBIL", "FLAG_PHONE", "FLAG_EMAIL"]
    for feat in binary_features:
        sim = matthews_corrcoef(credit_data[feat], credit_data["GOOD_REPUTATION"])
        print(f"{feat} correlation coefficient {sim}")
    numerical_features = ["CNT_CHILDREN", "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS", "DAYS_EMPLOYED", "DAYS_BIRTH"]
    for feat in numerical_features:
        correlation_coefficient, p_value = pearsonr(credit_data[feat], credit_data["GOOD_REPUTATION"])
        print(f"{feat} correlation coefficient {correlation_coefficient}, p-value {p_value}")
    categorical_features = ["NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"]
    for feat in categorical_features:
        correlation_coefficient, p_value = pearsonr(credit_data[feat], credit_data["GOOD_REPUTATION"])
        print(f"{feat} correlation coefficient {correlation_coefficient}, p-value {p_value}")

def train_customized():
    print("train with MLE")
    data_df = pd.read_csv("./dataset/preprocessed_data.csv")
    # data_df = data_df.drop(columns=["ID", "FLAG_WORK_PHONE", "FLAG_MOBIL", "FLAG_PHONE", "FLAG_EMAIL"])
    print(data_df["GOOD_REPUTATION"].unique())
    X_train, X_test, y_train, y_test = train_test_split(data_df, data_df["GOOD_REPUTATION"], test_size=0.2, random_state=42)
    X_train, y_train = oversample(X_train)
    
    risk_num = (X_train["GOOD_REPUTATION"] == 1).sum()
    print(risk_num / len(X_train))
    used_features = ["NAME_HOUSING_TYPE", "FLAG_OWN_REALTY", "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE", "FLAG_MOBIL", "OCCUPATION_TYPE", "NAME_INCOME_TYPE", "FLAG_WORK_PHONE", "FLAG_OWN_CAR", "DAYS_BIRTH", "AMT_INCOME_TOTAL"]
    # used_features = ["AMT_INCOME_TOTAL", "NAME_EDUCATION_TYPE", "FLAG_OWN_REALTY", "DAYS_EMPLOYED", "CNT_CHILDREN", "NAME_HOUSING_TYPE", "FLAG_OWN_CAR", "NAME_FAMILY_STATUS"]
    model = BayesianNetwork([("NAME_HOUSING_TYPE", "GOOD_REPUTATION"), 
                             ("FLAG_OWN_REALTY", "GOOD_REPUTATION"), 
                             ("NAME_FAMILY_STATUS", "GOOD_REPUTATION"),
                             ("NAME_EDUCATION_TYPE", "GOOD_REPUTATION"),
                             ("FLAG_MOBIL", "GOOD_REPUTATION"),
                             ("OCCUPATION_TYPE", "GOOD_REPUTATION"),
                             ("NAME_INCOME_TYPE", "GOOD_REPUTATION"),
                             ("FLAG_WORK_PHONE", "GOOD_REPUTATION"),
                             ("FLAG_OWN_CAR", "GOOD_REPUTATION"), 
                             ("DAYS_BIRTH", "GOOD_REPUTATION"),
                             ("AMT_INCOME_TOTAL", "FLAG_OWN_CAR"),
                             ("AMT_INCOME_TOTAL", "NAME_HOUSING_TYPE"),
                             ("FLAG_OWN_REALTY", "AMT_INCOME_TOTAL"),
                             ("NAME_EDUCATION_TYPE", "AMT_INCOME_TOTAL"),
                             ("OCCUPATION_TYPE", "NAME_INCOME_TYPE")])
    model.fit(X_train, estimator=BayesianEstimator)
    dump(model, "./BayesianNetwork_MLE.joblib")
    predictions = model.predict(X_test[used_features])
    print(predictions["GOOD_REPUTATION"].unique())
    print(len(y_test))
    print(len(predictions))
    accuracy = accuracy_score(y_test, predictions["GOOD_REPUTATION"].to_numpy())

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions["GOOD_REPUTATION"].to_numpy())
    f1 = f1_score(y_test, predictions["GOOD_REPUTATION"].to_numpy())

    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"f1: {f1}")
    
def search_model(X_train):
    hc = PC(X_train)
    best_model = hc.estimate(scoring_method=BicScore(X_train))
    edges = list(best_model.edges())
    used = []
    for i in edges:
        if i[1] == "GOOD_REPUTATION":
            used.append(i[0])
    print(edges)
    print(used)
    return edges, used
                
    


def train_others():
    print("train with expectation maximization")
    data_df = pd.read_csv("./dataset/oversampling_records.csv")
    data_df = data_df.drop(columns=["ID", "FLAG_WORK_PHONE", "FLAG_MOBIL", "FLAG_PHONE", "FLAG_EMAIL"])
    print(data_df["GOOD_REPUTATION"].unique())
    X_train, X_test, y_train, y_test = train_test_split(data_df, data_df["GOOD_REPUTATION"], test_size=0.1, random_state=42)
    risk_num = (X_train["GOOD_REPUTATION"] == 1).sum()
    print(risk_num / len(X_train))
    used_features = ["AMT_INCOME_TOTAL", "NAME_EDUCATION_TYPE", "FLAG_OWN_REALTY", "DAYS_EMPLOYED", "CNT_CHILDREN",
                     "NAME_HOUSING_TYPE", "FLAG_OWN_CAR"]
    model = MarkovModel([('DAYS_EMPLOYED', 'GOOD_REPUTATION'), ('DAYS_EMPLOYED', 'AMT_INCOME_TOTAL'), ('AMT_INCOME_TOTAL', 'GOOD_REPUTATION'),
                        ("NAME_EDUCATION_TYPE", "GOOD_REPUTATION"), ("FLAG_OWN_REALTY", "GOOD_REPUTATION"), ("CNT_CHILDREN", "GOOD_REPUTATION"),
                        ("NAME_HOUSING_TYPE", "GOOD_REPUTATION"), ("FLAG_OWN_CAR", "GOOD_REPUTATION")])
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

    # factor_daysEmp_risk = DiscreteFactor(['DAYS_EMPLOYED', 'GOOD_REPUTATION'], cardinality=[10, 2], values=np.array(ten_to_two_array))
    # factor_daysEmp_income = DiscreteFactor(['DAYS_EMPLOYED', 'AMT_INCOME_TOTAL'], cardinality=[10, 10], values=np.array(ten_to_ten_array))
    # factor_income_risk = DiscreteFactor(['AMT_INCOME_TOTAL', 'GOOD_REPUTATION'], cardinality=[10, 2], values=np.array(ten_to_two_array))
    # factor_education_risk = DiscreteFactor(["NAME_EDUCATION_TYPE", "GOOD_REPUTATION"], cardinality=[5, 2], values=np.array(five_to_two_array))
    # factor_realty_risk = DiscreteFactor(["FLAG_OWN_REALTY", "GOOD_REPUTATION"], cardinality=[2, 2], values=np.array(two_to_two_array))
    # factor_children_risk = DiscreteFactor(["CNT_CHILDREN", "GOOD_REPUTATION"], cardinality=[10, 2], values=np.array(ten_to_two_array))
    # factor_housing_risk = DiscreteFactor(["NAME_HOUSING_TYPE", "GOOD_REPUTATION"], cardinality=[6, 2], values=np.array(six_to_two_array))
    # factor_car_risk = DiscreteFactor(["FLAG_OWN_CAR", "GOOD_REPUTATION"], cardinality=[2, 2], values=np.array(two_to_two_array))
    # model.add_factors(factor_daysEmp_risk, factor_car_risk, factor_housing_risk, factor_daysEmp_income, factor_children_risk, factor_realty_risk, factor_income_risk, factor_education_risk)
    mle = MaximumLikelihoodEstimator(model, X_train)

    factors = []
    for node, cpt in mle.get_parameters().items():
        cardinality = [len(X_train[node].unique())]
        factor = DiscreteFactor([node], cardinality, values=cpt.values)
        factors.append(factor)

    model.add_factors(*factors)

    inference = VariableElimination(model)

    prediction = inference.map_query(variables=["GOOD_REPUTATION"], evidence={"DAYS_EMPLOYED": X_test["DAYS_EMPLOYED"], "AMT_INCOME_TOTAL": X_test["AMT_INCOME_TOTAL"],
                                                                   "NAME_EDUCATION_TYPE": X_test["NAME_EDUCATION_TYPE"], "FLAG_OWN_REALTY": X_test["FLAG_OWN_REALTY"],
                                                                   "CNT_CHILDREN": X_test["CNT_CHILDREN"], "NAME_HOUSING_TYPE": X_test["NAME_HOUSING_TYPE"],
                                                                   "FLAG_OWN_CAR": X_test["FLAG_OWN_CAR"]})
    accuracy = accuracy_score(y_test, prediction['GOOD_REPUTATION'])
    conf_matrix = confusion_matrix(y_test, prediction['GOOD_REPUTATION'])

    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(conf_matrix)

def train():
    print("train with MLE")
    data_df = pd.read_csv("./dataset/preprocessed_data.csv")
    data_df = data_df.drop(columns=["ID", "FLAG_WORK_PHONE", "FLAG_MOBIL", "FLAG_PHONE", "FLAG_EMAIL"])
    print(data_df["GOOD_REPUTATION"].unique())
    X_train, X_test, y_train, y_test = train_test_split(data_df, data_df["GOOD_REPUTATION"], test_size=0.2, random_state=42)
    X_train, y_train = oversample(X_train)
    
    risk_num = (X_train["GOOD_REPUTATION"] == 1).sum()
    print(risk_num / len(X_train))
    # X_train, _ = oversample(X_train) # to be commented
    used_features = ["AMT_INCOME_TOTAL", "NAME_EDUCATION_TYPE", "FLAG_OWN_REALTY", "DAYS_EMPLOYED", "CNT_CHILDREN", "NAME_HOUSING_TYPE", "FLAG_OWN_CAR"]
    model = BayesianNetwork([('DAYS_EMPLOYED', 'GOOD_REPUTATION'), ('DAYS_EMPLOYED', 'AMT_INCOME_TOTAL'), ('AMT_INCOME_TOTAL', 'GOOD_REPUTATION'), 
                             ("NAME_EDUCATION_TYPE", "GOOD_REPUTATION"),("FLAG_OWN_REALTY", "GOOD_REPUTATION"), ("CNT_CHILDREN", "GOOD_REPUTATION"),
                             ("NAME_HOUSING_TYPE", "GOOD_REPUTATION"), ("FLAG_OWN_CAR", "GOOD_REPUTATION"), ("DAYS_BIRTH", "GOOD_REPUTATION")])
    model.fit(X_train)
    infer = BeliefPropagation(model)
    test_dict = X_test[used_features].to_dict("records")
    result = []
    for i in test_dict:
        query_result = infer.query(variables=['GOOD_REPUTATION'], evidence=i)
        result.append(np.argmax(query_result.values))
    conf_matrix = confusion_matrix(y_test, result, labels=[0, 1])
    f1 = f1_score(y_test, result)

    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"f1: {f1}")

def train_BP():
    print("train with MLE")
    data_df = pd.read_csv("./dataset/preprocessed_data.csv")
    data_df = data_df.drop(columns=["ID", "FLAG_WORK_PHONE", "FLAG_MOBIL", "FLAG_PHONE", "FLAG_EMAIL"])
    print(data_df["GOOD_REPUTATION"].unique())
    X_train, X_test, y_train, y_test = train_test_split(data_df, data_df["GOOD_REPUTATION"], test_size=0.2, random_state=42)
    print(len(X_train))
    X_train, y_train = oversample(X_train)
    
    risk_num = (X_train["GOOD_REPUTATION"] == 1).sum()
    print(risk_num / len(X_train))
    # X_train, _ = oversample(X_train) # to be commented
    used_features = ["AMT_INCOME_TOTAL", "NAME_EDUCATION_TYPE", "FLAG_OWN_REALTY", "DAYS_EMPLOYED", "CNT_CHILDREN", "NAME_HOUSING_TYPE", "FLAG_OWN_CAR"]
    model = BayesianNetwork([('DAYS_EMPLOYED', 'GOOD_REPUTATION'), ('DAYS_EMPLOYED', 'AMT_INCOME_TOTAL'), ('AMT_INCOME_TOTAL', 'GOOD_REPUTATION'), 
                             ("NAME_EDUCATION_TYPE", "GOOD_REPUTATION"),("FLAG_OWN_REALTY", "GOOD_REPUTATION"), ("CNT_CHILDREN", "GOOD_REPUTATION"),
                             ("NAME_HOUSING_TYPE", "GOOD_REPUTATION"), ("FLAG_OWN_CAR", "GOOD_REPUTATION"), ("DAYS_BIRTH", "GOOD_REPUTATION")])
    model.fit(X_train, estimator=BayesianEstimator)
    infer = BeliefPropagation(model)
    test_dict = X_test[used_features].to_dict("records")
    result = []
    for i in test_dict:
        query_result = infer.query(variables=['GOOD_REPUTATION'], evidence=i)
        result.append(np.argmax(query_result.values))
    conf_matrix = confusion_matrix(y_test, result, labels=[0, 1])
    f1 = f1_score(y_test, result)

    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"f1: {f1}")

if __name__ == "__main__":
    # combine_data()
    # data_preprocess()
    # analyze_data()
    # train_model()
    # train_customized()
    # train_others()
    # train()
    train_BP()
