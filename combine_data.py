import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianNetwork, MarkovNetwork
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator, BicScore, BayesianEstimator, PC
from pgmpy.inference import VariableElimination, BeliefPropagation

def combine_data():
    application = pd.read_csv("./dataset/application_record.csv")
    credit = pd.read_csv("./dataset/credit_record.csv")
    join_df = pd.merge(application, credit, on="ID", how='inner')
    columns_to_replace = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY", "FLAG_WORK_PHONE", "FLAG_MOBIL", "FLAG_PHONE", "FLAG_EMAIL"]
    join_df[columns_to_replace] = join_df[columns_to_replace].replace({'Y': 1, 'N': 0})
    indue_status = ["X", "C", "0", "1", "2"]
    risk = join_df["STATUS"].isin(indue_status)
    risk_value = np.where(risk, 0, 1)
    join_df['RISK'] = risk_value
    print(join_df.columns)

    risk_by_id = join_df.groupby('ID')['RISK'].transform('max') == 1
    join_df.loc[risk_by_id, 'RISK'] = 1
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
    total_rows = len(credit_data)
    risk_count = (credit_data["RISK"] == 1).sum()
    diff = risk_count - (total_rows - risk_count)
    if diff > 0:
        risk_0_rows = credit_data[credit_data['RISK'] == 0]
        duplicated_rows = risk_0_rows.sample(n=diff, replace=True)
        credit_data = pd.concat([credit_data, duplicated_rows], ignore_index=True)

    columns_to_normalize = ["CNT_CHILDREN", "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS", "DAYS_EMPLOYED", "DAYS_BIRTH"]
    scaler = StandardScaler()
    credit_data[columns_to_normalize] = scaler.fit_transform(credit_data[columns_to_normalize])
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
    credit_data.to_csv("./dataset/oversampling_records.csv")


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
    
    hc = PC(X_train)
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
        # try:
        #     query_result = infer.query(variables=['RISK'], evidence=i)
        #     result.append(np.argmax(query_result.values))
        # except Exception as e:
        #     print(e)
        #     exit(0)
    matrix = confusion_matrix(y_test, result, labels=[0, 1])
    print(str(matrix))

combine_data()
data_preprocess()
analyze_data()
train_model()
