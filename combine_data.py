import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianNetwork, MarkovNetwork
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator, BicScore, BayesianEstimator, PC, MmhcEstimator, ExpectationMaximization
from pgmpy.inference import VariableElimination, BeliefPropagation
from sklearn.metrics import f1_score

def combine_data():
    application = pd.read_csv("./dataset/application_record.csv", index_col= None)
    credit = pd.read_csv("./dataset/credit_record.csv", index_col= None)
    join_df = pd.merge(application, credit, on="ID", how='inner')
    columns_to_replace = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY", "FLAG_WORK_PHONE", "FLAG_MOBIL", "FLAG_PHONE", "FLAG_EMAIL"]
    join_df[columns_to_replace] = join_df[columns_to_replace].replace({'Y': 1, 'N': 0})
    indue_status = ["X", "C", "0", ]
    risk = join_df["STATUS"].isin(indue_status)
    risk_value = np.where(risk, 1, 0)
    join_df['RISK'] = risk_value
    print(join_df.columns)

    risk_by_id = join_df.groupby('ID')['RISK'].transform('min') == 0
    join_df.loc[risk_by_id, 'RISK'] = 0
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

    join_df.to_csv("./dataset/complete_credit_record.csv", index=None)


def data_preprocess():
    credit_data = pd.read_csv("./dataset/complete_credit_record.csv")

    columns_to_normalize = ["AMT_INCOME_TOTAL", "DAYS_EMPLOYED", "DAYS_BIRTH"]
    for col in columns_to_normalize:
        credit_data[col] = pd.qcut(credit_data[col], q=30, labels=False, duplicates='drop') 
    
    credit_data["OCCUPATION_TYPE"] = credit_data["OCCUPATION_TYPE"].fillna("UNKOWN")

    income_type_dict = {"Working": 5, "Commercial associate": 3, "Pensioner": 3, "State servant": 5, "Student": 1}
    education_type_dict = {"Higher education": 4, "Secondary / secondary special": 2, "Incomplete higher": 3, "Lower secondary": 1, "Academic degree": 5}
    family_type_dict = {"Civil marriage": 3, "Married": 3, "Single / not married": 1, "Separated": 2, "Widow": 2}
    housing_type_dict = {"House / apartment": 5, "Municipal apartment": 4, "Co-op apartment": 3, "Office apartment": 2, "Rented apartment": 1, "With parents": 0}
    credit_data["NAME_INCOME_TYPE"] = credit_data["NAME_INCOME_TYPE"].replace(income_type_dict)
    credit_data["NAME_EDUCATION_TYPE"] = credit_data["NAME_EDUCATION_TYPE"].replace(education_type_dict)
    credit_data["NAME_FAMILY_STATUS"] = credit_data["NAME_FAMILY_STATUS"].replace(family_type_dict)
    credit_data["NAME_HOUSING_TYPE"] = credit_data["NAME_HOUSING_TYPE"].replace(housing_type_dict)
    credit_data.to_csv("./dataset/oversampling_records.csv", index=None)

def oversample(train_data):
    total_rows = len(train_data)
    risk_count = (train_data["RISK"] == 0).sum()
    diff = risk_count - (total_rows - risk_count)
    
    if diff > 0:
        risk_0_rows = train_data[train_data['RISK'] == 1]
    else:
        risk_0_rows = train_data[train_data['RISK'] == 0]
    duplicated_rows = risk_0_rows.sample(n=abs(diff), replace=True)
    train_data = pd.concat([train_data, duplicated_rows], ignore_index=True)
    return train_data, train_data["RISK"]

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
    
    X_train, X_test = oversample(X_train)
    
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
    data_df = pd.read_csv("./dataset/oversampling_records.csv")
    data_df = data_df.drop(columns=["ID", "FLAG_WORK_PHONE", "FLAG_MOBIL", "FLAG_PHONE", "FLAG_EMAIL"])
    print(data_df["RISK"].unique())
    X_train, X_test, y_train, y_test = train_test_split(data_df, data_df["RISK"], test_size=0.2, random_state=42)
    # X_train, y_train = oversample(X_train)
    
    risk_num = (X_train["RISK"] == 1).sum()
    print(risk_num / len(X_train))
    # used_features = ['DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', "NAME_EDUCATION_TYPE", "FLAG_OWN_REALTY"]
    # model = BayesianNetwork([('DAYS_EMPLOYED', 'AMT_INCOME_TOTAL'), ('NAME_INCOME_TYPE', 'AMT_INCOME_TOTAL'),
    #                          ('AMT_INCOME_TOTAL', 'RISK'), ("NAME_EDUCATION_TYPE", "RISK"), ("FLAG_OWN_REALTY", "RISK")])
    # estimator = MaximumLikelihoodEstimator(model, X_train)
    
    used_features = ["AMT_INCOME_TOTAL", "NAME_EDUCATION_TYPE", "FLAG_OWN_REALTY", "DAYS_EMPLOYED", "CNT_CHILDREN", "NAME_HOUSING_TYPE", "FLAG_OWN_CAR"]
    model = BayesianNetwork([('DAYS_EMPLOYED', 'RISK'), ('DAYS_EMPLOYED', 'AMT_INCOME_TOTAL'), ('AMT_INCOME_TOTAL', 'RISK'), 
                             ("NAME_EDUCATION_TYPE", "RISK"),("FLAG_OWN_REALTY", "RISK"), ("CNT_CHILDREN", "RISK"), 
                             ("NAME_HOUSING_TYPE", "RISK"), ("FLAG_OWN_CAR", "RISK")])
    
    # edges, used_features = search_model(X_train=X_train)
    # model = BayesianNetwork(edges)
    
    model.fit(X_train, estimator=BayesianEstimator)
    predictions = model.predict(X_test[used_features])
    print(predictions["RISK"].unique())
    print(len(y_test))
    print(len(predictions))
    accuracy = accuracy_score(y_test, predictions["RISK"].to_numpy())

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions["RISK"].to_numpy())
    f1 = f1_score(y_test, predictions["RISK"].to_numpy())

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
        if i[1] == "RISK":
            used.append(i[0])
    print(edges)
    print(used)
    return edges, used
                
    


combine_data()
data_preprocess()
analyze_data()
# train_model()
train_customized()
