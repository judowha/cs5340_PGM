import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def combine_data():
    application = pd.read_csv("./dataset/application_record.csv")
    credit = pd.read_csv("./dataset/credit_record.csv")
    join_df = pd.merge(application, credit, on="ID", how='inner')
    columns_to_replace = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY", "FLAG_WORK_PHONE", "FLAG_MOBIL", "FLAG_PHONE", "FLAG_EMAIL"]
    join_df[columns_to_replace] = join_df[columns_to_replace].replace({'Y': 1, 'N': 0})
    indue_status = ["X", "C"]
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

    print("unique income types: ", unique_income_types)
    print("unique education types: ", unique_education_types)
    print("unique family status: ", unique_family_status)
    print("unique housing types: ", unique_housing_types)

    join_df.to_csv("./dataset/complete_credit_record.csv")


def data_analysis():
    credit_data = pd.read_csv("./dataset/complete_credit_record.csv")
    total_rows = len(credit_data)
    risk_count = (credit_data["RISK"] == 1).sum()
    diff = risk_count - (total_rows - risk_count)
    if diff > 0:
        risk_0_rows = credit_data[credit_data['RISK'] == 0]
        duplicated_rows = risk_0_rows.sample(n=diff, replace=True)
        credit_data = pd.concat([credit_data, duplicated_rows], ignore_index=True)

    credit_data.to_csv("./dataset/oversampling_records.csv")


def preprocess_data():
    columns_to_normalize = ["CNT_CHILDREN", "AMT_INCOME_TOTAL"]


# combine_data()
data_analysis()
