import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming `data` is your data
data = pd.read_csv("./dataset/combined_data.csv")

columns = ["CNT_CHILDREN", "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS", "DAYS_EMPLOYED", "DAYS_BIRTH"]

stat, p = stats.kstest(np.random.normal(size=30000), stats.norm.cdf)
print('Statistics=%.3f, p=%.3f' % (stat, p))

for column in columns:
    print(f"Processing column: {column}")
    column_data = data[column].to_numpy()

    loc, scale = stats.norm.fit(column_data)
    n = stats.norm(loc=loc, scale=scale)

    stat, p = stats.kstest(column_data, n.cdf)
    print(stat)
    print(p)
    print('Statistics=%f, p=%f' % (stat, p))

    # interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

    # Q-Q plot
    plt.figure()
    stats.probplot(column_data, dist="norm", plot=plt)
    plt.title(f"{column} Q-Q plot")
    plt.show()