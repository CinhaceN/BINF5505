import openpyxl
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sklearn.inspection import permutation_importance

# Load dataset
data_raw = pd.read_excel("RADCURE_Clinical_v04_20241219.xlsx", engine='openpyxl')
smoker_mapping = {'current':1, 'Ex-smoker':-1, 'Non-smoker':2, 'unknown':0}
HPV_mapping = {'Yes, Negative':-1, 'Yes, positive':1}
stage_mapping = {'I':1, 'IB':1.2, 'II':2, 'IIA':2.1, 'IIB':2.2, 'III':3, 'IIIA':3.1,
                 'IIIC':3.3, 'IV': 4, 'IVA': 4.1,'IVB': 4.2,'IVC': 4.3, 'X':5}
status_mapping = {'Alive':1, 'Dead':0}
data_raw['Smoking Status'] = data_raw['Smoking Status'].map(smoker_mapping)
data_raw['HPV'] = data_raw['HPV'].map(HPV_mapping)
data_raw['Stage'] = data_raw['Stage'].map(stage_mapping)
data_raw['Status'] = data_raw['Status'].map(status_mapping)
data = data_raw.fillna({'Age': 0, 'Smoking Status': 0, 'HPV':0, 'Stage':0})
data['Status'] = data['Status'].astype(bool)

# print(data['Smoking PY'].dtypes)
# print(data['Age'].dtypes)

# Fit the Kaplan-Meier estimator
kmf = KaplanMeierFitter()
kmf.fit(data['Age'], event_observed=data['Smoking Status'])

# Plot the Kaplan-Meier curve
kmf.plot_survival_function()
plt.title('Kaplan-Meier Curve')
plt.xlabel('Age')
plt.ylabel('Smoking Status')
plt.show()

# Cox Proportional Hazards Model
cph = CoxPHFitter()
cph.fit(data[['Age', 'Stage', 'HPV', 'Length FU']], duration_col='Length FU',
        event_col='Stage')
cph.print_summary()
cph.plot()
plt.show()

# Random Survival Forest
X = data[['Age', 'Stage', 'Smoking Status', 'HPV']]
y = data[['Status', 'Length FU']].to_records(index=False)
rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, random_state=42)
rsf.fit(X, y)

# Feature Importance
result = permutation_importance(rsf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
importances = result.importances_mean
plt.barh(X.columns, importances)
plt.xlabel('Feature Importance')
plt.title('Random Survival Forest Feature Importance')
plt.show()

# Concordance Index Comparison
cph_cindex = concordance_index_censored(data['Stage'], data['Length FU'], cph.predict_partial_hazard(data))[0]
rsf_cindex = concordance_index_censored(y['Status'], y['Length FU'], rsf.predict(X))[0]
print(f"Cox Model C-Index: {cph_cindex:.3f}, RSF C-Index: {rsf_cindex:.3f}")
