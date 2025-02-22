# BINF5507 - Assignment 2
# Candice Chen
# Feb. 21, 2025

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Load dataset
df = pd.read_csv('heart_disease_uci.csv')

# Preprocessing for Regression (Target: 'chol')
# Drop rows where 'chol' or other key features are missing
df_reg = df.dropna(subset=['chol']).copy()
X_reg = df_reg.drop(columns=['chol', 'num'])  # Features
y_reg = df_reg['chol']  # Target

# Preprocessing for Classification (Target: 'num')
# Convert 'num' to binary: 0 (no disease) vs. 1 (disease present)
df_cls = df.dropna(subset=['num']).copy()
df_cls['num'] = df_cls['num'].apply(lambda x: 1 if x > 0 else 0)
X_cls = df_cls.drop(columns=['num', 'chol'])  # Features
y_cls = df_cls['num']  # Target

# Identify categorical and numerical columns
categorical_cols = X_reg.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X_reg.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessor pipeline: OneHotEncode categorical, scale numerical
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ])

# Split data into train/test sets
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# %% [markdown]
# ## Part 2: Regression Model (ElasticNet)

# %%
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# Define hyperparameter grid
alphas = [0.001, 0.01, 0.1, 1, 10]
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
results = []

# Train models and evaluate
for alpha in alphas:
    for l1_ratio in l1_ratios:
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42))
        ])
        model.fit(X_reg_train, y_reg_train)
        y_pred = model.predict(X_reg_test)
        r2 = r2_score(y_reg_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
        results.append({'alpha': alpha, 'l1_ratio': l1_ratio, 'R2': r2, 'RMSE': rmse})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Pivot for heatmap
heatmap_r2 = results_df.pivot(index='alpha', columns='l1_ratio', values='R2')
heatmap_rmse = results_df.pivot(index='alpha', columns='l1_ratio', values='RMSE')

# Plot R2 heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_r2, annot=True, fmt=".2f", cmap='viridis')
plt.title('R² Scores for ElasticNet Hyperparameters')
plt.xlabel('l1_ratio')
plt.ylabel('alpha')
plt.show()

# Plot RMSE heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_rmse, annot=True, fmt=".1f", cmap='viridis')
plt.title('RMSE Scores for ElasticNet Hyperparameters')
plt.xlabel('l1_ratio')
plt.ylabel('alpha')
plt.show()

# Identify best configuration
best_config = results_df.loc[results_df['R2'].idxmax()]
print(
    f"Best Configuration: Alpha={best_config['alpha']}, l1_ratio={best_config['l1_ratio']}, R²={best_config['R2']:.2f}")

# %% [markdown]
# ## Part 3: Classification Models

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, roc_curve, \
    precision_recall_curve

# %% [markdown]
# ### Logistic Regression

# %%
# Define hyperparameters
penalties = ['l1', 'l2', 'none']
solvers = ['liblinear', 'saga']  # 'saga' supports 'elasticnet' but simplified here
cls_results = []

for penalty in penalties:
    for solver in solvers:
        if penalty == 'none' and solver != 'lbfgs':
            continue  # Skip incompatible combinations
        try:
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(penalty=penalty, solver=solver, max_iter=1000))
            ])
            model.fit(X_cls_train, y_cls_train)
            y_pred = model.predict(X_cls_test)
            y_proba = model.predict_proba(X_cls_test)[:, 1]

            accuracy = accuracy_score(y_cls_test, y_pred)
            f1 = f1_score(y_cls_test, y_pred)
            auroc = roc_auc_score(y_cls_test, y_proba)
            auprc = average_precision_score(y_cls_test, y_proba)

            cls_results.append({
                'model': 'Logistic Regression',
                'penalty': penalty,
                'solver': solver,
                'Accuracy': accuracy,
                'F1': f1,
                'AUROC': auroc,
                'AUPRC': auprc
            })
        except:
            continue  # Skip invalid parameter combinations

# %% [markdown]
# ### k-Nearest Neighbors (k-NN)

# %%
for n_neighbors in [1, 5, 10]:
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=n_neighbors))
    ])
    model.fit(X_cls_train, y_cls_train)
    y_pred = model.predict(X_cls_test)
    y_proba = model.predict_proba(X_cls_test)[:, 1]

    accuracy = accuracy_score(y_cls_test, y_pred)
    f1 = f1_score(y_cls_test, y_pred)
    auroc = roc_auc_score(y_cls_test, y_proba)
    auprc = average_precision_score(y_cls_test, y_proba)

    cls_results.append({
        'model': 'k-NN',
        'n_neighbors': n_neighbors,
        'Accuracy': accuracy,
        'F1': f1,
        'AUROC': auroc,
        'AUPRC': auprc
    })

# Convert results to DataFrame
cls_results_df = pd.DataFrame(cls_results)

# Display results
print(cls_results_df)

# %% [markdown]
# ### Plot AUROC and AUPRC Curves

# %%
# Plot for best Logistic Regression model
best_lr = cls_results_df[cls_results_df['model'] == 'Logistic Regression'].sort_values('AUROC', ascending=False).iloc[0]
model_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(penalty=best_lr['penalty'], solver=best_lr['solver'], max_iter=1000))
])
model_lr.fit(X_cls_train, y_cls_train)
y_proba_lr = model_lr.predict_proba(X_cls_test)[:, 1]

# Compute ROC curve and PR curve
fpr, tpr, _ = roc_curve(y_cls_test, y_proba_lr)
precision, recall, _ = precision_recall_curve(y_cls_test, y_proba_lr)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'Logistic Regression (AUROC = {best_lr["AUROC"]:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'Logistic Regression (AUPRC = {best_lr["AUPRC"]:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve')
plt.legend()
plt.show()

# Plot for best k-NN model
best_knn = cls_results_df[cls_results_df['model'] == 'k-NN'].sort_values('AUROC', ascending=False).iloc[0]
model_knn = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=best_knn['n_neighbors']))
])
model_knn.fit(X_cls_train, y_cls_train)
y_proba_knn = model_knn.predict_proba(X_cls_test)[:, 1]

fpr_knn, tpr_knn, _ = roc_curve(y_cls_test, y_proba_knn)
precision_knn, recall_knn, _ = precision_recall_curve(y_cls_test, y_proba_knn)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr_knn, tpr_knn, label=f'k-NN (AUROC = {best_knn["AUROC"]:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall_knn, precision_knn, label=f'k-NN (AUPRC = {best_knn["AUPRC"]:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve')
plt.legend()
plt.show()