# BINF5507 ICA vs PCA
# March 26, 2025
# Rachel Yorke, Candace Peters, Candice Chen

import arff
import labels as labels
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
from urllib.request import urlretrieve
from sklearn.preprocessing import StandardScaler

# Load EEG dataset, formatting dataset
df = pd.read_csv('EEG_Eye_State.csv', header=0, skiprows=[1])
X = df.drop('eyeDetection', axis=1).values
labels = df['eyeDetection'].values

# Preprocessing - Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA and ICA
pca = PCA(n_components=5, random_state=42)
pca_components = pca.fit_transform(X_scaled)

ica = FastICA(n_components=5, random_state=42, whiten='unit-variance', max_iter=1000)
ica_components = ica.fit_transform(X_scaled)

# Visualize PCA and ICA components
plt.figure(figsize=(12, 6))

# PCA visualization
for i in range(5):
    plt.subplot(2, 5, i+1)
    plt.plot(pca_components[:, i])
    plt.title(f'PCA Comp {i+1}')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')

# ICA visualization
for i in range(5):
    plt.subplot(2, 5, i+6)
    plt.plot(ica_components[:, i])
    plt.title(f'ICA Comp {i+1}')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Scatter plot comparison for first two components
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=labels, cmap='coolwarm', alpha=0.6, edgecolor='k')
plt.title('PCA Components (EEG Eye State)')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.subplot(1, 2, 2)
plt.scatter(ica_components[:, 0], ica_components[:, 1], c=labels, cmap='coolwarm', alpha=0.6, edgecolor='k')
plt.title('ICA Components (EEG Eye State)')
plt.xlabel('IC1')
plt.ylabel('IC2')

plt.tight_layout()
plt.show()

# PCA explained variance for reference
print("Explained variance by PCA components:")
print(pca.explained_variance_ratio_)