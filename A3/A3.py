# BINF5507 Assignment 3
# Candice Chen
# Feb. 28, 2025

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering

# Dataset where DBSCAN excels
"""
make_moons(n_samples=300, noise=0.05) → Good for DBSCAN (non-spherical
clusters)
"""
# Generate moons dataset
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# Apply clustering algorithms
dbscan = DBSCAN(eps=0.3, min_samples=5).fit(X_moons)
kmeans = KMeans(n_clusters=2, random_state=42).fit(X_moons)
agg = AgglomerativeClustering(n_clusters=2).fit(X_moons)

# Plot results
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
titles = ['DBSCAN', 'k-Means', 'Hierarchical Clustering']
for i, (labels, title) in enumerate(zip([dbscan.labels_, kmeans.labels_, agg.labels_], titles)):
    ax[i].scatter(X_moons[:, 0], X_moons[:, 1], c=labels, cmap='viridis', edgecolor='k')
    ax[i].set_title(title, fontsize=12)
    ax[i].set_xlabel('Feature 1')
    ax[i].set_ylabel('Feature 2')
plt.tight_layout()
plt.show()

# Dataset where DBSCAN struggles
"""
make_blobs(n_samples=300, centers=3, cluster_std=[1.0, 2.5, 0.5])
→ Shows DBSCAN struggles with varying densities.
"""
# Generate blobs dataset
X_blobs, _ = make_blobs(
    n_samples=300, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=42
)

# Apply clustering algorithms
dbscan = DBSCAN(eps=0.8, min_samples=5).fit(X_blobs)
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_blobs)
agg = AgglomerativeClustering(n_clusters=3).fit(X_blobs)

# Plot results
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, (labels, title) in enumerate(zip([dbscan.labels_, kmeans.labels_, agg.labels_], titles)):
    ax[i].scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels, cmap='viridis', edgecolor='k')
    ax[i].set_title(title, fontsize=12)
    ax[i].set_xlabel('Feature 1')
    ax[i].set_ylabel('Feature 2')
plt.tight_layout()
plt.show()
