# K-means Clustering for the Screening of High-Performance Phosphorus-Centered Lithium Salt Additives

### Description
This code from the manuscript S. Zhang, et al. Intramolecular Charge Redistribution of Phosphate Anions Enabling Ultra-High Voltage Lithium Metal Battery of 600 Wh kg-1.

### Contents
```
├── main.py 
├── wash.py
├── final_score.py
├── data/
│   └── molecules1.csv    # raw data 1 originates from the PubChem database
│   └── molecules2.csv    # raw data 2 originates from the in-house database
│   └── molecules_filtered.csv    # the data obtained from preliminary screening of raw data 1 and 2 after running wash.py
│   └── molecules_properties.csv    # input samples that by manually importing the structural properties of the molecules from molecules_filtered.csv for KMeans clustering
├── kmeans_output/    # obtained after running main.py
│   ├── clustering_results.csv    # KMeans clustering data
│   ├── clustering_pca_plot.png    # plot of KMeans clustering results
│   ├── cluster_centers_pca.csv    # coordinates of cluster centers obtained via PCA dimensionality reduction
│   ├── pca_results.csv    # PCA dimensionality reduction coordinates for individual molecules
│   ├── silhouette_plot.png    # plot of silhouette coefficients for different K values
│   └── silhouette_scores.csv    # table of silhouette coefficients for different K values
└── score_output    # obtained after running final_score.py
     ├── cluster_2_scores.csv    # detailed scores of molecules in cluster 2
     ├── ranking_results.csv    # final scores and rankings in cluster 2
     ├── salt_performance_scores.png    # detailed score plot for the top 15 molecules
```
### Operating system
- Windows, Mac, Linux

### Dependencies needed to reproduce the machine learning results reported in the paper.
- RDKit
- scikit‑learn
- scipy
- numpy
- matplotlib
- pandas
- seaborn
- requests
### License
This project uses the [MIT LICENSE](LICENSE).
