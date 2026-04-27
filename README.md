# K-means Clustering for the Screening of High-Performance Phosphorus-Centered Lithium Salt Additives

### Description
This code from the manuscript S. Zhang, et al. Intramolecular Charge Redistribution of Phosphate Anions Enabling Ultra-High Voltage Lithium Metal Battery of 600 Wh kg-1.

### Contents
```
├── main.py 
├── wash.py
├── final_score.py
├── data/
│   ├── molecules1.csv    # raw data 1 originates from the PubChem database
│   ├── molecules2.csv    # raw data 2 originates from the in-house database
│   ├── molecules_filtered.csv    # the data obtained from preliminary screening of raw data 1 and 2 after running wash.py
│   └── molecules_properties.csv    # input samples that by manually importing the structural properties of the molecules from molecules_filtered.csv for KMeans clustering
├── kmeans_output/    # obtained after running main.py
│   ├── clustering_results.csv    # KMeans clustering data
│   ├── clustering_pca_plot.png    # plot of KMeans clustering results
│   ├── cluster_centers_original.csv    # coordinates of cluster centers
│   ├── cluster_centers_pca.csv    # coordinates of cluster centers obtained via PCA dimensionality reduction
│   ├── pca_results.csv    # PCA dimensionality reduction coordinates for individual molecules
│   ├── silhouette_plot.png    # plot of silhouette coefficients for different K values
│   └── silhouette_scores.csv    # table of silhouette coefficients for different K values
└── score_output/    # obtained after running final_score.py
    ├── cluster_2_scores.csv    # detailed scores of molecules in cluster 2
    ├── ranking_results.csv    # final scores and rankings in cluster 2
    └── salt_performance_scores.png    # detailed score plot for the top 15 molecules
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

### Steps for Use
- Run wash.py to perform preliminary screening of the raw data (molecules1.csv and molecules2.csv).
This generates molecules_filtered.csv containing the filtered molecules.

- Perform density functional theory (DFT) calculations on the molecules in molecules_filtered.csv to obtain their electrostatic potentials and polarizabilities.
Save the computed properties in molecules_properties.csv.

- Run main.py to perform K‑means clustering on the pre‑screened molecules using their electrostatic potentials and polarizabilities.
The molecules belonging to cluster 2 are identified as the candidates with high performance.

- Run final_score.py to compute final scores and rank the molecules in cluster 2.
The output includes detailed scores and ranking results in the score_output/ folder.

### License
This project uses the [MIT LICENSE](LICENSE).

### Disclaimer
This code is intended for educational and research purposes only. Please ensure that you comply with relevant laws and regulations as well as the terms of service of the target website when using this code. The author is not responsible for any legal liabilities or other issues arising from the use of this code.
