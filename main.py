import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

def kmeans_clustering_analysis(data, feature_columns, min_k, max_k, random_state=42):
    """
    K-means++ clustering analysis function

    Parameters:
    data: DataFrame containing the data
    feature_columns: list, feature column names
    min_k: int, minimum number of clusters to try
    max_k: int, maximum number of clusters
    random_state: int, random seed

    Returns:
    dict with keys:
        'data': original DataFrame with clustering results and PCA coordinates
        'best_k': optimal number of clusters
        'labels': cluster labels for each sample
        'centers_original': cluster centers in original feature space
        'centers_pca': cluster centers in PCA space
        'silhouette_score': silhouette score for the optimal k
        'X_pca': PCA coordinates of all samples
        'scaler': fitted StandardScaler
        'pca': fitted PCA object
        'kmeans_model': fitted KMeans model
        'pca_data': DataFrame with PCA coordinates, cluster labels and distances
        'feature_columns': list of used feature column names
        'silhouette_data': DataFrame with k and corresponding silhouette scores
        'silhouette_fig': matplotlib figure of silhouette scores vs. k
    """

    # 1. Extract feature data
    X = data[feature_columns].values

    # 2. Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Find the best K using silhouette score, starting from min_k
    silhouette_scores = []
    k_values = range(min_k, max_k + 1)

    print("Finding the best K (based on silhouette score):")
    print("-" * 50)

    for k in k_values:
        # Use K-means++ initialization
        kmeans = KMeans(n_clusters=k, init='k-means++',
                        random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)

        print(f"K = {k}: Silhouette Score = {silhouette_avg:.4f}")

    # 4. Select the best K (maximum silhouette score)
    best_k = k_values[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)

    print("-" * 50)
    print(f"Best K: {best_k}, Silhouette Score: {best_score:.4f}")

    # 5. Perform final clustering with the best K
    best_kmeans = KMeans(n_clusters=best_k, init='k-means++',
                         random_state=random_state, n_init=10)
    best_labels = best_kmeans.fit_predict(X_scaled)
    data['Cluster'] = best_labels

    # 6. Standardized scale cluster centers
    cluster_centers_scaled = best_kmeans.cluster_centers_
    cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
    cluster_centers_df = pd.DataFrame(cluster_centers,
                                      columns=feature_columns)
    cluster_centers_df['Cluster'] = range(best_k)

    # 7. PCA scale cluster centers
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centers_pca = pca.transform(cluster_centers_scaled)
    cluster_centers_pca_df = pd.DataFrame(centers_pca,
                                          columns=['PCA1_center', 'PCA2_center'])
    cluster_centers_pca_df['Cluster'] = range(best_k)

    # 8. Add PCA results to the original data
    data['PCA1'] = X_pca[:, 0]
    data['PCA2'] = X_pca[:, 1]

    # Create a separate DataFrame for PCA data
    pca_data = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': best_labels,
    })
    pca_data['idx'] = data['idx'].values
    pca_data['smiles'] = data['smiles'].values

    # 9. Visualization - PCA clustering plot only
    plt.figure(figsize=(10, 8))

    # Scatter plot of all samples
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                          c=best_labels, cmap='viridis',
                          alpha=0.7, edgecolors='w', s=80)

    # Mark cluster centers
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
                c='red', marker='X', s=300, alpha=1.0,
                edgecolors='k', linewidth=2, label='Cluster Centers', zorder=10)

    # Add labels and legend
    plt.xlabel(f'PCA1')
    plt.ylabel(f'PCA2')
    plt.title(f'K-means++ Clustering Results (K={best_k}, Silhouette Score={best_score:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster Labels')

    plt.tight_layout()
    plt.show()

    # 10. Create a DataFrame for silhouette scores
    silhouette_data = pd.DataFrame({
        'K': k_values,
        'Silhouette_Score': silhouette_scores
    })

    # Create silhouette score plot (display and save)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    ax.axvline(x=best_k, color='r', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title(f'Silhouette Score vs K (Best K={best_k})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)

    # Mark the best point
    ax.scatter([best_k], [best_score], color='red', s=200,
               edgecolors='k', linewidth=2, zorder=10)

    plt.tight_layout()
    plt.show()

    return {
        'data': data,
        'best_k': best_k,
        'labels': best_labels,
        'centers_original': cluster_centers_df,
        'centers_pca': cluster_centers_pca_df,
        'silhouette_score': best_score,
        'X_pca': X_pca,
        'scaler': scaler,
        'pca': pca,
        'kmeans_model': best_kmeans,
        'pca_data': pca_data,
        'feature_columns': feature_columns,
        'silhouette_data': silhouette_data,
        'silhouette_fig': fig
    }

def save_results(results, data, output_dir='./kmeans_output'):
    """
    Save clustering results to files

    Parameters:
    results: dict, clustering analysis results from kmeans_clustering_analysis
    data: DataFrame, original data with clustering results
    output_dir: str, output directory path
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save all molecule clustering results (including PCA data)
    output_path = os.path.join(output_dir, 'clustering_results.csv')
    data.to_csv(output_path, index=False)
    print(f"\nClustering results saved to '{output_path}'")

    # 2. Save cluster centers
    centers_original_path = os.path.join(output_dir, 'cluster_centers_original.csv')
    results['centers_original'].to_csv(centers_original_path, index=False)
    centers_pca_path = os.path.join(output_dir, 'cluster_centers_pca.csv')
    results['centers_pca'].to_csv(centers_pca_path, index=False)

    # 3. Save PCA data
    pca_path = os.path.join(output_dir, 'pca_results.csv')
    results['pca_data'].to_csv(pca_path, index=False)
    print(f"PCA data saved to '{pca_path}'")

    # 4. Save silhouette score data
    silhouette_data_path = os.path.join(output_dir, 'silhouette_scores.csv')
    results['silhouette_data'].to_csv(silhouette_data_path, index=False)
    print(f"Silhouette score data saved to '{silhouette_data_path}'")

    # 5. Save silhouette plot
    silhouette_plot_path = os.path.join(output_dir, 'silhouette_plot.png')
    results['silhouette_fig'].savefig(silhouette_plot_path, dpi=300, bbox_inches='tight')
    print(f"Silhouette plot saved to '{silhouette_plot_path}'")

    # 6. Create and save the clustering PCA plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot of samples
    scatter = ax.scatter(results['X_pca'][:, 0], results['X_pca'][:, 1],
                         c=results['labels'], cmap='viridis',
                         alpha=0.7, edgecolors='w', s=80)

    # Mark cluster centers
    centers_pca = results['centers_pca'][['PCA1_center', 'PCA2_center']].values
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
               c='red', marker='X', s=300, alpha=1.0,
               edgecolors='k', linewidth=2, label='Cluster Centers', zorder=10)

    # Add labels and legend
    ax.set_xlabel(f'PCA1')
    ax.set_ylabel(f'PCA2')
    ax.set_title(f'K-means++ Results (K={results["best_k"]}, Silhouette Score={results["silhouette_score"]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster Labels')

    plt.tight_layout()

    # Save clustering PCA plot
    pca_plot_path = os.path.join(output_dir, 'clustering_pca_plot.png')
    fig.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Clustering PCA plot saved to '{pca_plot_path}'")


if __name__ == "__main__":
    rawdata = pd.read_csv('./data/molecules_properties.csv')
    data = pd.DataFrame(rawdata)

    feature_columns = ['ESPmin', 'ESPmax', 'polarizability']

    results = kmeans_clustering_analysis(
        data=data,
        feature_columns=feature_columns,
        min_k=5,
        max_k=15,
        random_state=42
    )
    save_results(results, results['data'])