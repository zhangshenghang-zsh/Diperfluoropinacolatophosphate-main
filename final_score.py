import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import requests as rq
import json
from collections import defaultdict
from rdkit import Chem

class FunctionalGroupScorer:
    def __init__(self):
        # Define functional groups and their SMARTS and weights
        self.functional_groups = [
            {'name': 'F', 'smarts': '[F]', 'weight': 1},
            {'name': 'O_ether', 'smarts': '[OD2]', 'weight': 0.5},      # Ether oxygen (divalent oxygen)
            {'name': 'NO2', 'smarts': '[N+](=O)[O-]', 'weight': 0.5},
            {'name': 'CN', 'smarts': 'C#N', 'weight': 1},
            {'name': 'oxo_or_anion', 'smarts': '[OX1]', 'weight': 2}, # Double-bonded oxygen or oxygen anion (monovalent oxygen)
            {'name': 'H', 'smarts': '[H]', 'weight': -1},
        ]

    def calculate_score(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0, {}

            mol = Chem.AddHs(mol)
            used_atoms = set()        # Record atom indices that have been matched
            total_score = 0
            group_counts = defaultdict(int)

            for group in self.functional_groups:
                pattern = Chem.MolFromSmarts(group['smarts'])
                if pattern is None:
                    continue
                matches = mol.GetSubstructMatches(pattern)
                for match in matches:
                    # Check whether all atoms in the current match are unused
                    if not any(atom_idx in used_atoms for atom_idx in match):
                        # Accept this match
                        used_atoms.update(match)
                        total_score += group['weight']
                        group_counts[group['name']] += 1

            return total_score, dict(group_counts)
        except Exception as e:
            print(f"Error processing SMILES: {smiles}, Error: {e}")
            return 0, {}

def get_scscore(smiles):
    time.sleep(2)
    api_url = 'https://askcos.mit.edu/api/molecular-complexity/call-async'
    headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0',
             "Accept": "application/json",
            "Content-Type": "application/json"
            }
    request_data = {
        "smiles": smiles,
        "complexity_metrics": ["scscore"]
    }
    print(f'SCScore: Processing {smiles}')
    re1 = rq.post(api_url, json=request_data,headers=headers)
    if re1.status_code == 200:
        re2=rq.get(f'https://askcos.mit.edu/api/legacy/celery/task/{re1.json()}')
        if re2.status_code == 200:
            print(f"Post sucess: {re2.status_code}")
            data=json.loads(re2.text)
            state = data["state"]
            result =data["output"]["result"]
            if state == "SUCCESS":
                print(f"Get sucess: {re2.status_code}")
                return 5-float(result["scscore"])
            else:
                print(f"Get failed: {re2.status_code}")
                return None
    else:
        print(f"Post failed: {re1.status_code}")
        return None


df = pd.read_csv('./kmeans_output/clustering_results.csv')
df = df[df['Cluster'] == 2].copy()
scorer = FunctionalGroupScorer()

# Extract all functional group names from the list
all_groups = [g['name'] for g in scorer.functional_groups]

# 1. Calculate functional group score
df['functionalgroupscorer'] = df['smiles'].apply(
    lambda x: scorer.calculate_score(x)[0] if pd.notna(x) else 0
)

# 2. Create a count column for each functional group
for group in all_groups:
    df[f'{group}_count'] = df['smiles'].apply(
        lambda x: scorer.calculate_score(x)[1].get(group, 0) if pd.notna(x) else 0
    )
#
df['scscore']=df['smiles'].apply(lambda x: get_scscore(x))
df.to_csv('score_output/cluster_2_scores.csv', index=False)

df = pd.read_csv('score_output/cluster_2_scores.csv')
# Set weights (can be customized)
weights = {
    'ESPmin': 1.0,
    'ESPmax': 1.0,
    'polarizability': 1.0,
    'dipole_moment': 1.0,  # weight can be set to 0 to exclude this feature
    'functionalgroupscorer': 4.0,
    'scscore': 2.0,
}

# Only select features with non-zero weight
selected_features = [feat for feat, weight in weights.items() if weight > 0]

# Normalization processing
def normalize_feature(df, feature_name, weight, direction='smaller'):
    """
    Normalize feature values

    Parameters:
    - df: DataFrame
    - feature_name: feature column name
    - weight: weight for this feature
    - direction: 'smaller' means the smaller the value the better, 'larger' means the larger the value the better
    """
    if direction == 'smaller':
        # For features where smaller is better: normalized score = (max - current) / (max - min)
        max_val = df[feature_name].max()
        min_val = df[feature_name].min()
        df[f'{feature_name}_norm'] = (max_val - df[feature_name]) / (max_val - min_val + 1e-8)
    elif direction == 'larger':
        # For features where larger is better: normalized score = (current - min) / (max - min)
        max_val = df[feature_name].max()
        min_val = df[feature_name].min()
        df[f'{feature_name}_norm'] = (df[feature_name] - min_val) / (max_val - min_val + 1e-8)

    # Apply weight
    df[f'{feature_name}_score'] = df[f'{feature_name}_norm'] * weight

    return df


# Specify the direction for each feature
feature_directions = {
    'ESPmin': 'smaller',  # smaller is better
    'ESPmax': 'larger',   # larger is better
    'polarizability': 'larger',
    'functionalgroupscorer':'larger',
    'scscore': 'larger',
    'dipole_moment': 'larger'
}

# Normalize and weight each feature
for feature in selected_features:
    direction = feature_directions.get(feature, 'larger')  # default: larger is better
    df = normalize_feature(df, feature, weights[feature], direction)

# Calculate weighted total score
score_columns = [f'{feat}_score' for feat in selected_features]
df['total_score'] = df[score_columns].sum(axis=1)

# Sort by weighted score in descending order
df_sorted = df.sort_values('total_score', ascending=False)


# Create ranking table
rank_data = []
for i, (idx, row) in enumerate(df_sorted.iterrows(), 1):
    rank_data.append([
        i,
        row['smiles'],
        row['idx'],
        f"{row['ESPmin']:.4f}",
        f"{row['ESPmax']:.4f}",
        f"{row['polarizability']:.4f}",
        f"{row['dipole_moment']:.4f}",
        f"{row['functionalgroupscorer']:.4f}",
        f"{row['scscore']:.4f}",
        f"{row['total_score']:.4f}",
    ])

headers = ['rank', 'smiles', 'idx',
           'ESPmin', 'ESPmax', 'polarizability', 'dipole_moment',
           'functionalgroupscorer', 'scscore','total_score']

# Convert to DataFrame
result_df = pd.DataFrame(rank_data, columns=headers)

# Save to CSV
result_df.to_csv('./score_output/ranking_results.csv', index=False, encoding='utf-8-sig')
print("Results saved to ranking_results.csv")

df_sorted = df_sorted.copy()
df_sorted.insert(0, 'rank', range(1, len(df_sorted) + 1))


# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Select the top N results to display (avoid overcrowding)
N = min(15, len(df_sorted))
top_N_df = df_sorted.head(N).copy()

# Prepare stacked bar chart data
features_scores = []
for feature in selected_features:
    features_scores.append(top_N_df[f'{feature}_score'].values)

features_scores = np.array(features_scores)  # shape: (num_features, N)
salt_names = top_N_df['smiles'].values
final_scores = top_N_df['total_score'].values

# Create color map
cmap = plt.cm.tab20c
colors = [cmap(i % 20) for i in range(len(selected_features))]

# Create figure
fig, ax1 = plt.subplots(figsize=(14, 8))

# First subplot: stacked bar chart
bottom = np.zeros(N)
bars = []
for idx, (feature, scores) in enumerate(zip(selected_features, features_scores)):
    bar = ax1.bar(range(N), scores, bottom=bottom, color=colors[idx],
                  edgecolor='white', linewidth=0.5, alpha=0.85, label=feature)
    bars.append(bar)
    bottom += scores

# Display total score above the bars
for i, (score, salt) in enumerate(zip(final_scores, salt_names)):
    ax1.text(i, score + 0.02 * max(final_scores), f'{score:.2f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Set x-axis labels
ax1.set_xticks(range(N))
ax1.set_xticklabels([f"{i+1}. {salt}" for i, salt in enumerate(salt_names)],
                   rotation=45, ha='right', fontsize=10)

# Set y-axis and title
ax1.set_ylabel('Total Score', fontsize=12, fontweight='bold')
ax1.set_title('Lithium Salt Performance Scores', fontsize=14, fontweight='bold', pad=20)

# Add legend
ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

plt.savefig('./score_output/salt_performance_scores.png', dpi=300, bbox_inches='tight')
plt.show()