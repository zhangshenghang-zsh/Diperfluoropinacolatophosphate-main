import re
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def extract_anion_with_rdkit(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        try:
            fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        except Exception as e:
            print(f"GetMolFrags error: {e}, for SMILES: {smiles}")
            return smiles

        # If there are no fragments or only one fragment, return directly
        if len(fragments) <= 1:
            return smiles

        # Look for fragments that do not contain lithium
        anion_fragments = []
        for frag in fragments:
            has_lithium = False
            for atom in frag.GetAtoms():
                if atom.GetSymbol() == 'Li':
                    has_lithium = True
                    break

            if not has_lithium:
                anion_fragments.append(frag)

        # If no anion fragment found, return the original
        if not anion_fragments:
            return smiles

        # If there are multiple anion fragments, combine them
        if len(anion_fragments) == 1:
            anion_mol = anion_fragments[0]
        else:
            # Combine multiple fragments
            anion_mol = anion_fragments[0]
            for frag in anion_fragments[1:]:
                anion_mol = Chem.CombineMols(anion_mol, frag)

        # Convert back to SMILES
        anion_smiles = Chem.MolToSmiles(anion_mol)

        return anion_smiles
    except Exception as e:
        print(f"Error processing SMILES: {smiles}, error: {e}")
        return smiles

def standardize_nitro_smiles(smiles):
    """
    Enhanced nitro standardization function, handles multiple nitro representation patterns
    """
    if pd.isna(smiles):  # Handle NaN values
        return smiles

    # Multiple nitro representation patterns
    patterns = [
        (r'\[N+\]\(=O\)\[O-\]', 'N(=O)=O'),
        (r'\[N+\]\(\[O-\]\)=O', 'N(=O)=O'),  # Another order
        (r'O=\[N+\]\(\[O-\]\)', 'N(=O)=O'),  # Double bond first
        (r'\[O-\]\[N+\]\(=O\)', 'N(=O)=O'),  # Oxygen first
    ]

    result = str(smiles)  # Ensure conversion to string
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)

    return result

def is_valid(smiles):
    # First check if it is a string; if not, return False directly
    if not isinstance(smiles, str):
        return False

    if '.' in smiles or '+' in smiles:
        return False
    else:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return True
            else:
                return False
        except:
            return False

def filter_molecules_by_elements(smiles_list):
    """
    Filter SMILES strings, keep only molecules containing specified elements

    Parameters:
    smiles_list: list, list of SMILES strings
    allowed_elements: list, allowed elements list, default=['H', 'C', 'N', 'O', 'F', 'P']

    Returns:
    list: filtered list of SMILES strings
    """
    allowed_elements = ['H', 'C', 'N', 'O', 'F', 'P']
    # Convert to set for fast lookup
    allowed_set = set(allowed_elements)
    filtered_smiles = []
    for smiles in smiles_list:
        try:
            # Convert SMILES to molecule object
            mol = Chem.MolFromSmiles(smiles)

            if mol is not None:
                # Check if all atoms in the molecule are in the allowed set
                all_allowed = True
                for atom in mol.GetAtoms():
                    element = atom.GetSymbol()
                    if element not in allowed_set:
                        all_allowed = False
                        break

                if all_allowed:
                    filtered_smiles.append(smiles)
        except:
            # Skip SMILES that cannot be parsed
            continue
    return filtered_smiles


def filter_molecules_without_functional_groups(smiles_list):
    """
    Filter out molecules containing specific functional groups

    Parameters:
    smiles_list: list of SMILES strings

    Returns:
    filtered DataFrame
    """
    filtered_data = []

    # Define functional group SMARTS patterns to exclude
    functional_group_patterns = {
        'OH': '[OH]',  # Hydroxyl
        'COOH': 'C(=O)O',  # Carboxyl
        'SH': '[SH]',  # Thiol
        'NH2': '[NH2]',  # Amino
    }

    p_h_bond_patterns = [
        '[P;H1]',
        '[P;H2]',
        '[P;H3]',
        '[PH]',
        '[P][H]',
        '[P]([H])([H])[H]',
        '[P+]([H])([H])[H]',
        '[PH2]',
        '[PH3]',
    ]

    # Compile all functional group patterns
    patterns = {name: Chem.MolFromSmarts(smarts) for name, smarts in functional_group_patterns.items()}

    # Compile phosphorus-hydrogen bond patterns
    p_h_patterns = [Chem.MolFromSmarts(pattern) for pattern in p_h_bond_patterns]

    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue

            # Check for any excluded functional groups
            has_excluded_group = False

            # Check regular functional groups
            for pattern_name, pattern in patterns.items():
                if pattern and mol.HasSubstructMatch(pattern):
                    has_excluded_group = True
                    break

            # If no regular functional group matched, check phosphorus-hydrogen bonds
            if not has_excluded_group:
                for p_h_pattern in p_h_patterns:
                    if p_h_pattern and mol.HasSubstructMatch(p_h_pattern):
                        has_excluded_group = True
                        break

            # If no excluded functional group, keep the molecule
            if not has_excluded_group:
                filtered_data.append({'smiles': smiles})

        except Exception:
            continue

    return pd.DataFrame(filtered_data)

def filter_phosphorus_coordination(df, smiles_column='smiles'):
    """
    Filter molecules containing [P-] with coordination number > 2

    Parameters:
    df: pandas DataFrame containing SMILES strings
    smiles_column: column name containing SMILES, default 'smiles'

    Returns:
    pandas DataFrame: filtered DataFrame
    """

    def has_phosphorus_with_high_coordination(smiles):
        """
        Check if the molecule contains [P-] with coordination number > 2

        Parameters:
        smiles: SMILES string

        Returns:
        bool: True if contains [P-] with coordination number > 2, else False
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            # Iterate over all atoms in the molecule
            for atom in mol.GetAtoms():
                # Check if it is a phosphorus atom and carries a negative charge
                if atom.GetSymbol() == 'P' :
                    # Get coordination number (number of directly connected atoms)
                    coordination_num = len(atom.GetNeighbors())
                    # If coordination number > 2, return True
                    if coordination_num > 2:
                        return True

            return False

        except Exception as e:
            print(f"Error processing SMILES: {smiles}, error: {e}")
            return False

    # Apply the filter function
    mask = df[smiles_column].apply(has_phosphorus_with_high_coordination)
    filtered_df = df[mask].copy()

    print(f"Original number of rows: {len(df)}")
    print(f"Number of rows after filtering: {len(filtered_df)}")

    return filtered_df


def filter_charge_on_P_or_O(smiles_list):
    """
    Filter molecules where the charge resides on phosphorus (P) or oxygen (O) atoms bonded to phosphorus

    Parameters:
    smiles_list: list of SMILES strings

    Returns:
    list: SMILES strings with charge on P or O bonded to P
    """
    filtered_smiles = []

    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue

            # Collect all negatively charged atoms
            charged_atoms = []
            for atom in mol.GetAtoms():
                charge = atom.GetFormalCharge()
                if charge < 0:  # Only focus on negatively charged atoms
                    charged_atoms.append(atom)

            # If there are no charged atoms, skip
            if not charged_atoms:
                continue

            # Check if any atom meets the condition
            found = False
            for atom in charged_atoms:
                symbol = atom.GetSymbol()
                if symbol == 'P':
                    found = True
                    break
                elif symbol == 'O':
                    # Check if this oxygen is directly connected to a phosphorus atom
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'P':
                            found = True
                            break
                    if found:
                        break

            if found:
                filtered_smiles.append(smiles)

        except Exception as e:
            print(f"Error processing SMILES: {smiles}, error: {e}")
            continue

    return filtered_smiles


def filter_charge_on_P_or_O_dataframe(df, smiles_column='smiles'):
    """
    Filter molecules from a DataFrame where the charge resides on phosphorus (P) or oxygen (O) atoms

    Parameters:
    df: DataFrame containing a SMILES column
    smiles_column: name of the SMILES column

    Returns:
    DataFrame: filtered DataFrame
    """
    if smiles_column not in df.columns:
        raise ValueError(f"Column {smiles_column} does not exist in the DataFrame")

    smiles_list = df[smiles_column].tolist()
    filtered_smiles = filter_charge_on_P_or_O(smiles_list)

    # Return filtered DataFrame containing all original columns
    return df[df[smiles_column].isin(filtered_smiles)]

def charge_num(smiles):
    mol = Chem.MolFromSmiles(smiles)
    total_charge = 0
    for atom in mol.GetAtoms():
        total_charge += atom.GetFormalCharge()
    return total_charge

def get_molwt(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    mol_weight = round(Descriptors.MolWt(mol),2)
    return mol_weight

def get_atom_number(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    total_atoms = mol.GetNumAtoms()
    return total_atoms

def filter_by_hetero_heavy_ratio(smiles_list, ratio_threshold=0.3):
    """
    Filter molecules where the ratio of heteroatoms to heavy atoms is ≥ threshold

    Parameters:
    smiles_list: list of SMILES strings
    ratio_threshold: threshold, default 0.35

    Returns:
    DataFrame: contains filtered molecules and related information
    """
    filtered_data = []

    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue

            # Count heavy atoms (non-hydrogen atoms)
            heavy_atoms = 0
            hetero_atoms = 0

            for atom in mol.GetAtoms():
                atomic_num = atom.GetAtomicNum()
                symbol = atom.GetSymbol()

                # If it is a hydrogen atom, skip
                if symbol == 'H':
                    continue

                # Count heavy atoms
                heavy_atoms += 1

                # Count heteroatoms (non-hydrogen atoms except C)
                if symbol != 'C':
                    hetero_atoms += 1

            # Calculate ratio
            if heavy_atoms > 0:
                hetero_ratio = hetero_atoms / heavy_atoms
            else:
                hetero_ratio = 0.0

            # Filter molecules with ratio ≥ threshold
            if hetero_ratio >= ratio_threshold:
                filtered_data.append({
                    'smiles': smiles,
                    'hetero_atoms': hetero_atoms,
                    'heavy_atoms': heavy_atoms,
                    'hetero_ratio': round(hetero_ratio, 4)
                })

        except Exception:
            continue

    return pd.DataFrame(filtered_data)

def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.RemoveHs(mol)  # Remove hydrogen atoms
        mol = Chem.AddHs(mol)     # Re-add hydrogen atoms
        mol = Chem.RemoveStereochemistry(mol)  # Optionally remove stereochemistry
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        pass
    return smiles

def filter_molecules_by_functional_groups(smiles_list):
    """
    Filter molecules that meet the functional group conditions:

    Parameters:
    smiles_list: list of SMILES strings

    Returns:
    list: list of SMILES strings that meet the conditions
    """
    # Define SMARTS patterns for functional groups
    smarts_patterns = {
        'F': '[F]',  # Fluorine atom
        'CN': 'C#N',  # Cyano group
        'NO2': 'N(=O)=O'  # Nitro group (more precise pattern)
    }

    # Compile SMARTS patterns
    patterns = {}
    for name, smarts in smarts_patterns.items():
        try:
            patterns[name] = Chem.MolFromSmarts(smarts)
        except:
            print(f"Warning: Unable to compile SMARTS pattern: {smarts} for {name}")
            patterns[name] = None

    # Alternate nitro pattern (if the above fails)
    if patterns['NO2'] is None:
        # Try other common nitro representations
        nitro_patterns = ['[N+](=O)[O-]', 'N(=O)=O', 'N(=O)(=O)']
        for pattern in nitro_patterns:
            try:
                patterns['NO2'] = Chem.MolFromSmarts(pattern)
                if patterns['NO2']:
                    break
            except:
                continue

    filtered_smiles = []

    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue

            # Count each functional group
            counts = {}
            for name, pattern in patterns.items():
                if pattern:
                    counts[name] = len(mol.GetSubstructMatches(pattern))
                else:
                    counts[name] = 0

            condition = counts['F'] >= 1 or counts['CN'] >= 1 or counts['NO2'] >= 1

            if condition:
                filtered_smiles.append(smiles)

        except Exception as e:
            print(f"Error processing SMILES: {smiles}, error: {e}")
            continue

    return filtered_smiles

def filter_from_dataframe(df, smiles_column='smiles'):
    """
    Filter molecules from a DataFrame

    Parameters:
    df: DataFrame containing a SMILES column
    smiles_column: name of the SMILES column

    Returns:
    DataFrame: filtered DataFrame
    """
    if smiles_column not in df.columns:
        raise ValueError(f"Column {smiles_column} does not exist in the DataFrame")

    smiles_list = df[smiles_column].tolist()
    filtered_smiles = filter_molecules_by_functional_groups(smiles_list)

    # Return filtered DataFrame containing all original columns
    return df[df[smiles_column].isin(filtered_smiles)]

# 1. Merge data
rawdata1 = pd.read_csv('./data/molecules1.csv')
df1= pd.DataFrame(rawdata1)
df1['smiles'] = df1['smiles'].apply(standardize_smiles)
rawdata2 = pd.read_csv('./data/molecules2.csv')
df2 = pd.DataFrame(rawdata2)
df2['smiles'] = df2['smiles'].apply(standardize_smiles)
smiles_set = set(df1['smiles']).union(set(df2['smiles']))
unique_smiles_list = list(smiles_set)
df = pd.DataFrame({'smiles': unique_smiles_list})
df['smiles'] = df['smiles'].apply(extract_anion_with_rdkit)
# 2. Keep specific elements
df['smiles'] = pd.DataFrame(filter_molecules_by_elements(df['smiles']))
# 3. Remove functional groups like -OH
df['smiles'] = pd.DataFrame(filter_molecules_without_functional_groups(df['smiles']))
df['smiles'] = df['smiles'].apply(standardize_nitro_smiles)
# 4. Remove hydrates, etc.
df['is_valid'] = df['smiles'].apply(is_valid)
df = df[df['is_valid'] == True]
# 5. Molecular weight 60-1000, anion has exactly one negative charge
df['anion_MW'] = df['smiles'].apply(get_molwt)
df['charge'] = df['smiles'].apply(lambda x: pd.Series(charge_num(x)))
df['atom_number'] = df['smiles'].apply(get_atom_number)
df_clean = df[(df['anion_MW'] >=60 ) & (df['anion_MW'] <= 1000 ) & (df['charge'] == -1) & (df['atom_number'] <= 55)]
df_sorted = df_clean.sort_values('anion_MW')

# 6. Apply functional group filtering (get filtered data)
print("\nStep 3: Applying functional group filtering...")
filtered_by_func = filter_from_dataframe(df_sorted, smiles_column='smiles')

# 7. Apply charge location filtering (get filtered data)
filtered_by_charge_location = filter_charge_on_P_or_O_dataframe(filtered_by_func, smiles_column='smiles')

# 8. Apply heteroatom to heavy atom ratio filtering
print("\nStep 10: Applying heteroatom to heavy atom ratio filtering...")
if len(filtered_by_charge_location) > 0:
    # Use heteroatom ratio filtering
    filtered_by_hetero_ratio = filter_by_hetero_heavy_ratio(
        filtered_by_charge_location['smiles'].tolist(),
        ratio_threshold=0.3
    )

    print(f"Number of rows after heteroatom ratio filtering: {len(filtered_by_hetero_ratio)}")

    # Merge detailed information
    if len(filtered_by_hetero_ratio) > 0:
        # Merge filtering results into original data
        final_filtered = pd.merge(
            filtered_by_charge_location,
            filtered_by_hetero_ratio[['smiles']],
            on='smiles',
            how='inner'
        )
        final_filtered = final_filtered.drop_duplicates(subset=['smiles'])
filtered_df1 = filter_phosphorus_coordination(final_filtered, smiles_column='smiles')
filtered_df1.to_csv('./data/molecules_filtered.csv', index=False)