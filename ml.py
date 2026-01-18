import pandas as pd
import numpy as np

import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Own imports
from utils import reshape_long_to_pca_ready, reshape_multi_variable_to_pca


def standardize(X_raw):
    scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X_raw)

    df_scaled = pd.DataFrame(
        X_scaled, 
        columns=X_raw.columns if hasattr(X_raw, 'columns') else None, 
        index=X_raw.index if hasattr(X_raw, 'index') else None
    )

    return df_scaled


def extract_modal_features(df_scenario, n_modes=5):
    """
    Reduces the 25,000 nodes into N dominant modal coefficients.
    
    Args:
        df_scenario: DataFrame where columns are time steps and rows are nodes.
        n_modes: Number of dominant mode shapes to extract.
    """
    # 1. Prepare the matrix (Nodes x Time)
    # We drop 'Node Number' and focus only on the stress values across time
    matrix = df_scenario.drop(columns=['Node Number']).values
    
    # 2. Initialize PCA (POD equivalent)
    # We want to find the patterns that explain the most variance across the bridge
    pca = PCA(n_components=n_modes)
    
    # 3. Fit and Transform
    # This reduces the 25,000 node dimensions to n_modes
    modal_coefficients = pca.fit_transform(matrix.T) # Transpose to get Time-wise modes
    
    # 4. Extract the 'Energy' of each mode (The Eigenvalues)
    # These represent the 'Structural Fingerprint'
    mode_energies = pca.explained_variance_ratio_
    
    # 5. Extract the Mode Shapes (The Components)
    # These tell you 'where' the bridge is reacting for each mode
    mode_shapes = pca.components_
    
    return mode_energies, mode_shapes



def apply_bridge_pca(df, n_components=10, one_hot_encoded: bool = False):
    """
    Transforms the long-format bridge data into a reduced PCA feature set.
    
    Args:
        df: The dataframe with columns [Node Number, health, load, EquivalentStress, etc.]
        n_components: How many 'Principal Components' to keep.
        
    Returns:
        X_pca: The reduced feature matrix (Ready for Classifier)
        pca_model: The fitted PCA object (To transform test data later)
    """
    # 1. Pivot the data so each row is a unique bridge state (Scenario)
    # and each column is the stress at a specific Node Number
    # Note: If you have multiple time steps, this will flatten them into features
    df_pivot = reshape_multi_variable_to_pca(df, one_hot_encoded=one_hot_encoded)
    
    # 2. Separate the features (Stress values) from the labels
    # The stress values start after the index columns we defined above
    if one_hot_encoded:
        metadata_cols = ['health', 'region_0', 'region_1', 'region_2', 'region_3', 'region_4', 'region_5',
                         'load_0', 'load_1', 'season_0', 'season_1',
                         'train_config_0', 'train_config_1', 'train_config_2', 'train_config_3',
                         'time']
    else:
        metadata_cols = ['health', 'region', 'load', 'season', 'train_config', 'time']
    metadata_cols = [col for col in df_pivot.columns if col in metadata_cols]
    X_raw = df_pivot.drop(columns=metadata_cols).values
    print(X_raw.shape)
    y = df_pivot['health'] # Use health as the target for the classifier later
    
    # 3. Standardize the data (PCA is sensitive to scale!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # 4. Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pd.DataFrame(
        pca.fit_transform(X_scaled),
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=df_pivot.index
    )
    # Add back the original metadata columns to the reduced feature set
    for col in metadata_cols:
        X_pca[col] = df_pivot[col].values
    X_pca.drop(columns=["health"], inplace=True) 
    
    print(f"Original features: {X_raw.shape}")
    print(f"Reduced features: {X_pca.shape}")
    print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.2%}")
    
    return X_pca, y, pca, scaler


