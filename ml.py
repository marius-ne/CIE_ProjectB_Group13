import pandas as pd
import numpy as np

import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Own imports
from constants import *
from utils import *


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



def apply_bridge_pca(df, n_components=10):
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
    df_pivot = reshape_multi_variable_to_wide(df)
    
    # 2. Separate the features (Stress values) from the labels
    # The stress values start after the index columns we defined above
    metadata_cols = ['health', 'scenario', 'time']
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


def filter_df_to_delta_nodes(df, variable_names=VARIABLE_NAMES, top_pct=1):
    """
    Mark delta nodes instead of rejecting nodes.

    - Adds a column "delta_health": 0 where node is a delta (unhealthy), 1 otherwise.
    - Adds (optionally) a "scenario_combination" column if a mapping function
        `map_scenario_to_combination` is available in globals(); otherwise the original
        scenario value is copied into that column.
    - Delta determination: for each (train_config, load, season) base scenario, use the
        health==0 rows as the healthy baseline and compare each damaged level (1..6)
        to that baseline using get_variable_difference_between_dataframes for each
        variable in variable_names. Any node flagged as delta for any variable is
        marked delta (delta_health=0) for all rows matching that damaged scenario.
    - Returns the full dataframe (no rows removed), the delta_nodes dict and diffs dict.
    """
    # default: all healthy
    df['delta_health'] = 1

    # Dummy 
    dummy = 0

    delta_nodes = {}
    diffs = {}

    # build set of unique base scenarios (train_config, load, season)
    scenarios = df['scenario'].unique()
    combos = [scenario_number_to_combination(s) for s in scenarios]
    base_scenarios = {combo[:3] for combo in combos}

    for base in base_scenarios:
        # healthy scenario number and dataframe
        healthy_scenario_num = combination_to_scenario_number((*base, 0, dummy))
        df_healthy = df[df['scenario'] == healthy_scenario_num]
        if df_healthy.empty:
            # no healthy baseline -> skip this base scenario
            continue

        # iterate damage levels
        for damage_level in range(1, 7):
            damaged_scenario_num = combination_to_scenario_number((*base, damage_level, dummy))
            df_damaged = df[df['scenario'] == damaged_scenario_num]
            if df_damaged.empty:
                continue

            nodes_for_damage = set()

            for var_idx, var_name in enumerate(variable_names):
                df_diff = get_variable_difference_between_dataframes(
                    df_healthy, df_damaged, var_name=var_name, top_pct=top_pct
                )

                key = (*base, damage_level, var_idx)
                diffs[key] = df_diff

                # determine nodes flagged as delta for this variable
                if var_name in df_diff.columns:
                    nodes = df_diff[~df_diff[var_name].isna()]["Node Number"].unique()
                else:
                    # fallback: assume last column contains the diff indicator
                    nodes = df_diff[~df_diff.iloc[:, -1].isna()]["Node Number"].unique()

                delta_nodes[key] = nodes
                nodes_for_damage.update(nodes.tolist())

            # mark all rows for this damaged scenario & node as delta (0)
            if nodes_for_damage:
                mask = (
                    (df['scenario'] == damaged_scenario_num) &
                    (df['health'] == damage_level) &
                    (df['Node Number'].isin(nodes_for_damage))
                )
                df.loc[mask, 'delta_health'] = 0

    # ensure healthy rows remain healthy (explicit)
    df.loc[df['health'] == 0, 'delta_health'] = 1

    return df, delta_nodes, diffs


def get_delta_nodes_from_df(
    df, 
    healthy_scenarios: list, 
    damaged_scenarios: list, 
    variable_names=VARIABLE_NAMES, 
    top_pct: float = 0.01
):
    """
    Computes delta nodes for each healthy scenario and its corresponding damaged scenarios from a single DataFrame.

    Args:
        df: DataFrame containing all scenarios, damage levels, and variables.
        healthy_scenarios: List of healthy scenario identifiers.
        damaged_scenarios: Nested list of damaged scenario identifiers corresponding to each healthy scenario.
        variable_names: List of variable names to compare.
        top_pct: Percentage of top delta nodes to select.

    Returns:
        delta_nodes: Dictionary with keys (healthy_scenario, damaged_scenario, variable) and values as arrays of delta node numbers.
        diffs: Dictionary with keys (healthy_scenario, damaged_scenario, variable) and values as DataFrames of differences.
    """
    delta_nodes = {}
    diffs = {}

    for h_idx, healthy in enumerate(healthy_scenarios):
        df_healthy = df[df['scenario'] == healthy]
        for damaged in damaged_scenarios[h_idx]:
            df_damaged = df[df['scenario'] == damaged]
            for var_idx, var_name in enumerate(variable_names):
                df_diff = get_variable_difference_between_dataframes(
                    df_healthy, df_damaged, var_name=var_name, top_pct=top_pct
                )
                key = (healthy, damaged, var_idx)
                delta_nodes[key] = df_diff[~df_diff[var_name].isna()]["Node Number"].unique()
                diffs[key] = df_diff
    return delta_nodes, diffs


def get_delta_nodes(top_pct: float = 0.01, drop_invalid_nodes: bool = False):
    """
    Obtain delta nodes for all scenarios, damage levels, and variables.
    Returns:
        delta_nodes: A dictionary with keys as (train_config, load, season, damage, variable)
                     and values as arrays of delta node numbers.
        diffs: A dictionary with keys as (train_config, load, season, damage, variable)
               and values as DataFrames containing the differences.
    """
    diffs = {}
    delta_nodes = {}
    # Iterate over all (train_config, load, season) combinations - "scenarios"
    for combo in combinations_grouped_by_region:
        scenario = combo[0][0][:3]
        df_healthy = get_data_variable_aggregated((*scenario, 0), drop_invalid_nodes=drop_invalid_nodes)

        # Iterate over damage levels for the given scenario
        for damage in range(1, 7):
            df_damaged = get_data_variable_aggregated((*scenario, damage), drop_invalid_nodes=drop_invalid_nodes)
            # Check delta for each variable between the healthy and the current damaged state
            for variable in range(8):
                var_name = VARIABLE_NAMES[variable]
                df_diff = get_variable_difference_between_dataframes(
                    df_healthy, df_damaged, var_name=var_name, top_pct=top_pct
                )
                # Delta nodes: those where the difference is not NaN
                delta_nodes[(*scenario, damage, variable)] = df_diff[~df_diff[var_name].isna()]["Node Number"].unique()
                diffs[(*scenario, damage, variable)] = df_diff
        print(f"There are {len(set(np.concatenate(list(delta_nodes.values()))))} delta nodes for scenario {scenario} and variable {variable}")
        print(f"Delta nodes for {scenario}: {[f'{i+1}:{len(dn)}' for i, dn in enumerate(delta_nodes.values())]}")

    return delta_nodes, diffs


def create_multichannel_voxel_dataset(df_data, grid_size=(60, 20, 20)):
    """
    Converts bridge data into 3D volumes with 8 physical channels.
    """
    # Use the 8 variables defined in your constants.py
    value_cols = VARIABLE_NAMES 
    
    # 1. Merge Coordinates
    if 'X' not in df_data.columns:
        raise ValueError("Data must contain node coordinates")
    else:
        df_full = df_data.copy()

    # 2. Discretize coordinates into grid indices
    # We use pd.cut to map physical X,Y,Z to 0 -> grid_size-1
    for axis, size in zip(['X', 'Y', 'Z'], grid_size):
        col_name = f'{axis.lower()}_idx'
        df_full[col_name] = pd.cut(df_full[axis], bins=size, labels=False, include_lowest=True)

    # 3. Process snapshots
    samples = []
    labels = []
    
    # Each scenario + time step is a unique 3D state
    grouped = df_full.groupby(['scenario', 'time'])
    
    for (scenario, time), group in grouped:
        # Initialize grid: (Length, Height, Width, Channels)
        vol = np.zeros(grid_size + (len(value_cols),))
        
        for i, col in enumerate(value_cols):
            # Aggregate: Take max stress/deformation in each voxel
            voxel_map = group.groupby(['x_idx', 'y_idx', 'z_idx'])[col].max().dropna()
            
            for index, val in voxel_map.items():
                # CORRECT INDEXING: Concatenate the 3D index tuple with the channel index
                # (x, y, z) + (channel,) -> (x, y, z, channel)
                vol[index + (i,)] = val
        
        samples.append(vol)
        labels.append(group['health'].iloc[0])

    X_vol = np.array(samples)
    y_binary = (np.array(labels) > 0).astype(int)
    
    return X_vol, y_binary


def compute_node_fft(df, node_number, scenario_id=1, variable='TotalDeformation'):
    """
    Computes FFT for a specific node and scenario, returns frequencies and magnitudes.

    Args:
        df: DataFrame containing 'scenario', 'time', 'Node Number' and physics columns.
        node_number: The ID of the node to analyze.
        scenario_id: The health/scenario identifier.
        variable: The physical variable to analyze.

    Returns:
        frequencies: Array of frequency bins.
        magnitude: Array of FFT magnitudes.
    """
    subset = df[(df['scenario'] == scenario_id) & (df['Node Number'] == node_number)].sort_values('time')
    if subset.empty:
        return None, None

    time = subset['time'].values
    signal = subset[variable].values

    dt = np.mean(np.diff(time))
    n = len(signal)

    fft_values = np.fft.rfft(signal - np.mean(signal))
    frequencies = np.fft.rfftfreq(n, d=dt)
    magnitude = np.abs(fft_values)

    return frequencies, magnitude


def compare_bridge_dynamics(df, node_number, healthy_scenario=1, damaged_scenario=0, variable='TotalDeformation'):
    """
    Computes FFT for a specific node to compare vibrations between healthy and damaged states.
    
    Args:
        df: Your main dataframe containing 'scenario', 'time', 'Node Number' and physics columns.
        node_number: The ID of the node to analyze (choose a mid-span node for best results).
        variable: The physical variable to analyze (TotalDeformation or DirectionalDeformation_Y_axis).
    """
    
    plt.figure(figsize=(15, 6))
    
    for scenario_id, label, color in [(healthy_scenario, 'Healthy', 'blue'), (damaged_scenario, 'Damaged', 'red')]:
        # 1. Extract the time series for this specific node/scenario
        subset = df[(df['scenario'] == scenario_id) & (df['Node Number'] == node_number)].sort_values('time')
        
        if subset.empty:
            print(f"No data for Node {node_number} in Scenario {scenario_id}")
            continue
                    
        time = subset['time'].values
        signal = subset[variable].values
        
        # 2. Calculate Sampling Frequency (fs)
        dt = np.mean(np.diff(time))
        fs = 1.0 / dt
        n = len(signal)
        
        # 3. Perform FFT
        # We subtract the mean (detrending) to ignore the static load and focus on the vibration
        fft_values = np.fft.rfft(signal - np.mean(signal))
        frequencies = np.fft.rfftfreq(n, d=dt)
        magnitude = np.abs(fft_values)
        
        # 4. Plotting the Spectrum
        plt.plot(frequencies, magnitude, label=f"{label} (Scenario {scenario_id})", color=color, alpha=0.8)
        
        # Find peak frequency
        peak_idx = np.argmax(magnitude)
        print(f"[{label}] Peak Frequency: {frequencies[peak_idx]:.3f} Hz")

    plt.title(f"Frequency Spectrum Comparison at Node {node_number} ({variable})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (Energy)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Bridges usually have low natural frequencies (0.5Hz to 20Hz)
    # We zoom in on this range to see the "Mode Shifts"
    plt.xlim(0, 30) 
    plt.show()

