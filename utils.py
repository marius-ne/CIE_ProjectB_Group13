import itertools
import functools
from pathlib import Path

import pandas as pd
import numpy as np
from constants import *


def combination_to_string(combination):
    train_config, load, season, region, variable = combination
    if DATA_FORMAT == "old":
        string = f"{TRAIN_CONFIGS[train_config]}__{LOADS[load]}__{SEASONS[season]}__{REGIONS[region]}__{VARIABLE_NAMES[variable]}"
    else:
        string = f"{TRAIN_CONFIGS[train_config]}__{SEASONS[season]}__{LOADS[load]}__{REGIONS[region]}__{VARIABLE_NAMES[variable]}"
    return string

def combination_to_scenario_number(combination):
    """
    Encodes a combination of (train_config, load, season, region, variable)
    into a unique scenario number using mixed radix encoding.
    The variable is ignored.
    Returns:
        int: Unique scenario number.
    """
    train_config, load, season, region, variable = combination
    scenario_number = (
        (((train_config * len(LOADS) + load) * len(SEASONS) + season) * len(REGIONS) + region) 
    )
    return scenario_number

# Construct list of scenarios (combinations of train configs, loads, seasons, healths, variables)
#   Each scenario is a tuple of (train_config, load, season, health, variable), each encoded
#   as the corresponding key in the dictionaries above
combinations = itertools.product(
        TRAIN_CONFIGS.keys(),
        LOADS.keys(),
        SEASONS.keys(),
        REGIONS.keys(),
        VARIABLES.keys()
    )
combinations = list(combinations)

# Group by variables, i.e. each group has all variables for one scenario
combinations_grouped_by_variable = [
    combinations[i:i + len(VARIABLES)] for i in range(0, len(combinations), len(VARIABLES))
]
# Group further by healths, i.e. each group has all healths for one scenario (train config, load, season)
combinations_grouped_by_region = [
    combinations_grouped_by_variable[i:i + len(REGIONS)] for i in range(0, len(combinations_grouped_by_variable), len(REGIONS))
]

def add_node_locations(df):
    """
    Adds X, Y, Z coordinates to a stress dataframe based on Node Number.

    Args:
        df (pd.DataFrame): The dataframe containing a 'Node Number' column.
        coords_source (str or pd.DataFrame): The path to 'nodeExport.txt' or
                                              a DataFrame already containing
                                              ['Node Number', 'X', 'Y', 'Z'].

    Returns:
        pd.DataFrame: The original dataframe with X, Y, and Z columns added.
    """

    # Merge the coordinates onto the main dataframe
    # We use 'left' join to keep all rows in your original data
    df['Node Number'] = df['Node Number'].astype(int)

    df_with_coords = pd.merge(df, COORDS_DF[['Node Number', 'X', 'Y', 'Z']],
                              on='Node Number',
                              how='left')

    # Check if any nodes failed to find a coordinate
    # missing_count = df_with_coords['X'].isna().sum()
    # if missing_count > 0:
        # print(f"Warning: {missing_count} nodes did not have matching coordinates.")

    return df_with_coords


def read_data_file(
    train_config: int = 0,
    load: int = 0,
    season: int = 0,
    region: int = 0,
    variable: int = 0,
    filter_out_invalid_nodes: bool = True,
    melt_time: bool = True,
):
    """Reads data according to format and provides the data-frame as-is, with
    the categorical variables added as columns."""

    combination = (train_config, load, season, region, variable)
    print("Reading file:", combination_to_string(combination))

    # Construct filename from scenario according to the folder structure
    results_paths = ["Results", "Results1"]
    base_path = Path(DATA_FOLDER_PATH)


    for results_path in results_paths:
        if DATA_FORMAT == "old":
            filename = base_path
            filename /= TRAIN_CONFIGS[train_config]
            filename /= LOADS[load]
            filename /= SEASONS[season]
            filename /= REGIONS[region]
            filename /= results_path
            filename /= VARIABLES[variable]
            filename = filename.with_suffix(".csv")
        else:
            filename = base_path
            filename /= TRAIN_CONFIGS[train_config]
            filename /= SEASONS[season]
            filename /= LOADS[load]
            filename /= REGIONS[region]
            filename /= results_path
            filename /= VARIABLES[variable]
            filename = filename.with_suffix(".csv")

        if filename.exists():
            break
    else:
        raise FileNotFoundError(f"Data file not found for: {str(filename)}")


    # Construct the expected filename string for comparison
    expected_filename_str = combination_to_string(combination) + ".csv"

    # Build the actual filename string from the path components
    actual_filename_str = VARIABLES[variable] + ".csv"

    # Check that the constructed string matches the filename
    if expected_filename_str.split("__")[-1] != actual_filename_str:
        raise ValueError(f"Variable name mismatch: expected {expected_filename_str}, got {actual_filename_str}")


    # Create a unique scenario number from the combination using mixed radix encoding
    # Each category uses only as many digits as needed for its range
    scenario_number = combination_to_scenario_number(combination)
        
    # Encode the scenario as categorical columns
    #   -> TODO: Is there a way of encoding that
    #   preserves information? E.g. like encoding the name of a city as its latitude
    df = pd.read_csv(filename)
    num_nodes = len(df)
    healthy = 1 if region == 0 else 0
    df["health"] = healthy*np.ones(num_nodes,dtype=np.uint8)
    df["scenario"] = scenario_number*np.ones(num_nodes,dtype=np.uint32)
    # df["season"] = season*np.ones(num_nodes,dtype=np.uint8)
    # df["region"] = region*np.ones(num_nodes,dtype=np.uint8)
    # df["load"] = load*np.ones(num_nodes,dtype=np.uint8)
    # df["train_config"] = train_config*np.ones(num_nodes,dtype=np.uint8)

    df = add_node_locations(df)

    # Identify nodes with missing coordinates
    if filter_out_invalid_nodes:
        
        df = df[df["Node Number"].isin(VALID_NODE_NUMBERS)]

        # Put stress to 0
        # Identify nodes with missing stress values
        stress_variables = list(ORIGINAL_VARIABLES.values())[4:] 
        stress_cols = [col for col in df.columns if any(var in col for var in stress_variables)]
        df.loc[df["Node Number"].isin(NODES_MISSING_STRESS), stress_cols] = \
            df.loc[df["Node Number"].isin(NODES_MISSING_STRESS), stress_cols].fillna(0)

        # Check that all nodes have coordinates and loads
        missing_coords_nodes = df[df[["X", "Y", "Z"]].isnull().any(axis=1)]["Node Number"].unique()
        if len(missing_coords_nodes) > 0:
            raise ValueError(f"{len(missing_coords_nodes)} nodes are missing coordinates after filtering.")
        missing_loads_nodes = df[df.isnull().any(axis=1)]["Node Number"].unique()
        if len(missing_loads_nodes) > 0:
            raise ValueError(f"{len(missing_loads_nodes)} nodes are missing load data after filtering.")

    # Melt time columns into a single column
    if melt_time:
        var_name = VARIABLE_NAMES[variable]

        # Turn the variable column into a single one and add a new time column
        df_melted = df.melt(
            id_vars=["Node Number","health","scenario","X","Y","Z",],
            var_name="variable",
            value_name=var_name
        )
        # Check that variable name matches original variable names
        assert all(df_melted["variable"].str[:-4] == ORIGINAL_VARIABLES[variable])

        # Extract time-stamp from variable name
        df_melted["time"] = df_melted["variable"].str[-3:].astype(np.float64)
        df_melted.drop(columns=["variable"],inplace=True)

        # Check that data contains all valid node numbers
        missing_nodes = set(VALID_NODE_NUMBERS) - set(df_melted["Node Number"].unique())
        if len(missing_nodes) > 0 and filter_out_invalid_nodes:
            raise ValueError(f"Data for variable {var_name} is missing node numbers: {missing_nodes}")

        df = df_melted

    return df


def get_variable_difference_between_combinations(comb1: tuple, comb2: tuple, top_pct: float = 1.0):
    """
    Computes the difference in the variable between two combinations.
    Args:
        comb1, comb2 (tuple): Each a (train_config, load, season, region, variable).
    Returns:
        pd.DataFrame: DataFrame with Node Number, time, X, Y, Z, and the difference in the variable.
    """
    # Ensure both combinations refer to the same variable
    if comb1[-1] != comb2[-1]:
        raise ValueError("Both combinations must refer to the same variable.")

    var_name = VARIABLE_NAMES[comb1[-1]]

    # df1 = read_data_file(*comb1, filter_out_invalid_nodes=filter_out_invalid_nodes)
    # df2 = read_data_file(*comb2, filter_out_invalid_nodes=filter_out_invalid_nodes)
    
    # NOTE: We now get the aggregated data instead to get rid of the invalid deformations
    df_agg1 = get_data_variable_aggregated(comb1[:-1])
    df_agg2 = get_data_variable_aggregated(comb2[:-1])

    df1 = select_df_subset(df_agg1, combination_to_scenario_number(comb1))
    df2 = select_df_subset(df_agg2, combination_to_scenario_number(comb2))

    return get_variable_difference_between_dataframes(df1, df2, var_name, top_pct=top_pct)


def get_variable_difference_between_dataframes(df1, df2, var_name: str, top_pct: float = 1.0):
    """
    Computes the difference in the variable between two data_frames and keeps only the top nodes
    by per-node max absolute difference over time.
    
    Args:
        df1, df2 (pd.DataFrame): Each a data-frame containing the same variable.
        top_pct (float):
            - If 0 < top_pct <= 1.0: keep the top fraction of nodes (ceil(top_pct * num_nodes_with_nonzero_diff)).
            - If top_pct > 1.0: keep the top N nodes.
    Returns:
        pd.DataFrame: DataFrame with Node Number, time, X, Y, Z, and the difference in the variable.
                      Rows from non-selected nodes have the variable set to NaN.
    """
    # Ensure both dataframes have the same columns
    if set(df1.columns) != set(df2.columns):
        raise ValueError(f"DataFrames have different columns: {set(df1.columns) ^ set(df2.columns)}")

    # Keep only relevant columns
    keep_cols = ["Node Number", "time", "X", "Y", "Z", var_name]
    df1 = df1[keep_cols]
    df2 = df2[keep_cols]

    # Merge on metadata
    merge_cols = ["Node Number", "time", "X", "Y", "Z"]
    merged = pd.merge(df1, df2, on=merge_cols, suffixes=('_1', '_2'), how='inner')

    # Safety check
    if not (merged[f"{var_name}_1"].shape == merged[f"{var_name}_2"].shape and merged[f"{var_name}_1"].index.equals(merged[f"{var_name}_2"].index)):
        raise ValueError("DataFrames to subtract do not have matching shapes or indices.")

    # Drop rows where either side is NaN for the target variable
    valid_rows = merged[f"{var_name}_1"].notna() & merged[f"{var_name}_2"].notna()
    merged = merged[valid_rows].copy()
    if merged.empty:
        merged[var_name] = np.nan
        return merged
    
    # Compute difference
    merged[var_name] = merged[f"{var_name}_1"] - merged[f"{var_name}_2"]
    abs_diff = np.abs(merged[var_name])

    # Per-node aggregate (max over time)
    per_node_max = abs_diff.groupby(merged["Node Number"]).max()

    # Consider only nodes with any non-zero difference
    nonzero_nodes = per_node_max[per_node_max > 0]
    if nonzero_nodes.empty:
        merged[var_name] = np.nan
        return merged

    # Determine how many nodes to keep
    if 0 < top_pct <= 1.0:
        k = int(np.ceil(top_pct * len(nonzero_nodes)))
        k = max(1, k)
    else:
        k = int(top_pct)
        k = max(1, min(k, len(nonzero_nodes)))

    # Select top nodes by max abs diff
    top_nodes = nonzero_nodes.sort_values(ascending=False).head(k).index

    # Mask out non-top nodes
    keep_mask = merged["Node Number"].isin(top_nodes)
    merged.loc[~keep_mask, var_name] = np.nan

    return merged


def get_data_variable_aggregated(
    scenario_combination: tuple,
    filter_out_invalid_nodes: bool = True,
    drop_invalid_nodes: bool = False,
):
    """
    Reads all data files from one scenario of load, train_config, season, and region and takes all variables
    and merges them into a single data-frame.
    Args:
        scenario_combination (tuple): A tuple of (train_config, load, season, region) representing
            the scenario for which to aggregate data across all variables.
    Returns:
        pd.DataFrame: Merged data-frame with all variables as columns.
    """

    # Create all possible variable-combinations for that scenario
    combinations_grouped_by_variable = [
       [*scenario_combination,i] for i in VARIABLES.keys()
    ]

    dfs_variables = []
    #  Merge data for all variables in the current health scenario
    for same_variable_combination in combinations_grouped_by_variable:

        # Get data file for current combination
        df = read_data_file(*same_variable_combination, filter_out_invalid_nodes=filter_out_invalid_nodes)

        dfs_variables.append(df)

    # Concatenating all variables into a single data frame
    # -> we do an OUTER join, meaning all keys are kept (A U B)
    #   this should be safe, node numbers and the other shared columns are kept
    shared_cols = ["Node Number","health","scenario","X","Y","Z","time"]
    df_vars = functools.reduce(lambda left,right: pd.merge(left,right,on=shared_cols,
                                              how='outer'), dfs_variables)
    # Check that data has been preserved
    for df in dfs_variables:
        for var_name in VARIABLE_NAMES:
            if var_name in df.columns:
                  merged = pd.merge(df[shared_cols + [var_name]], df_vars[shared_cols + [var_name]],
                            on=shared_cols, how='inner')
                  assert len(merged) == len(df)

    # Filter out deformations where the total deformation is negative or
    #   where the root of the sum of squares of the directional deformations is different
    #   from the total deformation by a large margin
    if filter_out_invalid_nodes and "TotalDeformation" in df_vars.columns:
        valid_deformation_mask = (
            (df_vars["TotalDeformation"] >= 0) &
            (np.abs(
                df_vars["TotalDeformation"] - np.sqrt(
                    df_vars["DirectionalDeformation_X_axis"]**2 +
                    df_vars["DirectionalDeformation_Y_axis"]**2 +
                    df_vars["DirectionalDeformation_Z_axis"]**2
                )
            ) <= 1e-3)
        )

        # NOTE ENABLE THIS ONLY FOR GETTING DELTA NODES:
        # Drop invalid nodes instead of setting them to 0 / NaN
        df_vars["valid_deformation"] = valid_deformation_mask.astype(np.uint8)
        if drop_invalid_nodes:
            df_vars = df_vars[valid_deformation_mask].copy()
        else:
            # mark invalid deformation values as NaN to be ignored downstream
            invalid_cols = [
                "TotalDeformation",
                "DirectionalDeformation_X_axis",
                "DirectionalDeformation_Y_axis",
                "DirectionalDeformation_Z_axis",
            ]
            df_vars.loc[~valid_deformation_mask, invalid_cols] = np.nan
            
    # Re-order columns
    df_vars = df_vars[shared_cols + VARIABLE_NAMES]

    return df_vars


def get_data_variable_and_region_aggregated(
    scenario_combination: tuple
):
    """
    Reads all data files from one scenario of load, train_config and season and takes all health groups
    and variables and merges them into a single data-frame.

    Args:
        scenario_combination (tuple): A tuple of (train_config, load, season) representing
            the scenario for which to aggregate data across all healths and variables.
    Returns:
        pd.DataFrame: Merged data-frame with all variables as columns.
    """
    train_config, load, season = scenario_combination

    # Merge data for all healths in the scenario
    dfs_regions = []
    for region in REGIONS.keys():

        df_vars = get_data_variable_aggregated((train_config, load, season, region))

        dfs_regions.append(df_vars)

    # Check that columns are the same
    assert all(all(df.columns == dfs_regions[0].columns) for df in dfs_regions)

    # Concatenate them together
    df = pd.concat(dfs_regions,ignore_index=True)

    return df


def get_data_variable_and_region_and_season_aggregated(
    scenario_combination: tuple
):
    """
    Reads all data files from one scenario of train_config and load, and aggregates across all seasons, regions, and variables.

    Args:
        scenario_combination (tuple): A tuple of (train_config, load) representing
            the scenario for which to aggregate data across all seasons, regions, and variables.
    Returns:
        pd.DataFrame: Merged data-frame with all variables as columns for all seasons and regions.
    """
    train_config, load = scenario_combination
    all_dfs = []
    for season in SEASONS.keys():
        for region in REGIONS.keys():
            try:
                df = get_data_variable_aggregated((train_config, load, season, region))
                all_dfs.append(df)
            except FileNotFoundError:
                print(f"Skipping missing file for scenario: {combination_to_string((train_config, load, season, region, 0))}")
                continue

    if not all_dfs:
        raise RuntimeError("No data files found for any season/region in this scenario.")
        
    df_all = pd.concat(all_dfs, ignore_index=True)
    return df_all


def get_data_variable_and_region_and_season_and_load_aggregated(
    scenario_combination: tuple
):
    """
    Reads all data files from one scenario of train_config and aggregates across all loads, seasons, regions, and variables.

    Args:
        scenario_combination (tuple): A tuple of (train_config,) representing
            the scenario for which to aggregate data across all loads, seasons, regions, and variables.
    Returns:
        pd.DataFrame: Merged data-frame with all variables as columns for all loads, seasons, and regions.
    """
    train_config = scenario_combination[0]
    all_dfs = []
    for load in LOADS.keys():
        for season in SEASONS.keys():
            for region in REGIONS.keys():
                try:
                    df = get_data_variable_aggregated((train_config, load, season, region))
                    all_dfs.append(df)
                except FileNotFoundError:
                    print(f"Skipping missing file for scenario: {combination_to_string((train_config, load, season, region, 0))}")
                    continue

    if not all_dfs:
        raise RuntimeError("No data files found for any load/season/region in this scenario.")

    df_all = pd.concat(all_dfs, ignore_index=True)
    return df_all


def get_data_all_aggregated(drop_invalid_nodes: bool = False):
    """
    Reads all data files for all combinations of season, load, train_config, health, and variable,
    and merges them into a single DataFrame.
    Returns:
      pd.DataFrame: Merged data-frame with all variables as columns for all scenarios.
    """
    all_dfs = []
    # Iterate over all combinations of season, load, train_config, and health
    for train_config in TRAIN_CONFIGS.keys():
        for load in LOADS.keys():
            for season in SEASONS.keys():
                for region in REGIONS.keys():
                    scenario = (train_config, load, season, region)
                    # get_data_variable_aggregated expects a 4-tuple (train_config, load, season, health)
                    try:
                        df_vars = get_data_variable_aggregated(scenario, drop_invalid_nodes=drop_invalid_nodes)
                        all_dfs.append(df_vars)
                    except FileNotFoundError as e:
                        raise ValueError(f"Skipping missing file for scenario: {combination_to_string((*scenario, 0))}")
    if not all_dfs:
        raise RuntimeError("No data files found for any scenario.")
    # Concatenate all scenarios together
    df_all = pd.concat(all_dfs, ignore_index=True)
    return df_all


def merge_two_dataframes_on_metadata(df1, df2, value_cols1=None, value_cols2=None):
    """
    Merges two long-format dataframes (with possibly different variables) into a single long-format dataframe.
    All original data and columns are preserved. Rows are matched on metadata columns and Node Number.
    If metadata columns differ, new rows are added (outer join).
    Variable columns are not suffixed; only unique variable columns are kept.

    Args:
        df1, df2: Input dataframes in long format.
        value_cols1, value_cols2: Lists of variable columns to include from each dataframe. If None, uses all except metadata.

    Returns:
        pd.DataFrame: Merged long-format dataframe.
    """
    # Identify metadata columns (intersection of both)
    metadata_cols = ['health', 'time', 'scenario', 'Node Number']
    metadata_cols = [col for col in metadata_cols if col in df1.columns and col in df2.columns]

    # Determine value columns if not provided
    if value_cols1 is None:
        value_cols1 = [col for col in df1.columns if col not in metadata_cols]
    if value_cols2 is None:
        value_cols2 = [col for col in df2.columns if col not in metadata_cols]

    # Prepare for merge: ensure no duplicate variable names
    overlap_vars = set(value_cols1) & set(value_cols2)
    if overlap_vars:
        raise ValueError(f"Variable columns overlap: {overlap_vars}. Please ensure variables are unique between dataframes.")

    # Merge on metadata columns and Node Number, outer join to preserve all rows
    df_merged = pd.merge(
        df1[metadata_cols + value_cols1],
        df2[metadata_cols + value_cols2],
        on=metadata_cols,
        how='outer'
    )

    # Fill missing values with 0 (optional, or use NaN if preferred)
    df_merged = df_merged.fillna(0)

    print(f"Merged dataframe shape: {df_merged.shape}")
    return df_merged


def select_df_subset(df, scenario_numbers):
    """Selects a subset of the main data-frame according to the given scenario number(s).

    Args:
        df (pd.DataFrame): The main data-frame containing all data.
        scenario_numbers (int or list/array-like): One or more unique scenario numbers.
    """
    if not isinstance(scenario_numbers, (list, tuple, set, np.ndarray)):
        scenario_numbers = [scenario_numbers]
    df_subset = df[df["scenario"].isin(scenario_numbers)]
    return df_subset


def filter_outliers(df: pd.DataFrame, lower_pct=0.001, upper_pct=1):
    """Filters out rows in the DataFrame where 'TotalDeformation' is outside the specified percentile range.
    Args:
        df (pd.DataFrame): The input DataFrame containing a 'TotalDeformation' column.
        lower_pct (float): The lower percentile threshold (default is 0.001 for 0.1%).
        upper_pct (float): The upper percentile threshold (default is 1 for no filtering).
    Returns:
        pd.DataFrame: The filtered DataFrame with outliers removed.
        pd.Index: Index of outlier node numbers that were filtered out.
    """

    # Remove outlier nodes based on TotalDeformation (e.g., outside 1st and 99th percentiles)
    lower, upper = df["TotalDeformation"].quantile([lower_pct, upper_pct])
    df_no_outliers = df[
        (df["TotalDeformation"] >= lower) &
        (df["TotalDeformation"] <= upper)
    ]
    filtered_count = len(df) - len(df_no_outliers)
    filtered_nodes = df[
        (df["TotalDeformation"] < lower) | (df["TotalDeformation"] > upper)
    ]
    outliers = filtered_nodes['Node Number'].unique()

    print(f"Filtered out {filtered_count} rows (nodes x time steps).")
    print("Filtered values (min/max):")
    print(filtered_nodes["TotalDeformation"].describe())
    print(f"Lower threshold: {lower}, Upper threshold: {upper}")
    print("Mean of remaining TotalDeformation:", df_no_outliers["TotalDeformation"].mean())
    print(f"Outliers node numbers: {sorted(outliers.tolist())}")
    print(f"Outliers values: {filtered_nodes['TotalDeformation'].unique()}")

    return df_no_outliers, outliers


def reshape_multi_variable_to_wide_nodes(
    df,
    value_cols=None,
    all_nodes=None,
    fill_value=0
):
    """
    Reshape long-format dataframe to node-centric rows stratified by scenario.
    Each row = one (scenario, Node Number). Columns = for each variable and timestep:
        <Variable>_t<time>
    Plus columns: X, Y, Z, health

    Args:
        df (pd.DataFrame): long-format dataframe with at least columns
            ['scenario','Node Number','time','X','Y','Z','health', <variables>]
        value_cols (list): list of variables to include. If None, defaults to 8 common vars.
        all_nodes (iterable or None): if provided, ensures every scenario has every node in this list.
            If None, the function will try to import VALID_NODE_NUMBERS from constants and use that.
        fill_value: value to fill for missing variable entries (default 0).

    Returns:
        pd.DataFrame: rows = (scenario, Node Number), columns = health,X,Y,Z and var_time columns.
    """

    if value_cols is None:
        value_cols = [
            'TotalDeformation',
            'DirectionalDeformation_X_axis',
            'DirectionalDeformation_Y_axis',
            'DirectionalDeformation_Z_axis',
            'EquivalentStress',
            'ShearStress_XY',
            'ShearStress_XZ',
            'ShearStress_YZ'
        ]

    # Ensure required columns exist
    required = {'scenario', 'Node Number', 'time', 'health', 'X', 'Y', 'Z'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Filter value_cols to those actually present
    value_cols = [c for c in value_cols if c in df.columns]
    if not value_cols:
        raise ValueError("No value columns found in dataframe.")

    # Get sorted unique times (preserve natural order)
    unique_times = sorted(df['time'].unique(), key=lambda x: float(x))
    time_cols = [str(t).replace('.', '_') for t in unique_times]

    # Prepare list of scenarios and nodes: use only node numbers present in the dataframe
    scenarios = sorted(df['scenario'].unique())
    all_nodes = sorted(df['Node Number'].unique())

    # We'll build a DataFrame indexed by (scenario, Node Number)
    index_all = pd.MultiIndex.from_product([scenarios, all_nodes], names=['scenario', 'Node Number'])

    # Aggregate metadata (health, X, Y, Z) per (scenario, Node Number) using first()
    meta = df.groupby(['scenario', 'Node Number'], sort=False).agg({
        'health': 'first',
        'X': 'first',
        'Y': 'first',
        'Z': 'first'
    })

    # Reindex metadata to full cartesian index so every scenario/node is present
    meta = meta.reindex(index_all)

    # If coordinates missing, try to fill from COORDS_DF by Node Number
    if 'Node Number' in COORDS_DF.columns:
        coords_map = COORDS_DF.set_index('Node Number')[['X', 'Y', 'Z']]
        # coords_map may have index dtype mismatch; ensure numeric
        # Fill missing X/Y/Z where available
        missing_coords_mask = meta[['X','Y','Z']].isnull().any(axis=1)
        if missing_coords_mask.any():
            nodes_missing = meta[missing_coords_mask].index.get_level_values('Node Number')
            coords_for_nodes = coords_map.reindex(nodes_missing).values
            meta.loc[missing_coords_mask, ['X','Y','Z']] = coords_for_nodes

    # Build variable-time wide block per variable and concatenate
    var_blocks = []
    for var in value_cols:
        # pivot to have times as columns; index = (scenario, Node Number)
        pivot = df.pivot_table(
            index=['scenario', 'Node Number'],
            columns='time',
            values=var,
            aggfunc='first'
        )
        # Ensure all time columns present and ordered
        pivot = pivot.reindex(columns=unique_times, fill_value=np.nan)
        # Rename columns to <var>_t<time>
        pivot.columns = [f"{var}_t{str(t).replace('.', '_')}" for t in pivot.columns]
        # Reindex to full cartesian index (scenarios x all_nodes)
        pivot = pivot.reindex(index_all)
        var_blocks.append(pivot)

    # Concatenate all variable blocks horizontally
    vars_wide = pd.concat(var_blocks, axis=1)

    # Fill missing variable values with fill_value (e.g., 0)
    vars_wide = vars_wide.fillna(fill_value)

    # Combine metadata and variables
    combined = pd.concat([meta, vars_wide], axis=1)

    # If health is missing (e.g., for some scenario/node), try to infer from scenario:
    # some pipeline uses scenario encoding where region==0 -> healthy; if health still missing leave NaN
    # Reset index to get columns
    combined = combined.reset_index()

    # Order columns: scenario, Node Number, health, X, Y, Z, then var columns
    var_cols_order = [c for c in combined.columns if any(c.startswith(v + "_t") for v in value_cols)]
    col_order = ['scenario', 'Node Number', 'health', 'X', 'Y', 'Z'] + var_cols_order
    # Keep only columns that exist (in case some were missing)
    col_order = [c for c in col_order if c in combined.columns]
    combined = combined[col_order]

    # Sanity check: verify that for each (scenario, Node Number, time, variable)
    # the value in the reconstructed wide DataFrame equals the original value in df.
    try:
        # Build long form of original values
        df_long = df[['scenario', 'Node Number', 'time', 'X', 'Y', 'Z', 'health'] + value_cols].melt(
            id_vars=['scenario', 'Node Number', 'time', 'X', 'Y', 'Z', 'health'],
            value_vars=value_cols,
            var_name='variable',
            value_name='orig_value'
        )

        # Determine expected wide columns present in combined
        expected_cols = []
        for var in value_cols:
            for t in unique_times:
                col = f"{var}_t{str(t).replace('.', '_')}"
                if col in combined.columns:
                    expected_cols.append(col)

        if expected_cols:
            combined_long = combined[['scenario', 'Node Number', 'X', 'Y', 'Z', 'health'] + expected_cols].melt(
                id_vars=['scenario', 'Node Number', 'X', 'Y', 'Z', 'health'],
                value_vars=expected_cols,
                var_name='var_time',
                value_name='new_value'
            )
            # extract variable and time from column name
            split = combined_long['var_time'].str.rsplit('_t', n=1)
            combined_long['variable'] = split.str[0]
            combined_long['time'] = split.str[1].str.replace('_', '.').astype(float)
            combined_long = combined_long[['scenario', 'Node Number', 'time', 'variable', 'new_value']]

            # Merge to compare original and reconstructed
            merged_chk = pd.merge(
                df_long[['scenario', 'Node Number', 'time', 'variable', 'orig_value']],
                combined_long,
                on=['scenario', 'Node Number', 'time', 'variable'],
                how='inner'
            )

            if merged_chk.empty:
                raise ValueError("Sanity check failed: no overlapping rows found between original and reconstructed data.")

            orig = merged_chk['orig_value'].to_numpy(dtype=float)
            new = merged_chk['new_value'].to_numpy(dtype=float)

            # consider NaN == NaN, otherwise numeric closeness
            orig_nan = np.isnan(orig)
            new_nan = np.isnan(new)
            both_nan = orig_nan & new_nan
            both_num = ~orig_nan & ~new_nan

            close_mask = np.zeros(len(merged_chk), dtype=bool)
            close_mask[both_nan] = True
            if both_num.any():
                close_mask[both_num] = np.isclose(orig[both_num], new[both_num], rtol=1e-6, atol=1e-8)

            mismatches = merged_chk.loc[~close_mask]
            if not mismatches.empty:
                sample = mismatches.head(10)
                raise ValueError(
                    f"Mismatch between original and reshaped data for {len(mismatches)} entries. "
                    f"Sample mismatches:\n{sample.to_string(index=False)}"
                )
            else:
                print("Sanity check passed: reshaped wide node dataframe matches original variables.")
    except Exception as e:
        # surface the error to the caller with context
        raise

    return combined


def reshape_multi_variable_to_wide(df, value_cols=None):
    """
    Reshapes long-format data with multiple variables into a wide 
    format suitable for training or PCA.
    
    Args:
        df: Long-format dataframe with metadata and multiple physical variables.
        value_cols: List of variables to include. If None, it uses the 8 variables 
                    from your snippet.
    """
    if value_cols is None:
        # Default physical variables from your data snippet
        value_cols = [
            'TotalDeformation', 
            'DirectionalDeformation_X_axis', 
            'DirectionalDeformation_Y_axis', 
            'DirectionalDeformation_Z_axis', 
            'EquivalentStress', 
            'ShearStress_XY', 
            'ShearStress_XZ', 
            'ShearStress_YZ'
        ]
    
    # 1. Filter for columns that actually exist in the dataframe
    value_cols = [col for col in value_cols if col in df.columns]
    
    # 2. Metadata columns that define a "sample" (the row identity)
    metadata_cols = ['health', 'time', 'scenario']
    metadata_cols = [col for col in metadata_cols if col in df.columns]
    
    print(f"Processing {len(value_cols)} variables across {df['Node Number'].nunique()} nodes...")
    
    # 3. Pivot the table
    # This creates a MultiIndex in the columns: (Variable Name, Node Number)
    df_wide = df.pivot_table(
        index=metadata_cols,
        columns='Node Number',
        values=value_cols
    )
    
    # 4. Flatten the column names
    # Converts (EquivalentStress, 1) -> "EquivalentStress_N1"
    df_wide.columns = [f"{var}_N{node}" for var, node in df_wide.columns]
    
    # 5. Bring metadata back as normal columns
    df_wide = df_wide.reset_index().fillna(0)
    
    print(f"Reshaping complete. Final Matrix Shape: {df_wide.shape}")
    return df_wide
