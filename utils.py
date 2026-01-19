import itertools
import functools
from pathlib import Path

import pandas as pd
import numpy as np
from constants import *


def combination_to_string(combination):
    train_config, load, season, region, variable = combination
    return f"{TRAIN_CONFIGS[train_config]}__{LOADS[load]}__{SEASONS[season]}__{REGIONS[region]}__{VARIABLE_NAMES[variable]}"

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
    from constants import COORDS_DF

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
    base_path = Path("data/Data2")

    for results_path in results_paths:
        filename = base_path
        filename /= TRAIN_CONFIGS[train_config]
        filename /= LOADS[load]
        filename /= SEASONS[season]
        filename /= REGIONS[region]
        filename /= results_path
        filename /= VARIABLES[variable] + ".csv"

        if filename.exists():
            break
    else:
        raise FileNotFoundError(f"Data file not found for combination: {combination_to_string(combination)}")


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
    Computes the difference in the variable between two data_frames.
    Args:
        df1, df2 (pd.DataFrame): Each a data-frame containing the same variable.
        top_pct (float): If <1.0, keep only the top percentage of differences by absolute value.
                         If >1.0, keep only the top N nodes by absolute difference.
    Returns:
        pd.DataFrame: DataFrame with Node Number, time, X, Y, Z, and the difference in the variable.
    """
    # Ensure both dataframes have the same columns
    if set(df1.columns) != set(df2.columns):
        raise ValueError(f"DataFrames have different columns: {set(df1.columns) ^ set(df2.columns)}")

    # Keep only relevant columns
    keep_cols = ["Node Number", "time", "X", "Y", "Z", var_name]
    df1 = df1[keep_cols]
    df2 = df2[keep_cols]

    # Check node numbers and time for consistency
    merge_cols = ["Node Number", "time", "X", "Y", "Z"]
    merged = pd.merge(df1, df2, on=merge_cols, suffixes=('_1', '_2'), how='inner')

    # Compute difference with safety check
    if not (merged[f"{var_name}_1"].shape == merged[f"{var_name}_2"].shape and merged[f"{var_name}_1"].index.equals(merged[f"{var_name}_2"].index)):
        raise ValueError("DataFrames to subtract do not have matching shapes or indices.")

    merged[var_name] = merged[f"{var_name}_1"] - merged[f"{var_name}_2"]

    # Keep only the top_pct number of top nodes by absolute difference
    abs_diff = np.abs(merged[var_name])
    # Only consider indices where the difference is non-zero
    # NOTE adjust r_tol, currently 10%
    # Mask out values close to zero for threshold calculation, but keep original indices for assignment
    abs_diff_mask = ~np.isclose(0, abs_diff, rtol=1e-1, atol=0)
    abs_diff_nonzero = abs_diff[abs_diff_mask]
    
    if len(abs_diff_nonzero) == 0:
        # If there are no non-zero differences, return an empty DataFrame
        # with the same columns and dtypes as `merged`
        merged[var_name] = np.nan
        return merged

    if top_pct < 1.0:
        # Keep only the top percentage of differences by absolute value
        threshold = np.percentile(abs_diff_nonzero, 100 * (1 - top_pct))
        # Set values below the threshold to null instead of filtering them out
        merged.loc[abs_diff < threshold, var_name] = np.nan
    elif top_pct > 1.0:
        # Get indices of top_pct largest differences (by value, not index)
        top_indices = abs_diff_nonzero.sort_values(ascending=False).head(int(top_pct)).index
        # Set all other values to NaN
        merged.loc[~merged.index.isin(top_indices), var_name] = np.nan

    # Select relevant columns
    # result_cols = merge_cols + [var_name]
    # result = merged[result_cols]

    return merged


def get_data_variable_aggregated(
    scenario_combination: tuple,
    filter_out_invalid_nodes: bool = True,
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
        # Set deformation of invalid nodes to 0 instead of filtering them out
        df_vars.loc[~valid_deformation_mask, ["TotalDeformation",
                              "DirectionalDeformation_X_axis",
                              "DirectionalDeformation_Y_axis",
                              "DirectionalDeformation_Z_axis"]] = 0

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


def get_data_all_aggregated():
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
                        df_vars = get_data_variable_aggregated(scenario)
                        all_dfs.append(df_vars)
                    except FileNotFoundError as e:
                        print(f"Skipping missing file for scenario: {combination_to_string((*scenario, 0))}")
                        continue
    if not all_dfs:
        raise RuntimeError("No data files found for any scenario.")
    # Concatenate all scenarios together
    df_all = pd.concat(all_dfs, ignore_index=True)
    return df_all


def select_df_subset(df, scenario_number):
    """Selects a subset of the main data-frame according to the given combination.

    Args:
        df (pd.DataFrame): The main data-frame containing all data.
        scenario_number (int): The unique scenario number.
    """
    df_subset = df[df["scenario"] == scenario_number]
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
