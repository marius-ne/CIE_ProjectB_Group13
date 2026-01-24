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


def scenario_number_to_combination(scenario_number):
    """
    Inverse of combination_to_scenario_number.
    Decodes an integer scenario_number into a tuple:
        (train_config, load, season, region)

    Accepts a single int and returns a 4-tuple of ints.
    If an iterable of ints is provided, returns a list of 4-tuples.
    """

    # support array-like input
    if np.ndim(scenario_number) > 0:
        return [scenario_number_to_combination(int(s)) for s in scenario_number]

    sn = int(scenario_number)
    n_regions = len(REGIONS)
    n_seasons = len(SEASONS)
    n_loads = len(LOADS)
    n_train = len(TRAIN_CONFIGS)

    region = sn % n_regions
    sn //= n_regions
    season = sn % n_seasons
    sn //= n_seasons
    load = sn % n_loads
    sn //= n_loads
    train_config = sn

    if train_config < 0 or train_config >= n_train:
        raise ValueError(f"scenario_number out of range: decoded train_config={train_config}")

    return (train_config, load, season, region)


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
    filter_negative_total_deformation: bool = False,
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

    # filter_node_numbers = set(VALID_NODE_NUMBERS)
    filter_node_numbers = set(X_BEAM_NODES).union(I_BEAM_NODES)

    # Identify nodes with missing coordinates
    if filter_out_invalid_nodes:
        
        # Take only filtered nodes
        df = df[df["Node Number"].isin(filter_node_numbers)]

        # Put stress to 0
        # Identify nodes with missing stress values
        stress_variables = list(ORIGINAL_VARIABLES.values())[4:] 
        stress_cols = [col for col in df.columns if any(var in col for var in stress_variables)]
        # df.loc[df["Node Number"].isin(NODES_MISSING_STRESS), stress_cols] = \
            # df.loc[df["Node Number"].isin(NODES_MISSING_STRESS), stress_cols].fillna(0)

        # Check that all nodes have coordinates and loads
        missing_coords_nodes = df[df[["X", "Y", "Z"]].isnull().any(axis=1)]["Node Number"].unique()
        if len(missing_coords_nodes) > 0:
            raise ValueError(f"{len(missing_coords_nodes)} nodes are missing coordinates after filtering.")
        missing_loads_nodes = df[df.isnull().any(axis=1)]["Node Number"].unique()
        if len(missing_loads_nodes) > 0:
            print(f"{len(missing_loads_nodes)} nodes are missing load data after filtering.")

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
        missing_nodes = set(filter_node_numbers) - set(df_melted["Node Number"].unique())
        if len(missing_nodes) > 0 and filter_out_invalid_nodes:
            raise ValueError(f"Data for variable {var_name} is missing node numbers: {missing_nodes}")

        df = df_melted
        del df_melted
    
    if filter_negative_total_deformation:
        # Put negative total deformation to nan
        if VARIABLE_NAMES[variable] == "TotalDeformation":
            df.loc[df["TotalDeformation"] < 0, "TotalDeformation"] = np.nan

    # Check that each time value has same count
    time_counts = df["time"].value_counts()
    assert time_counts.nunique() == 1, "Not all time values have the same count"

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


def get_variable_difference_between_dataframes(
    df1, 
    df2, 
    var_name: str, 
    top_pct: float = 1.0,
    aggregate_by_time: bool = True,
):
    """
    Computes the difference in the variable between two dataframes and keeps only the top nodes
    by per-node max absolute difference over time.

    CHANGE vs old behavior:
    - Even if a node is selected as "top", only the time-steps where the diff is non-zero are kept.
      All other time-steps for that node are set to NaN.
    - If a row's diff is zero, it becomes NaN (so only non-zero deltas remain non-NaN).
    """
    if set(df1.columns) != set(df2.columns):
        raise ValueError(f"DataFrames have different columns: {set(df1.columns) ^ set(df2.columns)}")

    keep_cols = ["Node Number", "time", "X", "Y", "Z", var_name]
    df1 = df1[keep_cols]
    df2 = df2[keep_cols]

    merge_cols = ["Node Number", "time", "X", "Y", "Z"]
    merged = pd.merge(df1, df2, on=merge_cols, suffixes=('_1', '_2'), how='inner')

    valid_rows = merged[f"{var_name}_1"].notna() & merged[f"{var_name}_2"].notna()
    merged = merged[valid_rows].copy()
    if merged.empty:
        merged[var_name] = np.nan
        return merged

    # Compute diff
    merged[var_name] = merged[f"{var_name}_1"] - merged[f"{var_name}_2"]
    abs_diff = merged[var_name].abs()

    # Per-node aggregate (max over time)
    per_node_max = abs_diff.groupby(merged["Node Number"]).max()

    # Nodes with any non-zero diff
    nonzero_nodes = per_node_max[per_node_max > 0]
    if nonzero_nodes.empty:
        merged[var_name] = np.nan
        return merged

    # Determine how many nodes to keep
    if 0 < top_pct < 1.0:
        k = max(1, int(np.ceil(top_pct * len(nonzero_nodes))))
        top_nodes = nonzero_nodes.sort_values(ascending=False).head(k).index
    elif top_pct == 1.0:
        top_nodes = nonzero_nodes.index
    else:
        k = int(top_pct)
        k = max(1, min(k, len(nonzero_nodes)))
        top_nodes = nonzero_nodes.sort_values(ascending=False).head(k).index

    # NEW: only keep non-zero diffs, and only for selected nodes
    if aggregate_by_time:
        keep_mask = merged["Node Number"].isin(top_nodes)
    else:
        keep_mask = merged["Node Number"].isin(top_nodes) & (merged[var_name] != 0)

    merged.loc[~keep_mask, var_name] = np.nan
    return merged


def get_data_variable_aggregated(
    scenario_combination: tuple,
    filter_out_invalid_nodes: bool = True,
    drop_invalid_nodes: bool = False,
    filter_negative_total_deformation: bool = False,
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
        df = read_data_file(
            *same_variable_combination, 
            filter_out_invalid_nodes=filter_out_invalid_nodes,
            filter_negative_total_deformation=filter_negative_total_deformation,
        )

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
    if filter_negative_total_deformation and "TotalDeformation" in df_vars.columns:
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
    scenario_combination: tuple,
    filter_invalid_nodes: bool = True,
    drop_invalid_nodes: bool = False,
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

        df_vars = get_data_variable_aggregated(
            (train_config, load, season, region),
            filter_out_invalid_nodes=filter_invalid_nodes,
            drop_invalid_nodes=drop_invalid_nodes,
        )

        dfs_regions.append(df_vars)

    # Check that columns are the same
    assert all(all(df.columns == dfs_regions[0].columns) for df in dfs_regions)

    # Concatenate them together
    df = pd.concat(dfs_regions,ignore_index=True)

    return df


def get_data_variable_and_region_and_season_aggregated(
    scenario_combination: tuple,
    drop_invalid_nodes: bool = False,
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
                df = get_data_variable_aggregated((train_config, load, season, region), drop_invalid_nodes=drop_invalid_nodes)
                all_dfs.append(df)
            except FileNotFoundError:
                print(f"Skipping missing file for scenario: {combination_to_string((train_config, load, season, region, 0))}")
                continue

    if not all_dfs:
        raise RuntimeError("No data files found for any season/region in this scenario.")
        
    df_all = pd.concat(all_dfs, ignore_index=True)
    return df_all


def get_data_variable_and_region_and_season_and_load_aggregated(
    scenario_combination: tuple,
    drop_invalid_nodes: bool = False,
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
                    df = get_data_variable_aggregated(
                        (train_config, load, season, region), 
                        drop_invalid_nodes=drop_invalid_nodes
                    )
                    all_dfs.append(df)
                except FileNotFoundError:
                    print(f"Skipping missing file for scenario: {combination_to_string((train_config, load, season, region, 0))}")
                    continue

    if not all_dfs:
        raise RuntimeError("No data files found for any load/season/region in this scenario.")

    df_all = pd.concat(all_dfs, ignore_index=True)
    return df_all


def get_data_all_aggregated(
    filter_invalid_nodes: bool = True,
    drop_invalid_nodes: bool = False,
    filter_negative_total_deformation: bool = False,
):
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
                        df_vars = get_data_variable_aggregated(
                            scenario, 
                            filter_out_invalid_nodes=filter_invalid_nodes,
                            drop_invalid_nodes=drop_invalid_nodes,
                            filter_negative_total_deformation=filter_negative_total_deformation,
                            )
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
    y_var_name: str,
    value_cols=None,
    all_nodes=None,
    fill_value=None,
    fail_on_missing: bool = True,
    sanity_check_rows: int = 200,   # 0 disables; avoids the gigantic melt/merge
    downcast_values: bool = False,   # True -> tries to downcast numeric value cols to save RAM
):
    """
    Memory-efficient reshape:
      rows   = (scenario, Node Number)
      cols   = health/delta_health, X, Y, Z + <Variable>_t<time>

    Notes:
      - If all_nodes is provided, we *expand* to scenarios x all_nodes (can increase RAM!).
      - Replaces full "melt sanity check" with a sampled check (sanity_check_rows).
    """
    import numpy as np
    import pandas as pd

    if value_cols is None:
        value_cols = [
            "TotalDeformation",
            "DirectionalDeformation_X_axis",
            "DirectionalDeformation_Y_axis",
            "DirectionalDeformation_Z_axis",
            "EquivalentStress",
            "ShearStress_XY",
            "ShearStress_XZ",
            "ShearStress_YZ",
        ]

    # ---- validate columns ----
    required = {"scenario", "Node Number", "time", y_var_name, "X", "Y", "Z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # keep only value cols that exist
    value_cols = [c for c in value_cols if c in df.columns]
    if not value_cols:
        raise ValueError("No value columns found in dataframe.")

    # minimal view of needed columns (avoid df.copy())
    use_cols = ["scenario", "Node Number", "time", y_var_name, "X", "Y", "Z"] + value_cols
    df_use = df.loc[:, use_cols]

    # ---- unique times, ordered ----
    # robust-ish sort: numeric if possible, otherwise stable string sort
    times = pd.Index(df_use["time"].unique())
    try:
        unique_times = list(times[np.argsort(times.astype(float))])
    except Exception:
        unique_times = sorted(times, key=lambda x: str(x))

    expected_time_count = len(unique_times)
    if expected_time_count == 0:
        raise ValueError("No time points found in dataframe.")

    # Make time categorical to keep ordering and reduce memory during unstack
    # (this copies only the 'time' column)
    df_use = df_use.assign(
        time=pd.Categorical(df_use["time"], categories=unique_times, ordered=True)
    )

    # Optional: downcast numeric value columns to save RAM
    if downcast_values:
        for c in value_cols:
            if pd.api.types.is_numeric_dtype(df_use[c]):
                df_use[c] = pd.to_numeric(df_use[c], downcast="float")

    # ---- drop incomplete (scenario, node) pairs (small intermediate only) ----
    counts = (
        df_use.groupby(["scenario", "Node Number"], sort=False, observed=True)["time"]
        .nunique()
    )
    good_pairs = counts[counts == expected_time_count]
    if good_pairs.empty:
        raise ValueError(
            f"All (scenario, node) pairs are missing time points. Expected {expected_time_count} unique times."
        )

    if len(good_pairs) != len(counts):
        bad_pairs = counts[counts < expected_time_count]
        sample = bad_pairs.head(10).reset_index()
        sample_str = ", ".join(
            f"(s={r['scenario']}, n={r['Node Number']})" for _, r in sample.iterrows()
        )
        print(
            f"WARNING: Dropping {len(bad_pairs)} (scenario, node) pairs missing time points. Examples: {sample_str}"
        )

        good_pairs_df = good_pairs.reset_index()[["scenario", "Node Number"]]
        # merge keeps only good pairs; avoids big MultiIndex isin
        df_use = df_use.merge(
            good_pairs_df, on=["scenario", "Node Number"], how="inner", sort=False, copy=False
        )

    # ---- metadata ----
    meta = (
        df_use.groupby(["scenario", "Node Number"], sort=False, observed=True)[
            [y_var_name, "X", "Y", "Z"]
        ]
        .first()
    )

    # fill missing coords from COORDS_DF if available
    if "COORDS_DF" in globals() and getattr(globals()["COORDS_DF"], "columns", None) is not None:
        COORDS_DF = globals()["COORDS_DF"]
        if "Node Number" in COORDS_DF.columns and {"X", "Y", "Z"}.issubset(COORDS_DF.columns):
            coords_map = COORDS_DF.set_index("Node Number")[["X", "Y", "Z"]]
            node_level = meta.index.get_level_values("Node Number")
            for c in ["X", "Y", "Z"]:
                if meta[c].isna().any():
                    meta[c] = meta[c].fillna(node_level.map(coords_map[c]))

    # ---- define final index (memory warning: product expansion can be huge) ----
    scenarios = meta.index.get_level_values("scenario").unique()
    if all_nodes is not None:
        # Explicit user request -> enforce all nodes for every scenario (can blow up RAM)
        index_all = pd.MultiIndex.from_product(
            [scenarios, list(all_nodes)], names=["scenario", "Node Number"]
        )
    else:
        index_all = meta.index  # only surviving pairs

    # ---- wide reshape in ONE shot ----
    # Ensure unique (scenario, node, time) by taking first (cheap + deterministic)
    df_vals = (
        df_use.groupby(["scenario", "Node Number", "time"], sort=False, observed=True)[value_cols]
        .first()
    )
    # unstack time -> columns become MultiIndex: (variable, time)
    wide = df_vals.unstack("time")

    # enforce time column order and presence without creating 8 pivots
    wide = wide.reindex(columns=unique_times, level=1)

    # align to final index (expands only if all_nodes was provided)
    wide = wide.reindex(index_all)

    # flatten columns
    def _tname(t):
        # keep consistent with your original replace('.', '_')
        return str(t).replace(".", "_")

    wide.columns = [f"{var}_t{_tname(t)}" for (var, t) in wide.columns]
    var_time_cols = list(wide.columns)

    if fill_value is not None:
        wide = wide.fillna(fill_value)

    # ---- combine ----
    meta = meta.reindex(index_all)
    combined = pd.concat([meta, wide], axis=1).reset_index()

    # ---- missing check (cheap; no melts) ----
    check_cols = ["scenario", "Node Number", y_var_name, "X", "Y", "Z"] + var_time_cols
    check_cols = [c for c in check_cols if c in combined.columns]

    missing_counts = combined[check_cols].isna().sum()
    total_missing = int(missing_counts.sum())
    if total_missing > 0:
        missing_summary = missing_counts[missing_counts > 0].to_dict()
        sample_rows = combined[combined[check_cols].isna().any(axis=1)].head(10)
        msg = (
            f"reshape_multi_variable_to_wide_nodes detected {total_missing} missing values "
            f"across {len(missing_summary)} columns.\n"
            f"Missing per column: {missing_summary}\n"
            f"Sample rows with missing values (up to 10):\n{sample_rows.to_string(index=False)}"
        )
        if fail_on_missing:
            raise ValueError(msg)
        else:
            print("WARNING:", msg)

    # ---- lightweight sanity check (sampled) ----
    if sanity_check_rows and sanity_check_rows > 0:
        # sample from df_vals (already deduped) to avoid huge intermediates
        n = min(int(sanity_check_rows), len(df_vals))
        if n > 0:
            sample = df_vals.sample(n=n, random_state=0)
            # build quick lookup into combined: set index once
            comb_idx = combined.set_index(["scenario", "Node Number"])
            bad = 0
            for (sc, node, t), row in sample.iterrows():
                # row: Series of value_cols
                t_suffix = _tname(t)
                try:
                    wide_row = comb_idx.loc[(sc, node)]
                except KeyError:
                    bad += 1
                    continue
                for var in value_cols:
                    col = f"{var}_t{t_suffix}"
                    if col not in wide_row.index:
                        bad += 1
                        continue
                    a = row[var]
                    b = wide_row[col]
                    # NaN-safe compare
                    if (pd.isna(a) and pd.isna(b)):
                        continue
                    if pd.isna(a) != pd.isna(b):
                        bad += 1
                        continue
                    # numeric close if possible, else exact
                    try:
                        if not np.isclose(float(a), float(b), rtol=1e-6, atol=1e-8):
                            bad += 1
                    except Exception:
                        if a != b:
                            bad += 1
            if bad > 0:
                raise ValueError(
                    f"Sanity check failed on sampled data: {bad} mismatches "
                    f"across ~{n*len(value_cols)} comparisons."
                )
            else:
                print(f"Sanity check passed on a sample of {n} rows.")

    # ---- row-count verification on filtered LONG data ----
    # Only valid for the filtered df_use (good pairs only, before any all_nodes expansion).
    surviving_pairs = df_use.groupby(["scenario", "Node Number"], sort=False, observed=True)["time"].nunique()
    surviving_pairs = surviving_pairs[surviving_pairs == expected_time_count]
    expected_long_rows = int(len(surviving_pairs) * expected_time_count)
    actual_long_rows = int(df_use.shape[0])
    if actual_long_rows != expected_long_rows:
        raise ValueError(
            f"Row count mismatch in long-form data: expected {expected_long_rows} rows "
            f"({len(surviving_pairs)} unique scenario-node pairs * {expected_time_count} times), "
            f"but found {actual_long_rows}."
        )

    # Wide rows: depends on whether we expanded via all_nodes
    expected_wide_rows = int(len(index_all))
    actual_wide_rows = int(combined.shape[0])
    if actual_wide_rows != expected_wide_rows:
        raise ValueError(
            f"Wide dataframe row count mismatch: expected {expected_wide_rows} rows, got {actual_wide_rows}."
        )

    print(
        f"Row count verification passed: {actual_long_rows} long rows -> {actual_wide_rows} wide rows "
        f"({len(surviving_pairs)} surviving node pairs, {expected_time_count} time points)."
    )

    # ---- order columns ----
    col_order = ["scenario", "Node Number", y_var_name, "X", "Y", "Z"] + var_time_cols
    combined = combined[[c for c in col_order if c in combined.columns]]

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


def _infer_scenario_labels(y: pd.Series, scenarios: pd.Series) -> pd.Series:
    """
    Infer one label per scenario. Verifies consistency; if inconsistent, uses mode and warns.
    Returns a Series indexed by scenario with int labels 0/1.
    """
    df_lab = pd.DataFrame({'scenario': scenarios, 'y': y}).dropna()
    agg = df_lab.groupby('scenario')['y'].agg(list)

    labels = {}
    for sc, vals in agg.items():
        vals = pd.Series(vals).astype(int)
        if vals.nunique() == 1:
            labels[sc] = int(vals.iloc[0])
        else:
            mode_val = int(vals.mode().iloc[0])
            print(f"WARNING: scenario {sc} has mixed labels {sorted(vals.unique())}; using mode={mode_val}.")
            labels[sc] = mode_val
    return pd.Series(labels, name='label')


def stratified_group_train_test_split(
    scenarios: np.ndarray,
    scenario_labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 0
):
    """
    Stratified split over groups (scenarios) with safeguards:
    - If a class has >=2 scenarios, ensure at least 1 stays in train and 1 goes to test.
    - If a class has only 1 scenario, it can only appear in one split (warns).
    Returns (train_scenarios, test_scenarios).
    """
    rng = np.random.RandomState(random_state)
    scenarios = np.asarray(scenarios)
    scenario_labels = np.asarray(scenario_labels).astype(int)

    classes = np.unique(scenario_labels)
    test_set = []

    for c in classes:
        cls_mask = scenario_labels == c
        cls_sc = scenarios[cls_mask]
        n = len(cls_sc)
        if n == 0:
            continue

        # desired test count (rounded), then enforce bounds
        n_test = int(round(n * test_size))
        if n >= 2:
            n_test = max(1, min(n_test, n - 1))  # keep at least 1 in each split
        else:
            # n == 1: cannot be in both splits
            if n_test not in (0, 1):
                n_test = int(bool(test_size >= 0.5))
            print(f"WARNING: class {c} has only one scenario; it will appear only in {'test' if n_test==1 else 'train'}.")

        if n_test > 0:
            chosen = rng.choice(cls_sc, size=n_test, replace=False)
            test_set.append(chosen)

    test_sc = np.unique(np.concatenate(test_set)) if test_set else np.array([], dtype=scenarios.dtype)
    test_sc_set = set(test_sc)
    train_sc = np.array([sc for sc in scenarios if sc not in test_sc_set])

    return train_sc, test_sc


def _scenario_class_counts(X: pd.DataFrame, y: pd.Series, scenario_col: str = 'scenario') -> pd.DataFrame:
    """
    Per-scenario row counts for classes 0 (healthy) and 1 (unhealthy).
    Returns a DataFrame: [scenario, n0, n1, n_total]
    """
    y_bin = (y > 0).astype(int)
    df = pd.DataFrame({scenario_col: X[scenario_col], 'y': y_bin})
    grp = df.groupby(scenario_col)['y'].value_counts().unstack(fill_value=0)
    # Ensure columns exist
    if 0 not in grp.columns:
        grp[0] = 0
    if 1 not in grp.columns:
        grp[1] = 0
    grp = grp.rename(columns={0: 'n0', 1: 'n1'})
    grp['n_total'] = grp['n0'] + grp['n1']
    return grp.reset_index()


def split_by_scenario(
    X: pd.DataFrame,
    y: pd.Series,
    scenario_col: str = 'scenario',
    test_size: float = 0.2,
    random_state: int = 0,
    stratified: bool = True,
    target_train_ratio_healthy: float | None = None,
    require_both_classes_in_train: bool = True
):
    """
    Group-aware split: no scenario leakage.
    If target_train_ratio_healthy is provided (0..1), chooses train scenarios to approximate that
    healthy(0)/unhealthy(1) ratio in training rows, while aiming for ~ (1-test_size) of total rows.
    Otherwise, falls back to stratified group split.
    Returns: X_train, X_test, y_train, y_test, scenarios_train, scenarios_test
    """
    if scenario_col not in X.columns:
        raise ValueError(f"'{scenario_col}' column not found in X")

    y_bin = (y > 0).astype(int)
    counts = _scenario_class_counts(X, y_bin, scenario_col)
    scenarios_all = counts[scenario_col].to_numpy()

    total_rows = int(counts['n_total'].sum())
    desired_train_rows = int(round((1.0 - test_size) * total_rows))

    # If no target ratio, do a simple stratified group split on scenarios
    if target_train_ratio_healthy is None:
        # Basic stratified choice at scenario level using scenario majority label
        labels_per_sc = (counts['n0'] >= counts['n1']).astype(int)  # 0-majority -> label 0; else 1
        # Keep at least one scenario per class in each split where possible
        pos_mask = labels_per_sc.values == 1
        neg_mask = labels_per_sc.values == 0
        rng = np.random.RandomState(random_state)

        pos_sc = scenarios_all[pos_mask]
        neg_sc = scenarios_all[neg_mask]

        # Pick test scenarios per class
        def pick(cls_sc):
            n = len(cls_sc)
            if n == 0:
                return np.array([], dtype=scenarios_all.dtype)
            n_test = int(round(n * test_size))
            if n >= 2:
                n_test = max(1, min(n_test, n - 1))
            else:
                # only 1 scenario -> it can only be in one split
                n_test = int(test_size >= 0.5)
            return rng.choice(cls_sc, size=min(n_test, len(cls_sc)), replace=False)

        test_sc = np.unique(np.concatenate([pick(pos_sc), pick(neg_sc)]))
        test_sc_set = set(test_sc)
        train_sc = np.array([sc for sc in scenarios_all if sc not in test_sc_set])
    else:
        # Greedy selection to hit desired ratio and approximate desired train size
        target = float(np.clip(target_train_ratio_healthy, 0.0, 1.0))
        rng = np.random.RandomState(random_state)
        remaining = counts.sample(frac=1.0, random_state=random_state)  # shuffle scenarios

        # Optional seeding to ensure both classes present in train
        train_sel = []
        train_n0 = 0
        train_n1 = 0
        train_rows = 0

        has_pos_rows = (remaining['n1'] > 0).any()
        has_neg_rows = (remaining['n0'] > 0).any()
        if require_both_classes_in_train and has_pos_rows and has_neg_rows:
            # Seed one mostly-healthy scenario and one mostly-unhealthy scenario
            seed_neg = remaining[remaining['n1'] == 0]
            seed_pos = remaining[remaining['n0'] == 0]
            if seed_neg.empty:
                seed_neg = remaining.assign(h_ratio=remaining['n0'] / (remaining['n_total'] + 1e-9)).sort_values('h_ratio', ascending=False).head(1)
            else:
                seed_neg = seed_neg.head(1)
            if seed_pos.empty:
                seed_pos = remaining.assign(u_ratio=remaining['n1'] / (remaining['n_total'] + 1e-9)).sort_values('u_ratio', ascending=False).head(1)
            else:
                seed_pos = seed_pos.head(1)

            for seed in [seed_neg, seed_pos]:
                row = seed.iloc[0]
                sc = row[scenario_col]
                if sc in train_sel:
                    continue
                train_sel.append(sc)
                train_n0 += int(row['n0'])
                train_n1 += int(row['n1'])
                train_rows += int(row['n_total'])
                remaining = remaining[remaining[scenario_col] != sc]

        def objective(n0, n1, rows):
            # Composite objective: match ratio and size
            if rows <= 0:
                ratio_err = 1.0
            else:
                ratio_err = abs(n0 / rows - target)
            size_err = abs(rows - desired_train_rows) / max(1, desired_train_rows)
            return ratio_err + 0.5 * size_err

        # Greedy add until reaching desired_train_rows
        while train_rows < desired_train_rows and len(remaining) > 0:
            best_idx = None
            best_obj = float('inf')
            for idx, row in remaining.iterrows():
                n0 = train_n0 + int(row['n0'])
                n1 = train_n1 + int(row['n1'])
                rows = train_rows + int(row['n_total'])
                obj = objective(n0, n1, rows)
                if obj < best_obj:
                    best_obj = obj
                    best_idx = idx
            if best_idx is None:
                break
            row = remaining.loc[best_idx]
            sc = row[scenario_col]
            train_sel.append(sc)
            train_n0 += int(row['n0'])
            train_n1 += int(row['n1'])
            train_rows += int(row['n_total'])
            remaining = remaining.drop(index=best_idx)

        train_sc = np.array(train_sel)
        test_sc = remaining[scenario_col].to_numpy()

    # Build masks and splits
    train_mask = X[scenario_col].isin(train_sc)
    test_mask = X[scenario_col].isin(test_sc)

    X_train = X[train_mask].reset_index(drop=True)
    X_test = X[test_mask].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)
    y_test = y[test_mask].reset_index(drop=True)

    # No leakage
    inter = set(train_sc).intersection(set(test_sc))
    assert len(inter) == 0, f"Scenario leakage: {inter}"

    # Diagnostics
    train_ratio_healthy = (y_train == 0).sum() / max(1, len(y_train))
    print(f"Scenarios: train={len(train_sc)}, test={len(test_sc)}")
    print(f"Training healthy ratio: {train_ratio_healthy:.3f} (target={target_train_ratio_healthy if target_train_ratio_healthy is not None else 'n/a'})")
    print("Row-level class distribution (train):\n", y_train.value_counts())
    print("Row-level class distribution (test):\n", y_test.value_counts())

    return X_train, X_test, y_train, y_test, train_sc, test_sc

