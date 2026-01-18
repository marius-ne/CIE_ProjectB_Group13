import itertools
import functools
from pathlib import Path

import pandas as pd
import numpy as np
from constants import TRAIN_CONFIGS, LOADS, SEASONS, REGIONS, VARIABLES, VARIABLE_NAMES


def combination_to_string(combination):
    train_config, load, season, region, variable = combination
    return f"{TRAIN_CONFIGS[train_config]}__{LOADS[load]}__{SEASONS[season]}__{REGIONS[region]}__{VARIABLE_NAMES[variable]}"

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

def node_numbers_cache(filepath):
    """
    Returns a function that will read node numbers from the given file once,
    and return the cached node numbers on subsequent calls.

    Args:
        filepath (str): Path to the file containing node numbers.

    Returns:
        function: A function that returns the cached node numbers.
    """
    node_numbers = None

    def get_node_numbers():
        nonlocal node_numbers
        if node_numbers is None:
            df = pd.read_csv(filepath)
            node_numbers = df["Node Number"].unique()
        return node_numbers

    return get_node_numbers
get_node_numbers = node_numbers_cache("data/Data2/One_train_1st_track/Bigger_train/Summer/ip_1and3track_3_arc_78910/Results1/DirectionalDeformation_X_axis.csv")

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
    missing_count = df_with_coords['X'].isna().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} nodes did not have matching coordinates.")

    return df_with_coords


def read_data_file(
    train_config: int = 0,
    load: int = 0,
    season: int = 0,
    region: int = 0,
    variable: int = 0,
):
  """Reads data according to format and provides the data-frame as-is, with
  the categorical variables added as columns."""

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
      raise FileNotFoundError(f"Data file not found for combination: {combination_to_string((train_config, load, season, region, variable))}")

  # Encode the scenario as categorical columns
  #   -> TODO: Is there a way of encoding that
  #   preserves information? E.g. like encoding the name of a city as its latitude
  df = pd.read_csv(filename)
  num_nodes = len(df)
  df["season"] = season*np.ones(num_nodes,dtype=np.uint8)
  df["region"] = region*np.ones(num_nodes,dtype=np.uint8)
  healthy = 1 if region == 0 else 0
  df["health"] = healthy*np.ones(num_nodes,dtype=np.uint8)
  df["load"] = load*np.ones(num_nodes,dtype=np.uint8)
  df["train_config"] = train_config*np.ones(num_nodes,dtype=np.uint8)

  df = add_node_locations(df)

  return df


def get_data_variable_aggregated(
    scenario_combination: tuple
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
        print("Processing combination:", combination_to_string(same_variable_combination))

        var_name = VARIABLE_NAMES[same_variable_combination[-1]]

        # Get data file for current combination
        df = read_data_file(*same_variable_combination)

        # Turn the variable column into a single one and add a new time column
        df_melted = df.melt(
            id_vars=["Node Number","season","load","region","health","train_config","X","Y","Z",],
            var_name="variable",
            value_name=var_name
        )
        df_melted["time"] = df_melted["variable"].str[-3:].astype(np.float64)
        df_melted.drop(columns=["variable"],inplace=True)

        # Get node numbers and ensure they're consistent
        NODE_NUMBERS = get_node_numbers()
        try:
            assert all(NODE_NUMBERS == df_melted["Node Number"].unique())
        except ValueError or AssertionError:
            print("WARNING: Node numbers differ between data files!")
            print("Previous node numbers:", NODE_NUMBERS)
            print("Current node numbers:", df_melted["Node Number"].unique())
        dfs_variables.append(df_melted)

    # Concatenating all variables into a single data frame
    # -> we do an OUTER join, meaning all keys are kept (A U B)
    #   this should be safe, node numbers and the other shared columns are kept
    shared_cols = ["Node Number","train_config","load","season","region","health","X","Y","Z","time"]
    df_vars = functools.reduce(lambda left,right: pd.merge(left,right,on=shared_cols,
                                              how='outer'), dfs_variables)
    # Check that data has been preserved
    for df in dfs_variables:
        for var_name in VARIABLE_NAMES:
            if var_name in df.columns:
                  merged = pd.merge(df[shared_cols + [var_name]], df_vars[shared_cols + [var_name]],
                            on=shared_cols, how='inner')
                  assert len(merged) == len(df)

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
    # Generate all health and variable combinations for the given (train_config, load, season)
    train_config, load, season = scenario_combination
    combinations_grouped_by_region = [
      [
        [train_config, load, season, region, variable]
        for variable in VARIABLES.keys()
      ]
      for region in REGIONS.keys()
    ]

    # Merge data for all healths in the scenario
    dfs_regions = []
    for same_health_combinations in combinations_grouped_by_region:

        df_vars = get_data_variable_aggregated(same_health_combinations[0][:-1])

        dfs_regions.append(df_vars)

    # Check that columns are the same
    assert all(all(df.columns == dfs_regions[0].columns) for df in dfs_regions)

    # Concatenate them together
    df = pd.concat(dfs_regions,ignore_index=True)

    return df

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


