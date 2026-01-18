import pandas as pd

# ------------------
# Pre-read node locations
NODE_LOCATIONS_DF = None
coords_source: str = "nodeExport.txt"

# 1. Load the coordinate data if a file path is provided
if isinstance(coords_source, str):
    # We read node locations, last column is NaN and is dropped
    print("Reading node coordinates from file:", coords_source)
    COORDS_DF = pd.read_csv(coords_source, sep='\t', engine="python", usecols=[0, 1, 2, 3])
    # Ensure column names match what we expect
    COORDS_DF.columns = ['Node Number', 'X', 'Y', 'Z']
else:
    COORDS_DF = coords_source

# 2. Convert Node Number to int to ensure matching types
COORDS_DF['Node Number'] = COORDS_DF['Node Number'].astype(int)
# ------------------


VARIABLES = {
      0: "TotalDeformation",
      1: "DirectionalDeformation_X_axis",
      2: "DirectionalDeformation_Y_axis",
      3: "DirectionalDeformation_Z_axis",
      4: "EquivalentStress",
      5: "ShearStress_XY",
      6: "ShearStress_XZ",
      7: "ShearStress_YZ",
  }
LOADS = {
    0: "Bigger_train",
    1: "Smaller_train",
}
SEASONS = {
    0: "Summer",
    1: "Winter",
}
REGIONS = {
    0: "Perfect_structure",
    1: "ip_frst_Arc_defect_all_tracks_111",
    2: "ip_1and3track_3_arc_78910",
    3: "ip_first_track_3arc_78910",
    4: "Ip_1track_1_arc_345",
    5: "ip_3track_1_arc_678",
    6: "ip_2_arc_all_tracks_222",
}
TRAIN_CONFIGS = {
    0: "One_train_1st_track",
    1: "One_train_middle_track",
    2: "Two_trains_extreme_track_different_direction",
    3: "Two_trains_extreme_track_same_direction",
}
VARIABLE_NAMES = list(VARIABLES.values())


with open("valid_node_numbers.txt", "r") as f:
    global VALID_NODE_NUMBERS
    VALID_NODE_NUMBERS = {int(line.strip()) for line in f.readlines()}

with open("all_node_numbers.txt", "r") as f:
    global NODE_NUMBERS
    NODE_NUMBERS = [int(line.strip()) for line in f.readlines()]
