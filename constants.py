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

DATA_FOLDER_PATH = "data/Data2/"

if DATA_FOLDER_PATH == "data/Train_new_data/":
    DATA_FORMAT = "new"
else:
    DATA_FORMAT = "old"

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
ORIGINAL_VARIABLES = {
    0: "Total Deformation (mm)",
    1: "X_axis (mm)",
    2: "Y_axis (mm)",
    3: "Z_axis (mm)",
    4: "Equivalent Stress (Pa)",
    5: "Shear Stress XY (Pa)",
    6: "Shear Stress XZ (Pa)",
    7: "Shear Stress YZ (Pa)",
}
if DATA_FORMAT == "old":
    LOADS = {
        0: "Bigger_train",
        1: "Smaller_train",
    }
else:
    LOADS = {
        0: "Big_train",
        1: "Small_train",
    }
SEASONS = {
    0: "Summer",
    1: "Winter",
}
if DATA_FORMAT == "old":
    REGIONS = {
        0: "Perfect_structure",                 # Healthy
        1: "ip_frst_Arc_defect_all_tracks_111", # Arc1, Track1-3
        2: "ip_1and3track_3_arc_78910",         # Arc3, Track1&3
        3: "ip_first_track_3arc_78910",         # Arc3, Track1 
        4: "Ip_1track_1_arc_345",               # Arc1, Track1
        5: "ip_3track_1_arc_678",               # Arc1, Track3
        6: "ip_2_arc_all_tracks_222",           # Arc2, Track1-3
    }
else:
    REGIONS = {
        0: "Perfect_Structure",                 # Healthy
        1: "ip_frst_Arc_defect_all_tracks_111", # Arc1, Track1-3
        2: "ip_1and3line_3_arc_78910",         # Arc3, Track1&3
        3: "ip_first_track_3arc_78910",         # Arc3, Track1 
        4: "Ip_1track_1_arc_345",               # Arc1, Track1
        5: "ip_3track_1_arc_678",               # Arc1, Track3
        6: "ip_2_arc_all_tracks_222",           # Arc2, Track1-3
    }
TRAIN_CONFIGS = {
    0: "One_train_1st_track",
    1: "One_train_middle_track",
    2: "Two_trains_extreme_track_different_direction",
    3: "Two_trains_extreme_track_same_direction",
}
INDICATORS = [
    "health",
    "region",
    "load",
    "season",
    "train_config",
    "scenario"
]
ONE_HOT_INDICATORS = [
    "health",
    "region_0",
    "region_1",
    "region_2",
    "region_3",
    "region_4",
    "region_5",
    "load_0",
    "load_1",
    "season_0",
    "season_1",
    "train_config_0",
    "train_config_1",
    "train_config_2",
    "train_config_3",
]
VARIABLE_NAMES = list(VARIABLES.values())



with open("all_node_numbers.txt", "r") as f:
    global NODE_NUMBERS
    NODE_NUMBERS = [int(line.strip()) for line in f.readlines()]

with open("nodes_missing_stress.txt", "r") as f:
    global NODES_MISSING_STRESS
    NODES_MISSING_STRESS = {int(line.strip()) for line in f.readlines()}

with open("nodes_missing_coords.txt", "r") as f:
    global NODES_MISSING_COORDS
    NODES_MISSING_COORDS = {int(line.strip()) for line in f.readlines()}

with open("nodes_missing_deformation.txt", "r") as f:
    global NODES_MISSING_DEFORMATION
    # NODES_MISSING_DEFORMATION = {int(line.strip()) for line in f.readlines()}
    NODES_MISSING_DEFORMATION = {int(line.strip()) for line in f.readlines()}

# Valid: have stress (and therefore coordinates and deformation)
#   We are filtering out nodes all nodes missing stress (these have weird deformation values)
VALID_NODE_NUMBERS = [nn for nn in NODE_NUMBERS if nn not in (NODES_MISSING_STRESS.union(NODES_MISSING_DEFORMATION))]

with open("superset_outliers.txt", "r") as f:
    global SUPERSET_OUTLIERS
    SUPERSET_OUTLIERS = {int(line.strip()) for line in f.readlines()}
