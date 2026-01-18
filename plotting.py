
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"

from ipywidgets import interact, FloatSlider
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Own imports
from constants import COORDS_DF
from utils import select_df_subset, combination_to_string, VARIABLE_NAMES


def plot_bridge_3d_structure(highlight_nodes=None, color_scale: dict=None, s: int=4):
    """
    Plots the bridge structure in 3D using Plotly.
    Optionally highlights nodes in highlight_nodes, or uses a color scale if provided.
    Args:
        highlight_nodes (list, optional): List of node numbers to highlight.
        color_scale (dictionary-like, optional): Dictionary of {node_number: value}
            of values bewteen 0 and 1 to use for coloring nodes.
    """
    if color_scale is not None:
        # color_scale is a dict: {node_number: value}, nodes not present get value 0
        node_numbers = COORDS_DF["Node Number"].values
        node_colors = [color_scale.get(nn, np.nan) for nn in node_numbers]
        fig = go.Figure(data=[go.Scatter3d(
            x=COORDS_DF["X"],
            y=COORDS_DF["Y"],
            z=COORDS_DF["Z"],
            mode='markers',
            marker=dict(
                size=s*2,
                color=node_colors,
                colorscale='Viridis',
                colorbar=dict(title="Value"),
                showscale=True,
            )
        )])
        fig.update_layout(
            title="Bridge Structure (colored by value)",
            scene=dict(
                xaxis_title='X [m]',
                yaxis_title='Y [m]',
                zaxis_title='Z [m]'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
    else:
        # Default color for all nodes
        colors = ['black'] * len(COORDS_DF)
        # Highlight specified nodes
        if highlight_nodes is not None:
            node_indices = COORDS_DF[COORDS_DF["Node Number"].isin(highlight_nodes)].index
            for idx in node_indices:
                colors[idx] = 'red'

        fig = go.Figure(data=[go.Scatter3d(
            x=COORDS_DF["X"],
            y=COORDS_DF["Y"],
            z=COORDS_DF["Z"],
            mode='markers',
            marker=dict(
                size=s,
                color=colors,
            )
        )])
        fig.update_layout(
            title="Bridge Structure (highlighted nodes in red)",
            scene=dict(
                xaxis_title='X [m]',
                yaxis_title='Y [m]',
                zaxis_title='Z [m]'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
    fig.write_html("visualization/bridge_structure_3d.html", auto_open=True)
    fig.show()

def plot_bridge_3d_load(combination, time_point):
    """
    Plots bridge loads in 3D for a single time point.
    Args:
        combination (tuple): (train_config, load, season, health, variable)
        time_point (float): Time value to plot
    """
    df_subset = select_df_subset(combination)
    variable_to_plot = VARIABLE_NAMES[combination[-1]]
    timestamp_subset = df_subset[df_subset["time"] == time_point]
    node_loads = [
        timestamp_subset[timestamp_subset["Node Number"] == nn][variable_to_plot].values
        for nn in COORDS_DF["Node Number"]
    ]
    node_loads_flat = [nl[0] if len(nl) > 0 else np.nan for nl in node_loads]

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        COORDS_DF["X"],
        COORDS_DF["Y"],
        COORDS_DF["Z"],
        c=node_loads_flat, marker='o', s=2,
        cmap="viridis"
    )
    fig.colorbar(scatter, shrink=0.5)
    ax.set_title(f"Bridge Loads at time {time_point}\nFor combination: {combination_to_string(combination)}")
    plt.show()


def plot_bridge_loads_3d_slider(df, combination):
    """
    Plots bridge loads in 3D with a time slider.
    Args:
        df (pd.DataFrame): Dataframe to visualize.
        combination (tuple): (train_config, load, season, health, variable)
    """
    time_points = np.sort(df["time"].unique())
    variable_to_plot = VARIABLE_NAMES[combination[-1]]

    def plot_at_time(time_point_index):
        time_point = time_points[int(time_point_index)]
        timestamp_subset = df[df["time"] == time_point]
        node_loads = [
            timestamp_subset[timestamp_subset["Node Number"] == nn][variable_to_plot].values
            for nn in COORDS_DF["Node Number"]
        ]
        node_loads_flat = [nl[0] if len(nl) > 0 else np.nan for nl in node_loads]

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            COORDS_DF["X"],
            COORDS_DF["Y"],
            COORDS_DF["Z"],
            c=node_loads_flat, marker='o', s=2,
            cmap="viridis"
        )
        fig.colorbar(scatter, shrink=0.5)
        ax.set_title(f"Bridge Loads at time {time_point}\nFor combination: {combination_to_string(combination)}")
        fig.savefig(f"../visualization/bridge_loads_3d_{time_point}.png")
        # fig.show()

    interact(
        plot_at_time,
        time_point_index=FloatSlider(
            min=0,
            max=len(time_points)-1,
            step=1,
            value=0,
            description='Time Index'
        )
    )