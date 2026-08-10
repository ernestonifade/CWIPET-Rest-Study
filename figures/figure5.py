import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib.patches import Ellipse
import streamlit as st
from utils import render_searchable_table
import warnings
warnings.filterwarnings("ignore")


@st.cache_data
def load_fig5_results():
    results_dir = 'results'
    def safe_read(filename):
        path = os.path.join(results_dir, filename)
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()

    data = {
        'Bodymetric_Met': safe_read('Bodymetric_Met Omni_Multi_Omic_Interaction_Master_Matrix.csv')
    }
    return data


def render_figure5_top_interactions(lmm_df, final_workspace):
  """Renders an automated 2x2 multi-panel regression grid of the top four

  interactions for Figure 5 in Streamlit.
  """
  # Set publication style configurations
  # --- MATPLOTLIB GLOBAL TYPOGRAPHY SETTINGS ---
  plt.rcParams['svg.fonttype'] = 'none'
  plt.rcParams.update({
      'font.family': 'sans-serif',
      'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
      'font.size': 8,
      'axes.titlesize': 9,
      'axes.labelsize': 8,
      'xtick.labelsize': 7,
      'ytick.labelsize': 7,
      'legend.fontsize': 7,
      'axes.labelweight': 'bold',
      'axes.titleweight': 'bold',
      'figure.dpi': 300,
      'savefig.dpi': 600
  })

  # 1. Isolate the top 4 strongest individual interaction rows
  top_4_interactions = lmm_df.head(4)
  if top_4_interactions.empty:
    st.warning("No interaction data available to plot.")
    return None

  # 2. Initialize a 2x2 multi-panel layout
  fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=300)
  axes = axes.flatten()  # Flatten into a 1D array to loop easily

  # Define high-contrast, colorblind-friendly cohort palette
  custom_palette = {
      "Group_8C": "#002df5",  # Electric Blue (Extreme Cold)
      "Group_15C": "#00f5d4",  # Neon Cyan (Cool Water)
      "Group_22C": "#f57a00",  # Deep Orange (Control)
  }

  # 3. Step through your top 4 hits and construct regression profiles
  for i, (_, row) in enumerate(top_4_interactions.iterrows()):
    if i >= len(axes):
      break
    metric_col = row["Physical_Metric"]
    molecule_col = f"{row['Molecular_Target']}_avg_delta"
    ax = axes[i]

    # Check if columns exist in workspace before plotting
    if metric_col not in final_workspace.columns or molecule_col not in final_workspace.columns:
      ax.text(
          0.5,
          0.5,
          "Columns not found",
          ha="center",
          va="center",
          transform=ax.transAxes,
      )
      continue

    # Generate separate group slope trends with 95% confidence intervals
    for grp, color, lbl in [
        ("Group_8C", custom_palette["Group_8C"], "8°C"),
        ("Group_15C", custom_palette["Group_15C"], "15°C"),
        ("Group_22C", custom_palette["Group_22C"], "22°C"),
    ]:
      subset = final_workspace[final_workspace["CWI_Group"] == grp]
      if not subset.empty:
        sns.regplot(
            x=metric_col,
            y=molecule_col,
            data=subset,
            color=color,
            label=lbl,
            ax=ax,
            scatter_kws={"s": 25, "alpha": 0.7},
            line_kws={"linewidth": 1.8},
        )

    # Clean up axis labels
    clean_x = (
        metric_col.replace("_", " ")
        .replace("degC", "(°C)")
        .replace("Percent", "%")
    )
    clean_y = row["Molecular_Target"].replace("_", " ") + " (Δ)"

    ax.set_xlabel(clean_x, fontweight="bold", fontsize=10)
    ax.set_ylabel(clean_y, fontweight="bold", fontsize=10)

    # Format subplot header with statistical metrics
    p_val_str = (
        f"{row['P_Value']:.5f}"
        if "P_Value" in row and not pd.isna(row["P_Value"])
        else "N/A"
    )
    r2_str = (
        f"{row['Model_R2']:.2f}"
        if "Model_R2" in row and not pd.isna(row["Model_R2"])
        else "N/A"
    )
    ax.set_title(
        f"Panel {chr(65+i)}: p = {p_val_str} | R² = {r2_str}",
        fontsize=10,
        pad=8,
        style="italic",
        loc="left",
    )

    ax.grid(True, linestyle=":", alpha=0.5)
    sns.despine(ax=ax, trim=True)

  # 4. Attach unified legend
  handles, labels = axes[0].get_legend_handles_labels()
  if handles:
    fig.legend(
        handles,
        ["8°C (Extreme Cold)", "15°C (Cool Water)", "22°C (Control)"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=True,
        facecolor="white",
    )

  plt.subplots_adjust(top=0.88, hspace=0.3, wspace=0.25)
  return fig
