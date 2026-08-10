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
def load_fig6_results():
    results_dir = 'results'
    def safe_read(filename):
        path = os.path.join(results_dir, filename)
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()

    data = {
        'Bodymetric_Cyt': safe_read('Bodymetric_Cyt Omni_Multi_Omic_Interaction_Master_Matrix.csv'),
        'Work_Space_Bodymetric': safe_read('final_workspace_Bodymetric_Cyt Omni_Multi_Omic_Interaction_Master_Matrix.csv'),
        'Bodytemp_Cyt': safe_read('Bodytemp_Cyt Omni_Multi_Omic_Interaction_Master_Matrix.csv'),
        'Work_SpaceBodytemp': safe_read('final_workspace_Bodytemp_Cyt Omni_Multi_Omic_Interaction_Master_Matrix.csv')
    }
    return data


def render_side_by_side_interaction_plot(
    lmm_df_1,
    workspace_1,
    target_metric_1,
    target_molecule_1,
    lmm_df_2,
    workspace_2,
    target_metric_2,
    target_molecule_2,
    layout_preset='micro',
):
  """Renders two individual interaction plots side by side from two different

  results dataframes and workspaces, optimized for Streamlit or publication.
  """
  # Publication style configurations
  mpl.rcParams['svg.fonttype'] = 'none'
  if layout_preset == 'micro':
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 8,
        'font.weight': 'bold',
        'axes.titlesize': 8.5,
        'axes.titleweight': 'bold',
        'axes.labelsize': 8,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.2,
        'savefig.bbox': 'standard',
    })
  else:
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 9,
        'font.weight': 'bold',
        'axes.titlesize': 10,
        'axes.titleweight': 'bold',
        'axes.labelsize': 9,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.linewidth': 1.2,
        'lines.linewidth': 1.5,
        'savefig.bbox': 'tight',
    })

  # Initialize a 1-row, 2-column side-by-side layout
  fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), dpi=150)

  custom_palette = {
      'Group_22C': '#f57a00',  # Deep Orange (Control)
      'Group_15C': '#00f5d4',  # Neon Cyan (Cool Water)
      'Group_8C': '#002df5',  # Electric Blue (Extreme Cold)
  }

  plots_data = [
      (lmm_df_1, workspace_1, target_metric_1, target_molecule_1, axes[0]),
      (lmm_df_2, workspace_2, target_metric_2, target_molecule_2, axes[1]),
  ]

  for lmm_df, workspace, t_metric, t_mol, ax in plots_data:
    # Extract specific row or fallback
    specific_row = lmm_df[
        (lmm_df['Physical_Metric'] == t_metric)
        & (lmm_df['Molecular_Target'] == t_mol)
    ]
    if specific_row.empty and not lmm_df.empty:
      row = lmm_df.iloc[0]
    elif not specific_row.empty:
      row = specific_row.iloc[0]
    else:
      ax.text(
          0.5,
          0.5,
          'No Data Available',
          ha='center',
          va='center',
          transform=ax.transAxes,
      )
      continue

    metric_col = row['Physical_Metric']
    molecule_col = f"{row['Molecular_Target']}_avg_delta"

    if (
        metric_col not in workspace.columns
        or molecule_col not in workspace.columns
    ):
      ax.text(
          0.5,
          0.5,
          'Columns Missing in Workspace',
          ha='center',
          va='center',
          transform=ax.transAxes,
      )
      continue

    # Plot regression trends for each cohort group
    for grp, color, lbl in [
        ('Group_22C', custom_palette['Group_22C'], '22°C'),
        ('Group_15C', custom_palette['Group_15C'], '15°C'),
        ('Group_8C', custom_palette['Group_8C'], '8°C'),
    ]:
      subset = workspace[workspace['CWI_Group'] == grp]
      if not subset.empty:
        sns.regplot(
            x=metric_col,
            y=molecule_col,
            data=subset,
            color=color,
            label=lbl,
            ax=ax,
            scatter_kws={'s': 30, 'alpha': 0.75},
            line_kws={'linewidth': 1.8},
        )

    # Format axis labels
    clean_x = (
        metric_col.replace('_', ' ')
        .replace('degC', '(°C)')
        .replace('Percent', '%')
        + ' (Δ)'
    )
    clean_y = row['Molecular_Target'].replace('_', ' ') + ' (Δ)'

    ax.set_xlabel(clean_x)
    ax.set_ylabel(clean_y)

    # Header stats format
    p_label = (
        f"FDR q = {row['FDR_Adjusted_P']:.5f}"
        if 'FDR_Adjusted_P' in row and not pd.isna(row['FDR_Adjusted_P'])
        else f"p = {row['P_Value']:.5f}"
    )
    r2_val = (
        f"{row['Model_R2']:.2f}"
        if 'Model_R2' in row and not pd.isna(row['Model_R2'])
        else 'N/A'
    )
    ax.set_title(
        f'{clean_y}\n({p_label} | R² = {r2_val})', fontsize=8.5, loc='center'
    )

    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(
        frameon=True,
        facecolor='white',
        loc='upper left',
        prop={'weight': 'bold', 'size': 6.5},
    )
    sns.despine(ax=ax, trim=True)

  plt.subplots_adjust(wspace=0.3)
  return fig
    

def render_ols_searchable_table(df_input, title_prefix):
  st.subheader(f'{title_prefix} - Multi-Omic Interactions Table')
  if df_input.empty:
    st.info('No records loaded.')
    return

  # Filter based on FDR-adjusted significance or raw interaction p-value
  sig_col = (
      'Significant_After_FDR'
      if 'Significant_After_FDR' in df_input.columns
      else 'Interaction_P_Value'
  )
  if sig_col == 'Significant_After_FDR':
    filtered_df = df_input[df_input[sig_col] == True].copy()
  else:
    filtered_df = df_input[df_input[sig_col] < 0.05].copy()

  col1, col2 = st.columns([3, 1])
  with col1:
    search_query = st.text_input(
        f'🔍 Search {title_prefix} Features:',
        '',
        key=f'search_{title_prefix}',
    )
  with col2:
    st.metric('Significant Interactions', len(filtered_df))

  # Search across Physical Metrics or Molecular Targets
  if search_query:
    mask = (
        filtered_df['Physical_Metric'].str.contains(
            search_query, case=False, na=False
        )
        | filtered_df['Molecular_Target'].str.contains(
            search_query, case=False, na=False
        )
    )
    filtered_df = filtered_df[mask]

  # Select columns matching your OLS schema
  desired_cols = [
      'Physical_Metric',
      'Molecular_Target',
      'Interaction_Contrast',
      'Interaction_Beta',
      'Interaction_P_Value',
      'FDR_Adjusted_P',
      'Significant_After_FDR',
      'Model_R2',
      'Simple_Slope_8C',
      'Simple_Slope_8C_P',
      'Simple_Slope_15C',
      'Simple_Slope_15C_P',
      'Simple_Slope_22C',
      'Simple_Slope_22C_P',
  ]
  display_cols = [c for c in desired_cols if c in filtered_df.columns]

  # Sort by FDR-adjusted p-value or interaction p-value
  sort_col = (
      'FDR_Adjusted_P'
      if 'FDR_Adjusted_P' in filtered_df.columns
      else 'Interaction_P_Value'
  )

  st.dataframe(
      filtered_df[display_cols].sort_values(by=sort_col),
      use_container_width=True,
      hide_index=True,
  )

def render_figure6():
    st.title("🧬 Figure 6: Ordinary Least Regressions-Modal")
    st.markdown("Explore How Baseline Body Metrics and temperatures Modulate Cytokine Responses Under Different Degrees of Immersion ")

    data = load_fig6_results()

    tab1, tab2, tab3= st.tabs([
        "1️⃣ CWI x Bodymetrics Interaction on Metabolites",
        "2️⃣ Table: CWI x Bodymetrics Interaction on Cytokines",
        "2️⃣ Table: CWI x Bodytemp Interaction on Cytokines",
    ])

    with tab1:
        st.subheader("Figure 5: Top Multi-Omic Interactions Grid")
        render_figure5_top_interactions(data['Bodymetric_Met'], data['Work_Space'])
        if fig_to_display is not None:
            st.pyplot(fig_to_display)

    with tab2:
        render_searchable_table(data['Bodymetric_Met'], "Multi-Omic Interactions Table")
