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

plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 1.0
plt.rcParams['agg.path.chunksize'] = 10000

# --- 1. DIRECT PRE-COMPUTED DATA LOADER ---
@st.cache_data
def load_results():
    if os.path.exists('results/fig4_ancova_stats.csv'):
        data_dir = 'results'
    elif os.path.exists('../results/fig4_ancova_stats.csv'):
        data_dir = '../results'
    else:
        data_dir = '.'
        
    ancova_df = pd.read_csv(os.path.join(data_dir, 'fig4_ancova_stats.csv'))
    posthoc_df = pd.read_csv(os.path.join(data_dir, 'fig4_posthoc_contrasts.csv'))
    #protein_diff_pathway_df = pd.read_csv(os.path.join(data_dir, 'enrichment_permutation_results_for_interacting_proteins.csv'))
    wide_df = pd.read_csv(os.path.join(data_dir, 'fig4_processed_data.csv'))

    metadata_cols = ['Subject_ID', 'sex', 'time', 'ID', 'Sex', 'Time', 'Group', 'TimePoint', 'BaselineValue']
    meta_in_df = [c for c in metadata_cols if c in wide_df.columns]
    protein_cols = [c for c in wide_df.columns if c not in meta_in_df and pd.api.types.is_numeric_dtype(wide_df[c])]

    fig4_long_df = wide_df.melt(
        id_vars=meta_in_df,
        value_vars=protein_cols,
        var_name='Protein',
        value_name='Value'
    )
    long_df = wide_df.copy()
    
    return ancova_df, posthoc_df, long_df, fig4_long_df

# Alias function to satisfy app.py import requirements cleanly
load_fig4_results = load_results


def plot_interaction_heatmap_19_proteins(
    long_df, full_anova_results,
    id_col='Subject_ID', sex_col='sex', time_col='time', prot_col='Protein', value_col='Value',
    time_order=('baseline', '10min', '2hrs'),
    time_display={'10min': '10min', '2hrs': '2hrs'},
    sex_order=('8°C', '15°C', '22°C'),
    sex_short={'8°C': '8°C', '15°C': '15°C', '22°C': '22°C'},
    use_p_col='p_value_raw', alpha=0.05,
    effect_term='TimePoint:Group',
    cmap='coolwarm', pseudocount=1e-9,
    figsize_w=3.8, row_height=0.22,
    title=None,
    proteins_of_interest=None, poi_color="red", poi_bold=True,
    x_tick_rotation=0
):
    mpl.rcParams['svg.fonttype'] = 'none'
    
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'FreeSans', 'DejaVu Sans', 'sans-serif'],
        'font.size': 8, 'font.weight': 'bold',
        'axes.titlesize': 8.5, 'axes.titleweight': 'bold',
        'axes.labelsize': 8, 'axes.labelweight': 'bold',
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.linewidth': 1.0, 'lines.linewidth': 1.2,
        'savefig.bbox': 'tight'
    })

    df = long_df[[id_col, sex_col, time_col, prot_col, value_col]].copy()
    for c in (sex_col, time_col, prot_col):
        df[c] = df[c].astype(str).str.strip()
    
    df[sex_col] = df[sex_col].map(lambda x: sex_short.get(x, x))
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df = df[df[time_col].isin(time_order) & df[sex_col].isin(sex_order)]

    baseline = time_order[0]
    post_times = [t for t in time_order if t != baseline]

    mean_tbl = (df.groupby([prot_col, sex_col, time_col])[value_col]
                  .mean().unstack(time_col).reindex(columns=time_order))

    abs_diff = mean_tbl[post_times].sub(mean_tbl[baseline], axis=0)
    log2fc = abs_diff

    long_fc = (log2fc.stack().to_frame('log2FC').reset_index()
               .rename(columns={'level_2': 'time'}))
    long_fc['col'] = long_fc['time'].map(time_display).fillna(long_fc['time']) \
                      + '_' + long_fc[sex_col].map(sex_short).fillna(long_fc[sex_col])
    mat = long_fc.pivot(index=prot_col, columns='col', values='log2FC')

    col_order = []
    for t in post_times:
        tdisp = time_display.get(t, t)
        for s in sex_order:
            col_order.append(f'{tdisp}_{sex_short.get(s, s)}')
    mat = mat.reindex(columns=col_order)

    stats = full_anova_results.copy()
    if 'Effect' in stats.columns:
        stats = stats[stats['Effect'] == effect_term]
        
    name_col = prot_col if prot_col in stats.columns else 'Protein'
    stats[name_col] = stats[name_col].astype(str).str.strip()
    stats = stats.set_index(name_col)

    common = mat.index.intersection(stats.index)
    mat, stats = mat.loc[common], stats.loc[common]

    sig_mask = stats[use_p_col].astype(float) < alpha
    mat, stats = mat.loc[sig_mask], stats.loc[sig_mask]

    order_idx = stats[use_p_col].astype(float).sort_values().index
    mat, stats = mat.loc[order_idx], stats.loc[order_idx]
    n = len(mat)
    fig_h = max(3.5, row_height * n)

    pvals = stats[use_p_col].astype(float)
    p_text = pvals.apply(lambda x: f'{x:.3g}')

    fig = plt.figure(figsize=(figsize_w, fig_h))
    gs = fig.add_gridspec(nrows=2, ncols=8, height_ratios=[0.08, 0.92], wspace=0.05, hspace=0.02,
                           width_ratios=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.35, 0.05])

    ax_heat = fig.add_subplot(gs[1, 0:6])
    ax_top  = fig.add_subplot(gs[0, 0:6], sharex=ax_heat)
    ax_cbar = fig.add_subplot(gs[1, 7])

    if title:
        fig.suptitle(title, y=0.98, fontweight='bold', fontsize=9)

    max_val = np.percentile(np.abs(mat.values), 98)

    hm = sns.heatmap(
        mat, ax=ax_heat, cmap=cmap, center=0, vmin=-3.5, vmax=3.5,
        cbar=False, linewidths=0.2, linecolor='white'
    )

    cbar = fig.colorbar(hm.collections[0], cax=ax_cbar)
    cbar.set_label('Log₂ Fold Change', fontsize=7.5, fontweight='bold', labelpad=2)
    cbar.ax.tick_params(width=1.0, labelsize=7)
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight('bold')

    ax_heat.set_yticks(np.arange(0.5, n + 0.5, 1.0))
    ax_heat.set_yticklabels(mat.index)
    for label in ax_heat.get_yticklabels():
        label.set_rotation(0)
        label.set_fontweight('bold')

    if proteins_of_interest:
        poi = set(map(str, proteins_of_interest))
        for tick in ax_heat.get_yticklabels():
            if tick.get_text() in poi:
                tick.set_color(poi_color)
                if poi_bold:
                    tick.set_fontweight("bold")

    ax_heat.set_xticks(np.arange(len(mat.columns)) + 0.5)
    ax_heat.set_xticklabels([])
    
    mf_labels = [c.split('_')[-1] for c in mat.columns]
    for i, lab in enumerate(mf_labels):
        ax_heat.text(
            i + 0.5, -0.02, lab, ha='center', va='top',
            transform=ax_heat.get_xaxis_transform(),
            rotation=x_tick_rotation, fontsize=7.5, fontweight='bold'
        )

    n_sexes = len(sex_order)
    for i in range(n_sexes, len(mat.columns), n_sexes):
        ax_heat.axvline(i, color='k', lw=0.6, alpha=0.25)

    ax_heat.set_xlabel('Sex', labelpad=14, fontsize=8, fontweight='bold')
    ax_heat.set_ylabel('Proteins', fontsize=8, fontweight='bold')

    ax_top.set_xlim(ax_heat.get_xlim())
    ax_top.set_ylim(0, 1)
    ax_top.axis('off')

    n_timepoints = len(post_times)
    centers = [i * n_sexes + (n_sexes - 1) / 2 + 0.5 for i in range(n_timepoints)]
    time_labels_disp = [time_display.get(t, t) for t in post_times]

    for xc, lab in zip(centers, time_labels_disp):
        ax_top.text(xc, 0.3, lab, ha='center', va='center', fontsize=8, fontweight='bold')

    ax_p = ax_heat.twinx()
    ax_p.set_ylim(ax_heat.get_ylim())
    ax_p.set_yticks(np.arange(0.5, n + 0.5, 1.0))
    ax_p.set_yticklabels(p_text)
    for label in ax_p.get_yticklabels():
        label.set_rotation(0)
        label.set_fontweight('bold')
    ax_p.set_ylabel('p_raw (Interaction)', rotation=90, labelpad=8, fontsize=8, fontweight='bold')
    ax_p.set_xticks([])

    plt.subplots_adjust(bottom=0.15)
    return fig

# --- MAIN RENDER FUNCTION FOR STREAMLIT ---
def render_figure4():
    ancova_df, posthoc_df, long_df, fig1_long_df = load_results()

    # Streamlit Selectbox replacing ipywidgets dropdown
    selected_view = st.selectbox(
        "Select Section View:",
        [
            '🔥 Heatmap: Time × Group Interactions Effect',
            '🧬 Est Marginal Mean Plot: Group Effect',
            '📄 RM-ANCOVA Model Summary (Main & Interaction Effects)',
            '🔍 Post-Hoc Pairwise Contrasts (emmeans)'
        ]
    )

    st.markdown("---")

    if selected_view == '🔥 Heatmap: Time × Group Interactions (19 Candidates)':
        st.markdown("""
        <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px; font-size: 11px; color: #856404;">
            <b>Figure 3A (Candidate Heatmap):</b> Relative fold change dynamics for 19 candidate proteins meeting nominal significance (<i>p</i><sub>raw</sub> &lt; 0.05) for Time × Group interaction across recovery.
        </div>
        """, unsafe_allow_html=True)

        fig_hm = plot_interaction_heatmap_19_proteins(
            long_df=fig3_long_df,
            full_anova_results=ancova_df,
            id_col='Subject_ID', sex_col='sex', time_col='time', prot_col='Protein', value_col='Value',
            time_order=('baseline', '10min', '2hrs'),
            time_display={'10min': '10min', '2hrs': '2hrs'},
            sex_order=('22°C', '15°C', '8°C'),
            use_p_col='p_value_raw', alpha=0.05,
            effect_term='TimePoint:Group',
            title=None
        )
        st.pyplot(fig_hm)

    elif selected_view == 'Est Marginal Mean Plot: Group Effect':
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5), layout='constrained')
        df_emm = pd.DataFrame({
             'Metabolite': ['Muscle Temp1', 'Muscle Temp1', 'Muscle Temp1', 'Muscle Temp2', 'Muscle Temp2', 'Muscle Temp2', 'Muscle Temp3', 'Muscle Temp3', 'Muscle Temp3'],
             'Group': ['22°C', '15°C', '8°C', '22°C', '15°C', '8°C', '22°C', '15°C', '8°C'],
             'EMM': [32.6, 31.3, 32.2, 33.7, 32.7, 33.3, 35.0, 34.4, 34.7],
             'CI_lower': [32.0, 30.6, 31.5, 33.2, 32.2, 32.8, 34.5, 33.9, 34.3],
             'CI_upper': [33.2, 31.9, 32.8, 34.2, 33.3, 33.8, 35.5, 34.9, 35.2],
             'SE': [0.307, 0.317, 0.309, 0.254, 0.275, 0.251, 0.235, 0.255, 0.226]
         })
        
        # Subplot 1: 
        sub_size = df_emm[df_emm["Metabolite"] == "Muscle Temp1"]
        
        # Define your custom group order
        group_order = ["22°C", "15°C", "8°C"]
        
        for idx, row in sub_size.iterrows():
          # Map group names to integer x-positions (0, 1, 2)
          group_val = row["Group"]
          if group_val in group_order:
            x_pos = group_order.index(group_val)
          else:
            continue
        
          # Define custom markers and colors for each of the 3 groups
          markers = ["o", "s", "^"]  # circle, square, triangle
          colors = ["#f57a00", "#00f5d4", "#002df5"]  # distinct colors for each group
        
          axes[0].errorbar(
              x_pos,
              row["EMM"],
              yerr=[
                  [row["EMM"] - row["CI_lower"]],
                  [row["CI_upper"] - row["EMM"]],
              ],
              fmt=markers[x_pos],
              color=colors[x_pos],
              capsize=5,
              markersize=7,
          )
        
        # Configure axes for 3 distinct positions
        axes[0].set_xticks([0, 1, 2])
        axes[0].set_xticklabels(group_order)
        axes[0].set_xlim(-0.5, 2.5)  # Expanded padding for 3 ticks
        
        axes[0].set_ylabel("Adjusted EMM (nm)")
        axes[0].set_title("Muscle Temp1")
        
        # Optional: Add significance brackets between specific pairs (e.g., index 0 and 2)
         add_bracket(axes[0], 0, 2, y_val, h_val, "* p < 0.05")
        
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)

        # Subplot 2: 
        sub_size = df_emm[df_emm["Metabolite"] == "Muscle Temp2"]

        # Define your custom group order
        group_order = ["22°C", "15°C", "8°C"]
        
        for idx, row in sub_size.iterrows():
          # Map group names to integer x-positions (0, 1, 2)
          group_val = row["Group"]
          if group_val in group_order:
            x_pos = group_order.index(group_val)
          else:
            continue
        
          # Define custom markers and colors for each of the 3 groups
          markers = ["o", "s", "^"]  # circle, square, triangle
          colors = ["#f57a00", "#00f5d4", "#002df5"]  # distinct colors for each group
        
          axes[0].errorbar(
              x_pos,
              row["EMM"],
              yerr=[
                  [row["EMM"] - row["CI_lower"]],
                  [row["CI_upper"] - row["EMM"]],
              ],
              fmt=markers[x_pos],
              color=colors[x_pos],
              capsize=5,
              markersize=7,
          )
        
        # Configure axes for 3 distinct positions
        axes[0].set_xticks([0, 1, 2])
        axes[0].set_xticklabels(group_order)
        axes[0].set_xlim(-0.5, 2.5)  # Expanded padding for 3 ticks
        
        axes[0].set_ylabel("Adjusted EMM (nm)")
        axes[0].set_title("Muscle Temp2")
        
        # Optional: Add significance brackets between specific pairs (e.g., index 0 and 2)
         add_bracket(axes[0], 0, 2, y_val, h_val, "* p < 0.05")
        
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)

        #Subplot 3
        sub_size = df_emm[df_emm["Metabolite"] == "Muscle Temp3"]

        # Define your custom group order
        group_order = ["22°C", "15°C", "8°C"]
        
        for idx, row in sub_size.iterrows():
          # Map group names to integer x-positions (0, 1, 2)
          group_val = row["Group"]
          if group_val in group_order:
            x_pos = group_order.index(group_val)
          else:
            continue
        
          # Define custom markers and colors for each of the 3 groups
          markers = ["o", "s", "^"]  # circle, square, triangle
          colors = ["#f57a00", "#00f5d4", "#002df5"]  # distinct colors for each group
        
          axes[0].errorbar(
              x_pos,
              row["EMM"],
              yerr=[
                  [row["EMM"] - row["CI_lower"]],
                  [row["CI_upper"] - row["EMM"]],
              ],
              fmt=markers[x_pos],
              color=colors[x_pos],
              capsize=5,
              markersize=7,
          )
        
        # Configure axes for 3 distinct positions
        axes[0].set_xticks([0, 1, 2])
        axes[0].set_xticklabels(group_order)
        axes[0].set_xlim(-0.5, 2.5)  # Expanded padding for 3 ticks
        
        axes[0].set_ylabel("Adjusted EMM (nm)")
        axes[0].set_title("Muscle Temp3")
        
        # Optional: Add significance brackets between specific pairs (e.g., index 0 and 2)
        # add_bracket(axes[0], 0, 2, y_val, h_val, "* p < 0.05")
        
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)
        
        plt.suptitle("Estimated Marginal Means (Baseline Adjusted)", fontweight='bold')
        st.pyplot(fig)

    elif selected_view == '📄 RM-ANCOVA Model Summary (Main & Interaction Effects)':
        st.markdown("""
        <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px; font-size: 12px; line-height: 1.5; color: #856404;">
            <b>📄 Reviewer Note on Output Alignment:</b> Displaying all nominal significant results (<i>p</i><sub>raw</sub> &lt; 0.05) organized by Model Effect without row truncation.
            To inspect the primary <b>Jamovi statistical report</b>: 
            <a href="https://github.com/ernestonifade/GLYMREG-Extracellular-Vesicle-Study/raw/main/data/Jamovi_Statistical_Report_Figure3.pdf" target="_blank" style="color: #533f03; font-weight: bold; text-decoration: underline;">
                Download Jamovi PDF (GitHub) ↗
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        # --- SEARCH BAR INTEGRATION ---
        search_query = st.text_input("🔍 Search Protein / Cytokine Name:", key="unique_ancova_search_input").strip()
        
        df_ancova_fmt = ancova_df.copy()
        
        # Filter dataframe globally if search query is entered
        if search_query:
            col_name = 'Protein' if 'Protein' in df_ancova_fmt.columns else ('Cytokine' if 'Cytokine' in df_ancova_fmt.columns else None)
            if col_name:
                df_ancova_fmt = df_ancova_fmt[df_ancova_fmt[col_name].astype(str).str.contains(search_query, case=False, na=False)]
        
        if not df_ancova_fmt.empty:
            df_ancova_fmt['F_statistic'] = df_ancova_fmt['F_statistic'].round(2)
            df_ancova_fmt['Partial_Eta_Squared'] = df_ancova_fmt['Partial_Eta_Squared'].round(3)
            df_ancova_fmt['p_value_raw_fmt'] = df_ancova_fmt['p_value_raw'].apply(lambda p: f"{p:.4f}" if pd.notnull(p) and p >= 0.0001 else ("< 0.0001" if pd.notnull(p) else "N/A"))
            df_ancova_fmt['p_value_FDR_fmt'] = df_ancova_fmt['p_value_FDR'].apply(lambda p: f"{p:.4f}" if pd.notnull(p) and p >= 0.0001 else ("< 0.0001" if pd.notnull(p) else "N/A"))

        effect_map = [
            ('TimePoint:Group', '1. Time × Group Interaction Effects (p_raw < 0.05)'),
            ('Group', '2. Group / Sex Main Effects (p_raw < 0.05)'),
            ('TimePoint', '3. Time Main Effects (p_raw < 0.05)')
        ]

        cols_to_show = [
            'Protein', 'Effect', 'N', 'num_df', 'den_df', 
            'F_statistic', 'p_value_raw_fmt', 'p_value_FDR_fmt', 
            'Partial_Eta_Squared', 'Significant_FDR'
        ]
        rename_dict = {'p_value_raw_fmt': 'p_value_raw', 'p_value_FDR_fmt': 'p_value_FDR'}

        for eff_key, eff_title in effect_map:
            if eff_key == 'TimePoint:Group':
                sub = df_ancova_fmt[(df_ancova_fmt['Effect'].str.contains('TimePoint:Group', case=False, na=False)) & (df_ancova_fmt['p_value_raw'] < 0.05)].sort_values('p_value_raw')
            else:
                # Strict match for pure main effects so interaction rows don't bleed in
                sub = df_ancova_fmt[(df_ancova_fmt['Effect'] == eff_key) & (df_ancova_fmt['p_value_raw'] < 0.05)].sort_values('p_value_raw')
                
                st.markdown(f'<h4 style="margin-top:22px; margin-bottom:6px; color:#2c3e50;">{eff_title}</h4>', unsafe_allow_html=True)
            if not sub.empty:
                st.dataframe(sub[cols_to_show].rename(columns=rename_dict), use_container_width=True)
            else:
                st.markdown('<p style="font-size:11px; color:#7f8c8d; font-style:italic;">No proteins met nominal significance (p_raw &lt; 0.05) for this effect.</p>', unsafe_allow_html=True)

        ancova_note = """
        <div style="background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 14px; margin-top: 25px; border-radius: 4px; font-size: 12px; line-height: 1.6; color: #212529;">
            <b>📊 Notes on ANCOVA Model Terms & Column Layout:</b><br>
            • <b>Tables Filtered:</b> Showing all proteins meeting nominal significance (<i>p</i><sub>raw</sub> &lt; 0.05) with full expansion.<br>
            • <b>Side-by-Side Statistics:</b> <code>p_value_raw</code> (uncorrected ANOVA <i>p</i>) and <code>p_value_FDR</code> (Benjamini-Hochberg adjusted).<br>
            • <b>Partial_Eta_Squared (η<sub>p</sub>²):</b> Effect size estimate (Small ≈ 0.01, Medium ≈ 0.06, Large ≥ 0.14).
        </div>
        """
        st.markdown(ancova_note, unsafe_allow_html=True)

    elif selected_view == '🔍 Post-Hoc Pairwise Contrasts (emmeans)':

        search_query = st.text_input("🔍 Search Protein / Cytokine Name:", key="posthoc_search_bar").strip()
        
        df_ph_fmt = posthoc_df.copy()
        
        if search_query:
            col_name = 'Protein' if 'Protein' in df_ph_fmt.columns else ('Cytokine' if 'Cytokine' in df_ph_fmt.columns else None)
            if col_name:
                df_ph_fmt = df_ph_fmt[df_ph_fmt[col_name].astype(str).str.contains(search_query, case=False, na=False)]
        
        if not df_ph_fmt.empty:
            for col in ['estimate', 'std_error', 'df', 't_ratio']:
                if col in df_ph_fmt.columns:
                    df_ph_fmt[col] = pd.to_numeric(df_ph_fmt[col], errors='coerce').round(3)
            
            df_ph_fmt['p_value_raw_fmt'] = df_ph_fmt['p_value_raw'].apply(lambda p: f"{p:.4f}" if pd.notnull(p) and p >= 0.0001 else ("< 0.0001" if pd.notnull(p) else "N/A"))
            df_ph_fmt['p_value_FDR_fmt'] = df_ph_fmt['p_value_FDR'].apply(lambda p: f"{p:.4f}" if pd.notnull(p) and p >= 0.0001 else ("< 0.0001" if pd.notnull(p) else "N/A"))

        ph_cols = ['Protein', 'contrast', 'TimePoint', 'Group', 'estimate', 'std_error', 'df', 't_ratio', 'p_value_raw_fmt', 'p_value_FDR_fmt']
        ph_rename = {'p_value_raw_fmt': 'p_value_raw', 'p_value_FDR_fmt': 'p_value_FDR'}
        existing_cols = [c for c in ph_cols if c in df_ph_fmt.columns]

        between_sub = df_ph_fmt[(df_ph_fmt['contrast'].str.contains('Male|Female|22°C|8°C', case=False, na=False)) & 
                                (df_ph_fmt['p_value_raw'] < 0.05)].sort_values('p_value_raw')
        
        st.markdown('<h4 style="margin-top:15px; margin-bottom:6px; color:#2c3e50;">1. Between-Group Pairwise Contrasts (Male vs. Female by TimePoint)</h4>', unsafe_allow_html=True)
        if not between_sub.empty:
            st.dataframe(between_sub[existing_cols].rename(columns=ph_rename), use_container_width=True)
        else:
            st.markdown('<p style="font-size:11px; color:#7f8c8d; font-style:italic;">No between-group contrasts met nominal significance (p_raw &lt; 0.05).</p>', unsafe_allow_html=True)

        within_sub = df_ph_fmt[(~df_ph_fmt['contrast'].str.contains('Male|Female|22°C|8°C', case=False, na=False)) & 
                               (df_ph_fmt['p_value_raw'] < 0.05)].sort_values('p_value_raw')
        
        st.markdown('<h4 style="margin-top:25px; margin-bottom:6px; color:#2c3e50;">2. Within-Group Pairwise Contrasts (Recovery Time Shifting)</h4>', unsafe_allow_html=True)
        if not within_sub.empty:
            st.dataframe(within_sub[existing_cols].rename(columns=ph_rename), use_container_width=True)
        else:
            st.markdown('<p style="font-size:11px; color:#7f8c8d; font-style:italic;">No within-group contrasts met nominal significance (p_raw &lt; 0.05).</p>', unsafe_allow_html=True)

        posthoc_note = """
        <div style="background-color: #f8f9fa; border-left: 4px solid #28a745; padding: 14px; margin-top: 25px; border-radius: 4px; font-size: 12px; line-height: 1.6; color: #212529;">
            <b>📊 Notes on Pairwise Contrasts & Column Layout:</b><br>
            • <b>estimate:</b> Difference in Baseline-Adjusted Estimated Marginal Means (EMMs).<br>
            • <b>t_ratio & std_error:</b> Test statistic and standard error for the specified contrast.
        </div>
        """
        st.markdown(posthoc_note, unsafe_allow_html=True)

