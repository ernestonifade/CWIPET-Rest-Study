import io
import os
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from utils import render_searchable_table
import warnings

warnings.filterwarnings("ignore")

plt.rcParams["path.simplify"] = True
plt.rcParams["path.simplify_threshold"] = 1.0
plt.rcParams["agg.path.chunksize"] = 10000


def render_pathway_enrichment_bubble_from_df(
    results_input=None,
    database_name="GO Biological Process",
    max_pvalue=0.05,
    pathway_indices=None,
):
    mpl.rcParams["svg.fonttype"] = "none"

    # 1. Handle dynamic file loading based on input path, list, or DataFrame
    if results_input is None or isinstance(
        results_input, (str, os.PathLike, list)
    ):
        if isinstance(results_input, list):
            file_candidates = results_input
        elif isinstance(results_input, (str, os.PathLike)):
            file_candidates = [
                str(results_input),
                "results/enrichment_results_for_correlating_metabolites.csv",
                "results/enrichment_results_for_correlating_cytokines.csv",
            ]
        else:
            file_candidates = [
                "results/enrichment_results_for_correlating_metabolites.xlsx",
                "results/enrichment_results_for_correlating_metabolites.csv",
                "../results/enrichment_results_for_correlating_cytokines.csv",
            ]

        filepath = None
        for path in file_candidates:
            if os.path.exists(path):
                filepath = path
                break

        if filepath is None:
            st.warning(
                "⚠️ Pathway results file not found. Please check your GitHub file path."
            )
            return None

        # Load dataframe based on file extension
        if filepath.endswith(".xlsx") or filepath.endswith(".xls"):
            df_master = pd.read_excel(filepath)
        else:
            df_master = pd.read_csv(filepath)
    else:
        df_master = results_input.copy()

    df = df_master.copy()
    title_suffix = database_name

    # 3. Apply clean index-based selection if specified
    if pathway_indices is not None and not df.empty:
        df = df.reset_index(drop=True)
        valid_indices = [i for i in pathway_indices if i < len(df)]
        df = df.iloc[valid_indices].copy()
        title_suffix = f"{database_name} (Indexed Selection)"

    if df.empty:
        st.warning(
            "⚠️ No pathways meet the significance threshold (p ≤ "
            f"{max_pvalue}) or valid indices."
        )
        return None

    # Map column names & calculations safely
    df["Pathway Name"] = df["Pathway name"]
    df["Significance"] = df["Entities FDR"]
    df["Number of Molecules Enriched"] = df["#Entities found"]
    df["Enrichment_Ratio"] = df["Log2_Enrichment_Ratio"].replace(0, 0.001)
    df["-log10Sig"] = -np.log10(df["Significance"].astype(float).clip(lower=1e-15))
    df = df.sort_values("-log10Sig", ascending=True)

    fig, ax = plt.subplots(figsize=(4.0, max(4.0, len(df) * 0.25)))
    df["PlotSize"] = (df["Number of Molecules Enriched"] + 1) * 35
    df["PlotSize"] = df["PlotSize"].clip(lower=60, upper=300)

    scatter = ax.scatter(
        x=df["Enrichment_Ratio"],
        y=df["Pathway Name"],
        s=df["PlotSize"],
        c=df["-log10Sig"],
        cmap="viridis",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
        zorder=3,
    )

    ax.axvline(
        x=1.0,
        color="red",
        linestyle=":",
        linewidth=1.2,
        label="Expected",
        alpha=0.8,
        zorder=1,
    )

    v_min, v_max = df["-log10Sig"].min(), df["-log10Sig"].max()
    cmap = plt.cm.viridis
    norm = plt.Normalize(v_min, v_max)
    color_vals = np.linspace(v_min, v_max, 3)

    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(norm(v)),
            markersize=6,
            markeredgecolor="black",
            label=f"{v:.1f}",
        )
        for v in color_vals
    ]

    legend_col = ax.legend(
        handles=color_handles[::-1],
        title=r"$\mathbf{-\log_{10}(FDR)}$",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=False,
        labelspacing=1.2,
        prop={"weight": "bold", "size": 6.5},
    )

    s_min, s_max = int(df["Number of Molecules Enriched"].min()), int(
        df["Number of Molecules Enriched"].max()
    )
    size_steps = np.unique(np.linspace(s_min, s_max, 3).astype(int))

    size_handles = []
    for s in size_steps:
        plot_size = (s + 1) * 35
        marker_d = np.sqrt(plot_size)
        size_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=marker_d,
                markeredgecolor="black",
                label=str(s),
            )
        )

    legend_siz = ax.legend(
        handles=size_handles[::-1],
        title=r"$\mathbf{Qty. Enriched}$",
        bbox_to_anchor=(1.02, 0.45),
        loc="upper left",
        frameon=False,
        labelspacing=1.4,
        prop={"weight": "bold", "size": 6.5},
    )

    ax.add_artist(legend_col)
    ax.set_xlabel(
        "Log2 Enrichment Ratio (Obs / Exp)\n(>1 = Enriched, <1 = Depleted)",
        fontweight="bold",
        fontsize=7.5,
    )
    ax.set_title(
        f"Top Enriched Pathways ({title_suffix})",
        y=1.03,
        fontweight="bold",
        fontsize=8.5,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle="--", alpha=0.5, zorder=0)
    ax.tick_params(axis="both", labelsize=7)
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    return fig


def find_pathway_file(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# --- MAIN RENDER FUNCTION FOR STREAMLIT ---
def render_figure3():
    prot_candidates = [
        "results/enrichment_results_for_correlating_metabolites.csv",
        "../results/enrichment_results_for_correlating_metabolites.csv",
        "enrichment_results_for_correlating_metabolites.xlsx",
        "enrichment_results_for_correlating_metabolites.csv",
    ]
    cyt_candidates = [
        "results/enrichment_results_for_correlating_cytokines.csv",
        "../results/enrichment_results_for_correlating_cytokines.csv",
        "../enrichment_results_for_correlating_cytokines.csv",
        "enrichment_results_for_correlating_cytokines.xlsx",
        "enrichment_results_for_correlating_cytokines.csv",
    ]

    prot_path = find_pathway_file(prot_candidates)
    cyt_path = find_pathway_file(cyt_candidates)

    selected_view = st.selectbox(
        "Select Section View:",
        [
            "🧬 Pathway Enrichment: Correlating Metabolites",
            "📄 Pathway Enrichment: Correlating Cytokines",
            "📋 Metabolite Pathway Enrichment Summary Table",
            "📋 Cytokine Pathway Enrichment Summary Table",
        ],
    )

    st.markdown("---")

    if selected_view == "🧬 Pathway Enrichment: Correlating Metabolites":
        st.markdown(
            """
            <div style="background-color: #e2e3e5; border-left: 4px solid #383d41; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px; font-size: 11px; color: #383d41;">
                <b>Pathway Enrichment (Metabolites):</b> Over-Representation Analysis mapping correlating candidate metabolites to functional pathways.
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig_path = render_pathway_enrichment_bubble_from_df(
            prot_path, database_name="Metabolite Correlates"
        )
        if fig_path:
            st.pyplot(fig_path)
            
            buf = io.BytesIO()
            fig_path.savefig(buf, format="svg", bbox_inches="tight")
            buf.seek(0)
            
            st.download_button(
                label="📥 Download Editable Vector (SVG)",
                data=buf,
                file_name="pathway_enrichment_metabolite_corr.svg",
                mime="image/svg+xml",
            )

    elif selected_view == "📄 Pathway Enrichment: Correlating Cytokines":
        st.markdown(
            """
            <div style="background-color: #e2e3e5; border-left: 4px solid #383d41; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px; font-size: 11px; color: #383d41;">
                <b>Pathway Enrichment (Cytokines):</b> Over-Representation Analysis mapping correlating cytokines to functional biological networks.
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig_path = render_pathway_enrichment_bubble_from_df(
            cyt_path, database_name="Cytokine Correlates"
        )
        if fig_path:
            st.pyplot(fig_path)
            
            buf = io.BytesIO()
            fig_path.savefig(buf, format="svg", bbox_inches="tight")
            buf.seek(0)
            
            st.download_button(
                label="📥 Download Editable Vector (SVG)",
                data=buf,
                file_name="pathway_enrichment_cytokine_corr.svg",
                mime="image/svg+xml",
            )

    elif selected_view == "📋 Metabolite Pathway Enrichment Summary Table":
        st.markdown(
            """
            <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px; font-size: 12px; line-height: 1.5; color: #856404;">
                <b>📄 Metabolite Pathway Summary Table:</b> Complete statistical enrichment metrics for metabolite correlates.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if prot_path and os.path.exists(prot_path):
            if prot_path.endswith(".xlsx") or prot_path.endswith(".xls"):
                master_results_df = pd.read_excel(prot_path)
            else:
                master_results_df = pd.read_csv(prot_path)

            display_df = master_results_df[
                (master_results_df["#Entities found"] > 0)
                & (master_results_df["Entities FDR"] <= 0.05)
            ].sort_values(by="Entities FDR")

            render_searchable_table(
                df_input=display_df,
                title_prefix="Metabolite Pathways",
            )
        else:
            st.warning("⚠️ Results file not found in GitHub paths.")

    elif selected_view == "📋 Cytokine Pathway Enrichment Summary Table":
        st.markdown(
            """
            <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px; font-size: 12px; line-height: 1.5; color: #856404;">
                <b>📄 Cytokine Pathway Summary Table:</b> Complete statistical enrichment metrics for cytokine correlates.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if cyt_path and os.path.exists(cyt_path):
            if cyt_path.endswith(".xlsx") or cyt_path.endswith(".xls"):
                master_results_df = pd.read_excel(cyt_path)
            else:
                master_results_df = pd.read_csv(cyt_path)

            display_df = master_results_df[
                (master_results_df["#Entities found"] > 0)
                & (master_results_df["Entities FDR"] <= 0.05)
            ].sort_values(by="Entities FDR")

            render_searchable_table(
                df_input=display_df,
                title_prefix="Cytokine Pathways",
            )
        else:
            st.warning("⚠️ Results file not found in GitHub paths.")


def load_fig3_results():
    """Helper loader for Figure 3 pathway data to satisfy app.py imports."""
    return {}
