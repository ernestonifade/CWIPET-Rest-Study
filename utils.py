import pandas as pd
import streamlit as st


def render_searchable_table(df, key_prefix, columns_to_show=None):
  if columns_to_show is None:
    columns_to_show = df.columns.tolist()

  # CSS to hide the native search toolbar button if desired
  st.markdown(
      """
        <style>
        [data-testid="stElementToolbar"] button[title*="Search"],
        [data-testid="stElementToolbar"] [aria-label*="Search"] {
            display: none !important;
        }
        </style>
    """,
      unsafe_allow_html=True,
  )

  search_key = f"{key_prefix}_search"

  # Initialize session state
  if search_key not in st.session_state:
    st.session_state[search_key] = ""


  # Define callback for the reset button to safely clear state before rerun
  def reset_search():
    st.session_state[search_key] = ""


  col1, col2 = st.columns([4, 1])

  with col1:
    # st.text_input automatically syncs with st.session_state[search_key]
    search_query = st.text_input(
        "Search table:",
        placeholder="Type to filter rows...",
        key=search_key,
        label_visibility="collapsed",
    )

  with col2:
    # Using on_click safely clears the state before the script re-runs
    st.button("Reset", key=f"{key_prefix}_reset", on_click=reset_search)

  # Filter DataFrame dynamically
  filtered_df = df.copy()
  current_query = st.session_state.get(search_key, "").strip().lower()

  if current_query:
    mask = False
    for col in filtered_df.columns:
      if filtered_df[col].dtype == object or pd.api.types.is_string_dtype(
          filtered_df[col]
      ):
        mask = mask | filtered_df[col].astype(str).str.lower().str.contains(
            current_query, na=False
        )
    filtered_df = filtered_df[mask]

  st.dataframe(
      filtered_df[columns_to_show], use_container_width=True, hide_index=True
  )
