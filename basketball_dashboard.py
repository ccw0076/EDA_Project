import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats as scipy_stats

st.set_page_config(page_title="CBB Dashboard", layout="wide")
st.title("College Basketball EDA Dashboard (2010-2025)")

# Load your data
df_all = pd.read_csv('final_data')

# Calculate additional columns if they don't exist
if "PPG" not in df_all.columns:
    df_all['PPG'] = (df_all['PTS'] / df_all['G']).round(2)
if "OPPG" not in df_all.columns:
    df_all['OPPG'] = (df_all['Opp PTS'] / df_all['G']).round(2)
if "NET" not in df_all.columns:
    df_all["NET"] = df_all["PTS"] - df_all["Opp PTS"]
if "Avg Point Dif" not in df_all.columns:
    df_all["Avg Point Dif"] = (df_all["PPG"] - df_all["OPPG"]).round(2)

# Create tabs for all questions
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Overview", 
    "Q1: Win Correlations", 
    "Q2: Trends Over Time",
    "Q3: SEC Signatures",
    "Q4: Outliers",
    "Q5: Point Differential",
    "Q6: Power Conferences",
    "Q7: Auburn vs Champions"
])

# ===== TAB 1: OVERVIEW =====
with tab1:
    st.header("Data Overview")
    st.write(f"**Total Records:** {len(df_all)}")
    st.write(f"**Seasons Covered:** 2010-2025")
    st.write(f"**Number of Teams:** {df_all['School'].nunique()}")
    st.dataframe(df_all.head(10))

# ===== TAB 2: Q1 - CORRELATIONS WITH WIN PERCENTAGE =====
with tab2:
    st.header("Q1: Which Statistics Correlate With Win Percentage?")
    
    # Calculate correlations
    numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
    if "W-L%" in numeric_cols:
        correlation_data = df_all[numeric_cols].corrwith(df_all["W-L%"]).sort_values()
        
        # Remove W-L%, W, and L from the chart
        correlation_data = correlation_data.drop(["W-L%", "W", "L"], errors='ignore')
        
        # Create correlation chart
        fig_corr = go.Figure(data=[
            go.Bar(y=correlation_data.index, x=correlation_data.values, orientation='h',
                   marker=dict(color=correlation_data.values, colorscale='RdBu', showscale=True))
        ])
        fig_corr.update_layout(title="Correlation with Win Percentage", 
                               xaxis_title="Correlation", yaxis_title="Statistic",
                               height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Scatterplots for key stats
        st.subheader("Key Stats vs Win Percentage")
        key_stats = ["SRS", "FG%", "PTS", "TOV", "Opp PTS", "FT%"]
        
        cols = st.columns(3)
        for i, stat in enumerate(key_stats):
            with cols[i % 3]:
                if stat in df_all.columns:
                    fig = px.scatter(df_all, x=stat, y="W-L%", trendline="ols",
                                    title=f"{stat} vs Win %", opacity=0.5)
                    fig.data[1].line.color = 'darkred'
                    fig.data[1].line.width = 3
                    st.plotly_chart(fig, use_container_width=True)

# ===== TAB 3: Q2 - TRENDS OVER TIME =====
with tab3:
    st.header("Q2: Are Statistics Shifting Over Time?")
    
    # Calculate seasonal averages
    season_trends = df_all.groupby("Season").mean(numeric_only=True).reset_index()
    
    # Plot seasonal trends
    stats_to_plot = ["PTS", "Opp PTS", "3PA", "3P", "TOV", "FGA", "FG"]
    available_stats = [s for s in stats_to_plot if s in season_trends.columns]
    
    fig_trends = go.Figure()
    for stat in available_stats:
        fig_trends.add_trace(go.Scatter(x=season_trends["Season"], y=season_trends[stat],
                                        mode='lines+markers', name=stat))
    
    fig_trends.update_layout(title="League Averages Over Time (2010-2025)",
                            xaxis_title="Season", yaxis_title="Average Value",
                            hovermode='x unified', height=500)
    st.plotly_chart(fig_trends, use_container_width=True)
    
    st.info("Note: 2020-2021 was affected by COVID-19 with modified schedules")

# ===== TAB 4: Q3 - SEC TEAM SIGNATURES =====
with tab4:
    st.header("Q3: Do Certain SEC Teams Have Statistical 'Signatures'?")
    
    sec_teams = [
        "Alabama", "Arkansas", "Auburn", "Florida", "Georgia", "Kentucky",
        "Louisiana State", "Mississippi State", "Mississippi", "Missouri",
        "South Carolina", "Tennessee", "Texas A&M", "Vanderbilt"
    ]
    
    sec_df = df_all[df_all["School"].isin(sec_teams)].copy()
    
    if len(sec_df) > 0:
        sec_stat_cols = [ 
            "PTS", "Opp PTS", "FG%", "3P", "3PA", "FT%",
            "TRB", "AST", "STL", "BLK", "TOV", "SRS", "SOS"
        ]
        available_cols = [c for c in sec_stat_cols if c in sec_df.columns]
        
        # Team averages
        sec_team_stats = sec_df.groupby("School")[available_cols].mean()
        
        # Standardize with z-scores
        sec_team_z = (sec_team_stats - sec_team_stats.mean()) / sec_team_stats.std()
        
        # Create heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=sec_team_z.values,
            x=sec_team_z.columns,
            y=sec_team_z.index,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Z-Score")
        ))
        fig_heatmap.update_layout(title="SEC Team Statistical Signatures (Z-Scores)",
                                 height=600)
        st.plotly_chart(fig_heatmap, use_container_width=True)

# ===== TAB 5: Q4 - OUTLIERS =====
with tab5:
    st.header("Q4: Outliers Per Season & 2024-2025 Outliers")
    
    # Calculate NET and z-scores
    df_all["NET_z"] = (df_all["NET"] - df_all["NET"].mean()) / df_all["NET"].std()
    
    # Scatterplot of outliers over time
    df_plot = df_all.copy()
    df_plot["Is Outlier"] = df_plot["NET_z"].abs() > 2
    
    fig_outliers = px.scatter(df_plot, x="Season", y="NET", color="Is Outlier",
                             title="Outlier Teams by Net Points (PTS – Opp PTS)",
                             color_discrete_map={True: 'red', False: 'gray'},
                             labels={"Is Outlier": "Is Outlier (|z| > 2)"})
    fig_outliers.update_layout(height=500)
    st.plotly_chart(fig_outliers, use_container_width=True)
    
    # 2024-2025 outliers
    st.subheader("2024-2025 Season Outliers")
    df_2025 = df_all[df_all["Season"] == "2024-2025"].copy()
    
    if len(df_2025) > 0:
        df_2025["NET_z_season"] = (df_2025["NET"] - df_2025["NET"].mean()) / df_2025["NET"].std()
        outliers_2025 = df_2025[df_2025["NET_z_season"].abs() > 2][["School", "NET", "NET_z_season"]].sort_values("NET_z_season", ascending=False)
        
        if len(outliers_2025) > 0:
            st.dataframe(outliers_2025.rename(columns={"NET_z_season": "Z-Score"}))
        else:
            st.write("No outliers found in 2024-2025 season")

# ===== TAB 6: Q5 - POINT DIFFERENTIAL VS WIN % =====
with tab6:
    st.header("Q5: How Does Point Differential Correlate With Win Percentage?")
    
    fig_pt_dif = px.scatter(df_all, x="Avg Point Dif", y="W-L%",
                           title="Point Differential vs Win Percentage",
                           trendline="ols",
                           labels={"Avg Point Dif": "Average Point Differential",
                                  "W-L%": "Win Percentage"},
                           opacity=0.6)
    fig_pt_dif.data[1].line.color = 'darkred'
    fig_pt_dif.data[1].line.width = 3
    fig_pt_dif.update_layout(height=500)
    st.plotly_chart(fig_pt_dif, use_container_width=True)

# ===== TAB 7: Q6 - POWER 5 CONFERENCES =====
with tab7:
    st.header("Q6: How Do Power Conferences Compare?")
    
    # Map conferences
    conference_map = {
        "Alabama": "SEC", "Auburn": "SEC", "Arkansas": "SEC",
        "Florida": "SEC", "Georgia": "SEC", "Kentucky": "SEC",
        "Louisiana State": "SEC", "Mississippi": "SEC", "Mississippi State": "SEC",
        "Missouri": "SEC", "South Carolina": "SEC", "Tennessee": "SEC", 
        "Texas A&M": "SEC", "Vanderbilt": "SEC", "Texas": "SEC", "Oklahoma": "SEC",
        "Duke": "ACC", "North Carolina": "ACC", "Virginia": "ACC",
        "Illinois": "Big 10", "Indiana": "Big 10", "Iowa": "Big 10",
        "Michigan": "Big 10", "Michigan State": "Big 10", "Ohio State": "Big 10",
        "Purdue": "Big 10", "Wisconsin": "Big 10",
        "Kansas": "Big 12", "Baylor": "Big 12", "Houston": "Big 12",
        "Iowa State": "Big 12", "TCU": "Big 12", "Texas Tech": "Big 12",
        "Villanova": "Big East", "Connecticut": "Big East", "Xavier": "Big East",
        "Creighton": "Big East", "Marquette": "Big East"
    }
    
    df_temp = df_all.copy()
    df_temp["Conference"] = df_temp["School"].map(conference_map)
    power5_df = df_temp[df_temp["Conference"].notna()]
    
    if len(power5_df) > 0:
        st.subheader("Select a Statistic to Analyze")
        stat_choice = st.selectbox("Statistic", ["SRS", "SOS", "PPG", "OPPG", "FG%", "3P%", "FT%"])
        
        if stat_choice in power5_df.columns:
            fig_conf = px.box(power5_df, x="Conference", y=stat_choice,
                            title=f"{stat_choice} Distribution by Conference",
                            color="Conference")
            st.plotly_chart(fig_conf, use_container_width=True)
        
        # Scatterplot option
        st.subheader("Compare Statistics vs Win Percentage")
        scatter_stat = st.selectbox("Select Statistic for Scatterplot", 
                                   ["SRS", "SOS", "PPG", "OPPG", "FG%", "3P%", "FT%"],
                                   key="scatter_stat")
        
        if scatter_stat in power5_df.columns:
            fig_scatter = px.scatter(power5_df, x=scatter_stat, y="W-L%",
                                   color="Conference",
                                   title=f"{scatter_stat} vs Win Percentage by Conference",
                                   trendline="ols",
                                   opacity=0.6)
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)

# ===== TAB 8: Q7 - AUBURN VS CHAMPIONS =====
with tab8:
    st.header("Q7: How Did Auburn Compare to Champions?")
    
    # Champion mapping
    champion_map = {
        "2010-2011": "Connecticut",
        "2011-2012": "Kentucky",
        "2012-2013": "Louisville",
        "2013-2014": "Connecticut",
        "2014-2015": "Duke",
        "2015-2016": "Villanova",
        "2016-2017": "North Carolina",
        "2017-2018": "Villanova",
        "2018-2019": "Virginia",
        "2019-2020": "No Champ",
        "2020-2021": "Baylor",
        "2021-2022": "Kansas",
        "2022-2023": "Connecticut",
        "2023-2024": "Connecticut",
        "2024-2025": "Florida",
    }
    
    # Get Auburn data
    auburn_all = df_all[df_all["School"] == "Auburn"].copy()
    auburn_all = auburn_all.sort_values("Season", ascending=False)
    
    if len(auburn_all) > 0:
        # Show season data
        st.subheader("Auburn Season Data (2010-2024)")
        st.dataframe(auburn_all[["Season", "W-L%", "PPG", "OPPG", "FG%", "SRS", "BLK", "STL", "TRB", "AST", "TOV"]].sort_values("Season", ascending=False), use_container_width=True)
        
        # Auburn win percentage line chart
        st.subheader("Auburn Win Percentage Over Time")
        auburn_line = auburn_all[["Season", "W-L%"]].copy().sort_values("Season")
        fig_auburn = px.line(auburn_line, x="Season", y="W-L%",
                           title="Auburn Win Percentage Over Time (2010-2024)",
                           markers=True)
        fig_auburn.update_layout(height=400)
        st.plotly_chart(fig_auburn, use_container_width=True)
        
        # Radar plots for Auburn vs Champions
        st.subheader("Auburn vs Champion Radar Charts")
        
        available_seasons = [s for s in sorted(auburn_all["Season"].unique(), reverse=True) if s in champion_map and champion_map[s] != "No Champ"]
        
        if len(available_seasons) > 0:
            selected_season = st.selectbox("Select Season to Compare", available_seasons)
            
            # Get Auburn stats for selected season
            aub_season = auburn_all[auburn_all["Season"] == selected_season]
            
            if len(aub_season) > 0:
                champ_name = champion_map.get(selected_season, "Unknown")
                champ_season = df_all[(df_all["Season"] == selected_season) & (df_all["School"] == champ_name)]
                
                if len(champ_season) > 0:
                    stats_to_plot = ["W-L%", "PPG", "BLK", "STL", "TRB", "AST"]
                    
                    # Get data
                    aub_values = []
                    champ_values = []
                    
                    for stat in stats_to_plot:
                        if stat in aub_season.columns and stat in champ_season.columns:
                            aub_values.append(float(aub_season[stat].values[0]) if len(aub_season[stat].values) > 0 else 0)
                            champ_values.append(float(champ_season[stat].values[0]) if len(champ_season[stat].values) > 0 else 0)
                    
                    if len(aub_values) > 0 and len(champ_values) > 0:
                        # Normalize
                        aub_array = np.array(aub_values, dtype=float)
                        champ_array = np.array(champ_values, dtype=float)
                        max_vals = np.maximum(aub_array, champ_array)
                        aub_norm = np.divide(aub_array, max_vals, where=max_vals!=0, out=np.zeros_like(aub_array))
                        champ_norm = np.divide(champ_array, max_vals, where=max_vals!=0, out=np.zeros_like(champ_array))
                        
                        # Create radar
                        fig_radar = go.Figure()
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=aub_norm,
                            theta=stats_to_plot,
                            fill='toself',
                            name='Auburn',
                            line=dict(color='orange'),
                            fillcolor='rgba(255, 165, 0, 0.3)'
                        ))
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=champ_norm,
                            theta=stats_to_plot,
                            fill='toself',
                            name=champ_name,
                            line=dict(color='blue'),
                            fillcolor='rgba(0, 0, 255, 0.3)'
                        ))
                        
                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            title=f"Auburn vs {champ_name} ({selected_season})",
                            height=600
                        )
                        
                        st.plotly_chart(fig_radar, use_container_width=True)
                    else:
                        st.warning(f"Could not find complete stats for comparison in {selected_season}")
                else:
                    st.warning(f"Champion {champ_name} not found in data for {selected_season}")
        else:
            st.info("No valid season-champion pairs available for radar charts")
    else:
        st.info("No Auburn data available")