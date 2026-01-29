import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="EcoSight AI: Carbon Tracker", page_icon="🌍", layout="wide")

# --- CONSTANTS & FACTORS ---
FACTORS = {
    "transport": {"Air Freight": 0.500, "Truck (Diesel)": 0.105, "Rail": 0.025, "Sea Freight": 0.010},
    "energy": {"Coal": 0.995, "Natural Gas": 0.420, "Grid Mix": 0.475, "Solar/Wind": 0.000}
}

STATE_MAP = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia"
}

# --- TITLE & SIDEBAR ---
st.title("🌍 EcoSight AI: Industrial Carbon & Energy Tracker")
st.markdown("""
    **Theme:** AI-Powered Carbon Footprint Tracker
    **data Sources:** EPA GHGP Facility data, World Energy Consumption & EIA Monthly data
""")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", [
    "Global Energy Trends",
    "Industrial Emissions Map & Analytics",
    "AI Emission Forecaster",
    "US Sector Deep Dive (Monthly)",
    "Live Calculator & Optimizer"
])


# # --- DATA LOADING ---
# @st.cache_data
# def load_data():
#     # 1. World Energy
#     try:
#         df_energy = pd.read_csv("data/world_energy.csv")
#     except FileNotFoundError:
#         st.error("File 'data/world_energy.csv' not found.")
#         df_energy = None
#
#     # 2. Facility data
#     try:
#         # Load with header=3 to skip metadata
#         df_facility = pd.read_csv("data/ghgp_data_2023.csv", header=3, low_memory=False)
#         df_facility.columns = df_facility.columns.str.strip()
#
#         # Clean Emission Columns (2011-2023)
#         year_cols = [str(y) + ' Total reported direct emissions' for y in range(2011, 2024)]
#         for col in year_cols:
#             if col in df_facility.columns:
#                 df_facility[col] = df_facility[col].astype(str).str.replace(',', '', regex=False)
#                 df_facility[col] = pd.to_numeric(df_facility[col], errors='coerce').fillna(0)
#     except FileNotFoundError:
#         st.error("File 'data/ghgp_data_2023.csv' not found.")
#         df_facility = None
#
#     # 3. US Monthly data
#     try:
#         df_us_monthly = pd.read_csv("data/co2_footprint.csv")
#         df_us_monthly['Value'] = pd.to_numeric(df_us_monthly['Value'], errors='coerce')
#         df_us_monthly = df_us_monthly.dropna(subset=['Value'])
#         df_us_monthly['YYYYMM'] = df_us_monthly['YYYYMM'].astype(str)
#         df_us_monthly['Date'] = pd.to_datetime(df_us_monthly['YYYYMM'], format='%Y%m', errors='coerce')
#         df_us_monthly = df_us_monthly.dropna(subset=['Date'])
#         df_us_monthly['Year'] = df_us_monthly['Date'].dt.year
#         df_us_monthly['Month'] = df_us_monthly['Date'].dt.month_name()
#     except FileNotFoundError:
#         df_us_monthly = None
#
#     return df_energy, df_facility, df_us_monthly

import os


# --- DATA LOADING FUNCTION ---
@st.cache_data
def load_data():
    # Helper to find file regardless of "data" vs "Data" casing
    def find_file(filename):
        # List of possible paths to check
        possible_paths = [
            f"data/{filename}",
            f"Data/{filename}",
            filename  # Check root directory just in case
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    # 1. World Energy
    path_energy = find_file("world_energy.csv")
    if path_energy:
        df_energy = pd.read_csv(path_energy)
    else:
        st.error("File 'world_energy.csv' not found in 'data/' or 'Data/' folder.")
        df_energy = None

    # 2. Facility Data
    path_facility = find_file("ghgp_data_2023.csv")
    if path_facility:
        # Load with header=3 to skip metadata
        df_facility = pd.read_csv(path_facility, header=3, low_memory=False)
        df_facility.columns = df_facility.columns.str.strip()

        # Clean Emission Columns
        year_cols = [str(y) + ' Total reported direct emissions' for y in range(2011, 2024)]
        for col in year_cols:
            if col in df_facility.columns:
                df_facility[col] = df_facility[col].astype(str).str.replace(',', '', regex=False)
                df_facility[col] = pd.to_numeric(df_facility[col], errors='coerce').fillna(0)
    else:
        st.error("File 'ghgp_data_2023.csv' not found.")
        df_facility = None

    # 3. US Monthly Data
    path_monthly = find_file("co2_footprint.csv")
    if path_monthly:
        df_us_monthly = pd.read_csv(path_monthly)
        df_us_monthly['Value'] = pd.to_numeric(df_us_monthly['Value'], errors='coerce')
        df_us_monthly = df_us_monthly.dropna(subset=['Value'])
        df_us_monthly['YYYYMM'] = df_us_monthly['YYYYMM'].astype(str)
        df_us_monthly['Date'] = pd.to_datetime(df_us_monthly['YYYYMM'], format='%Y%m', errors='coerce')
        df_us_monthly = df_us_monthly.dropna(subset=['Date'])
        df_us_monthly['Year'] = df_us_monthly['Date'].dt.year
        df_us_monthly['Month'] = df_us_monthly['Date'].dt.month_name()
    else:
        # Check if user maybe named it differently in the repo?
        st.warning("File 'co2_footprint.csv' not found.")
        df_us_monthly = None

    return df_energy, df_facility, df_us_monthly

df_energy, df_facility, df_us_monthly = load_data()

# --- PAGE 1: GLOBAL ENERGY TRENDS ---
if page == "Global Energy Trends":
    st.header("📊 Global Energy Intelligence Dashboard")

    if df_energy is not None and 'country' in df_energy.columns:
        countries = df_energy['country'].unique()
        selected_country = st.selectbox("Select a Country/Region", countries, index=0)
        c_data = df_energy[df_energy['country'] == selected_country].copy()
        c_data = c_data.sort_values('year')

        # 1. AI Forecast
        st.subheader(f"🔮 AI Forecast: Fossil vs. Renewables ({selected_country})")
        model_data = c_data.dropna(subset=['year', 'fossil_share_elec', 'renewables_share_elec'])

        if len(model_data) > 5:
            X = model_data['year'].values.reshape(-1, 1)
            y_fossil = model_data['fossil_share_elec'].values
            y_renew = model_data['renewables_share_elec'].values

            model_fossil = LinearRegression().fit(X, y_fossil)
            model_renew = LinearRegression().fit(X, y_renew)

            future_years = np.arange(model_data['year'].min(), 2036).reshape(-1, 1)
            pred_fossil = model_fossil.predict(future_years)
            pred_renew = model_renew.predict(future_years)

            fig_mix = go.Figure()
            fig_mix.add_trace(go.Scatter(x=model_data['year'], y=y_fossil, mode='lines', name='Fossil (Hist)',
                                         line=dict(color='red')))
            fig_mix.add_trace(go.Scatter(x=model_data['year'], y=y_renew, mode='lines', name='Renewables (Hist)',
                                         line=dict(color='green')))
            fig_mix.add_trace(go.Scatter(x=future_years.flatten(), y=pred_fossil, mode='lines', name='Fossil Forecast',
                                         line=dict(color='red', dash='dot')))
            fig_mix.add_trace(
                go.Scatter(x=future_years.flatten(), y=pred_renew, mode='lines', name='Renewables Forecast',
                           line=dict(color='green', dash='dot')))

            fig_mix.update_layout(xaxis_title="Year", yaxis_title="Share of Electricity (%)",
                                  title="Decarbonization Trajectory")
            st.plotly_chart(fig_mix, use_container_width=True)

            idx = np.argwhere(pred_renew > pred_fossil)
            if idx.size > 0:
                crossover_year = future_years[idx[0][0]][0]
                if crossover_year <= 2023:
                    st.success(f"✅ Renewables overtook Fossil Fuels in **{crossover_year}**!")
                else:
                    st.info(f"📅 Prediction: Renewables will overtake Fossil Fuels in **{crossover_year}**.")
        else:
            st.warning("Not enough data points for AI prediction.")

        # 2. RESTORED: Efficiency & Low Carbon Charts
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💰 Economic Energy Efficiency")
            st.markdown("Energy needed to generate wealth (Lower is Better)")
            if 'energy_per_gdp' in c_data.columns:
                fig_gdp = px.line(c_data, x='year', y='energy_per_gdp', title="Energy Intensity (kWh per $ GDP)")
                st.plotly_chart(fig_gdp, use_container_width=True)
            else:
                st.info("GDP data not available.")

        with col2:
            st.subheader("⚡ Low-Carbon Transition")
            st.markdown("Share of Electricity from Nuclear + Renewables")
            if 'low_carbon_share_elec' in c_data.columns:
                fig_lc = px.area(c_data, x='year', y='low_carbon_share_elec', title="Low Carbon Share (%)")
                st.plotly_chart(fig_lc, use_container_width=True)
            else:
                st.info("Low Carbon data not available.")

        # 3. Key Metrics
        st.markdown("---")
        latest = c_data.iloc[-1]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Population", f"{latest['population'] / 1e6:,.1f} M")
        k2.metric("CO2 per Capita", f"{latest.get('co2_per_capita', 0):.2f} tons")
        k3.metric("Fossil Share", f"{latest.get('fossil_share_energy', 0):.1f}%")
        k4.metric("Renewable Share", f"{latest.get('renewables_share_energy', 0):.1f}%")

    else:
        st.warning("Please upload 'world_energy.csv'.")

# --- PAGE 2: INDUSTRIAL EMISSIONS MAP & ANALYTICS ---
elif page == "Industrial Emissions Map & Analytics":
    st.header("🏭 Industrial Analytics Dashboard")
    st.markdown("Geospatial analysis and aggregated predictions for US Industrial Sectors.")

    if df_facility is not None and 'State' in df_facility.columns:
        # --- FILTERS ---
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            states = df_facility['State'].dropna().unique()
            selected_state_code = st.selectbox(
                "Filter by State",
                sorted(states),
                format_func=lambda x: f"{STATE_MAP.get(x, x)} ({x})"
            )

        with col_f2:
            # Sector Filter
            sectors = df_facility['Latest Reported Industry Type (sectors)'].dropna().unique()
            sector_options = ["All Industries"] + sorted(list(sectors))
            selected_sector = st.selectbox("Filter by Industry Sector", sector_options)

        # Filter Logic
        if selected_sector == "All Industries":
            filtered_df = df_facility[df_facility['State'] == selected_state_code].copy()
        else:
            filtered_df = df_facility[
                (df_facility['State'] == selected_state_code) &
                (df_facility['Latest Reported Industry Type (sectors)'] == selected_sector)
                ].copy()

        state_name = STATE_MAP.get(selected_state_code, selected_state_code)
        target_col = '2023 Total reported direct emissions'
        industry_col = 'Latest Reported Industry Type (sectors)'

        if not filtered_df.empty:
            # --- MAP SECTION ---
            st.subheader(f"📍 Facility Map: {state_name} ({selected_sector})")

            # KPI Cards
            total_emit = filtered_df[target_col].sum()
            avg_emit = filtered_df[target_col].mean()
            fac_count = len(filtered_df)

            k1, k2, k3 = st.columns(3)
            k1.metric("Total Emissions (2023)", f"{total_emit:,.0f} t")
            k2.metric("Average per Facility", f"{avg_emit:,.0f} t")
            k3.metric("Facility Count", f"{fac_count}")

            # Map Plot
            filtered_df['Latitude'] = pd.to_numeric(filtered_df['Latitude'], errors='coerce')
            filtered_df['Longitude'] = pd.to_numeric(filtered_df['Longitude'], errors='coerce')
            map_df = filtered_df.dropna(subset=['Latitude', 'Longitude'])

            fig_map = px.scatter_mapbox(
                map_df, lat="Latitude", lon="Longitude",
                hover_name="Facility Name", hover_data=["City", industry_col, target_col],
                color=target_col, size=target_col,
                color_continuous_scale=px.colors.cyclical.IceFire,
                size_max=40, zoom=5, height=500
            )
            fig_map.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig_map, use_container_width=True)

            # --- ANALYTICS SECTION ---
            st.markdown("---")
            st.subheader("📈 Regional Trends & AI Prediction")

            # Aggregated Historical Trend
            years_hist = list(range(2011, 2024))
            agg_emissions = []
            for y in years_hist:
                c = f"{y} Total reported direct emissions"
                if c in filtered_df.columns:
                    agg_emissions.append(filtered_df[c].sum())
                else:
                    agg_emissions.append(0)

            # AI Forecast for the Whole Region/Sector
            X_reg = np.array(years_hist).reshape(-1, 1)
            y_reg = np.array(agg_emissions)
            model_reg = LinearRegression().fit(X_reg, y_reg)

            future_years_reg = np.array([2024, 2025, 2026]).reshape(-1, 1)
            pred_reg = model_reg.predict(future_years_reg)

            # Plot Trend + Forecast
            all_years_reg = years_hist + [2024, 2025, 2026]
            all_vals_reg = list(agg_emissions) + list(pred_reg)
            types_reg = ['Historical'] * 13 + ['Predicted'] * 3

            trend_df = pd.DataFrame({'Year': all_years_reg, 'Emissions': all_vals_reg, 'Type': types_reg})

            fig_trend = px.line(trend_df, x='Year', y='Emissions', color='Type', markers=True,
                                title=f"Total Emissions Trend: {state_name} - {selected_sector}")
            st.plotly_chart(fig_trend, use_container_width=True)

            # --- LEADERBOARD ---
            c_lead, c_pie = st.columns(2)

            with c_lead:
                st.subheader("🏆 Top 10 Polluters")
                top_emitters = filtered_df.sort_values(target_col, ascending=False).head(10)
                fig_bar = px.bar(top_emitters, x=target_col, y="Facility Name", orientation='h',
                                 title="Highest Emitting Facilities (2023)", color=target_col)
                fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)

            with c_pie:
                if selected_sector == "All Industries":
                    st.subheader("📊 Sector Breakdown")
                    fig_pie = px.pie(filtered_df, names=industry_col, values=target_col,
                                     title=f"Emissions by Sector in {state_name}")
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("Select 'All Industries' to see the Sector Breakdown pie chart.")

            # --- FACILITY FACE-OFF ---
            st.markdown("---")
            st.subheader("⚔️ Facility Face-Off")
            col1, col2 = st.columns(2)
            with col1:
                f1 = st.selectbox("Select Facility A", filtered_df['Facility Name'].unique(), key="f1")
                v1 = filtered_df[filtered_df['Facility Name'] == f1][target_col].values[0]
                st.metric(f1, f"{v1:,.0f} t")
            with col2:
                f2 = st.selectbox("Select Facility B", filtered_df['Facility Name'].unique(), key="f2")
                v2 = filtered_df[filtered_df['Facility Name'] == f2][target_col].values[0]
                st.metric(f2, f"{v2:,.0f} t")

            diff = v1 - v2
            if diff > 0:
                st.warning(f"**{f1}** emits **{diff:,.0f} tons MORE** than {f2}.")
            else:
                st.success(f"**{f1}** emits **{abs(diff):,.0f} tons LESS** than {f2}.")

        else:
            st.warning("No facilities found for this selection.")
    else:
        st.warning("Please upload 'ghgp_data_2023.csv'.")

# --- PAGE 3: AI EMISSION FORECASTER (FACILITY LEVEL) ---
elif page == "AI Emission Forecaster":
    st.header("🤖 AI-Powered Emission Prediction")

    if df_facility is not None:
        fac_list = df_facility['Facility Name'].unique()
        sel_fac = st.selectbox("Search Facility", fac_list)
        row = df_facility[df_facility['Facility Name'] == sel_fac].iloc[0]

        years = np.array(range(2011, 2024)).reshape(-1, 1)
        emissions = []
        for y in range(2011, 2024):
            c = f"{y} Total reported direct emissions"
            if c in df_facility.columns:
                emissions.append(row[c])
            else:
                emissions.append(0)

        model = LinearRegression().fit(years, emissions)
        r2 = model.score(years, emissions)
        future = np.array([2024, 2025, 2026]).reshape(-1, 1)
        pred = model.predict(future)

        # Plot
        all_y = list(range(2011, 2027))
        all_e = list(emissions) + list(pred)
        types = ['Historical'] * 13 + ['Predicted'] * 3
        chart_data = pd.DataFrame({'Year': all_y, 'Emissions': all_e, 'Type': types})

        fig = px.line(chart_data, x='Year', y='Emissions', color='Type', markers=True, title=f"Forecast: {sel_fac}")
        st.plotly_chart(fig, use_container_width=True)

        curr = emissions[-1]
        fut = pred[1]
        change = ((fut - curr) / curr) * 100 if curr != 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("2023 Actual", f"{curr:,.0f} t")
        c2.metric("2025 Predicted", f"{fut:,.0f} t", f"{change:.1f}%")
        c3.metric("Model Confidence (R²)", f"{r2:.2f}")

        # Audit Report
        st.markdown("---")
        st.subheader("📄 Automated Carbon Audit Report")
        report = f"""
        *** CARBON AUDIT: {sel_fac} ***
        Location: {row['City']}, {row['State']}
        Industry: {row['Latest Reported Industry Type (sectors)']}
        --------------------------------
        Current Emissions (2023): {curr:,.2f} t
        Predicted (2025): {fut:,.2f} t
        Trend: {change:.1f}%
        Model Accuracy: {r2:.2f}
        --------------------------------
        Generated by EcoSight AI
        """
        st.download_button("Download Report", report, file_name=f"{sel_fac}_Audit.txt")

# --- PAGE 4: US MONTHLY DEEP DIVE ---
elif page == "US Sector Deep Dive (Monthly)":
    st.header("🇺🇸 US Energy Sector Trends (Monthly)")

    if df_us_monthly is not None:
        sectors = df_us_monthly['Description'].unique()
        sel_sector = st.selectbox("Select Sector", sectors)

        col_y, _ = st.columns([1, 3])
        with col_y:
            yrs = sorted(df_us_monthly['Year'].unique(), reverse=True)
            sel_yr = st.selectbox("Select Year", ["All Years"] + list(yrs))

        base = df_us_monthly[df_us_monthly['Description'] == sel_sector].copy()
        chart_data = base[base['Year'] == sel_yr].sort_values('Date') if sel_yr != "All Years" else base.sort_values(
            'Date')

        if not chart_data.empty:
            st.subheader(f"📈 Emission Trend ({sel_yr})")
            fig_t = px.area(chart_data, x='Date', y='Value', title=f"Emissions: {sel_sector}")
            st.plotly_chart(fig_t, use_container_width=True)

            # Stats
            total = chart_data['Value'].sum()
            avg = chart_data['Value'].mean()
            peak = chart_data['Value'].max()

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Emissions", f"{total:,.1f} M Tons")
            m2.metric("Monthly Average", f"{avg:,.1f} M Tons")
            m3.metric("Peak Value", f"{peak:,.1f} M Tons")

            if sel_yr == "All Years":
                st.subheader("🔥 Seasonality Heatmap")
                try:
                    piv = chart_data.pivot_table(index='Year', columns='Month', values='Value', aggfunc='sum')
                    piv = piv.reindex(
                        columns=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
                                 'October', 'November', 'December'])
                    fig_h = px.imshow(piv, color_continuous_scale='Magma')
                    st.plotly_chart(fig_h, use_container_width=True)
                except:
                    st.warning("Not enough data for heatmap.")
        else:
            st.warning("No data.")
    else:
        st.warning("Please upload 'co2_footprint.csv'.")

# --- PAGE 5: LIVE CALCULATOR & OPTIMIZER (RESTORED) ---
elif page == "Live Calculator & Optimizer":
    st.header("🧮 Live Carbon Calculator & AI Optimizer")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🚛 Logistics Calculator")
        cargo_weight = st.number_input("Cargo Weight (tons)", 10.0)
        distance = st.number_input("Distance (km)", 1000.0)
        transport_mode = st.selectbox("Transport Mode", list(FACTORS["transport"].keys()))
        logistics_co2 = cargo_weight * distance * FACTORS["transport"][transport_mode]
        st.info(f"Logistics Footprint: **{logistics_co2:,.2f} kg CO2e**")

    with col2:
        st.subheader("⚡ Energy Goal Setting")
        energy_usage = st.number_input("Monthly Energy Usage (kWh)", 5000.0)
        energy_source = st.selectbox("Energy Source", list(FACTORS["energy"].keys()))

        energy_co2 = energy_usage * FACTORS["energy"][energy_source]
        st.info(f"Current Footprint: **{energy_co2:,.2f} kg CO2e**")

        # RESTORED: GAMIFIED SLIDER
        st.write("**🎯 Set Reduction Goal**")
        reduction_target = st.slider("Target Reduction %", 0, 100, 20)
        target_emissions = energy_co2 * (1 - (reduction_target / 100))
        st.write(f"Target: **{target_emissions:,.2f} kg**")

        if reduction_target > 0:
            st.caption(
                f"💡 Save **{(energy_co2 - target_emissions):,.0f} kg** by switching **{(reduction_target / 100) * energy_usage:.0f} kWh** to Solar.")

    st.markdown("---")
    st.subheader("📊 Combined Footprint")
    total_co2 = logistics_co2 + energy_co2
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        fig_pie = px.pie(names=["Logistics", "Energy"], values=[logistics_co2, energy_co2],
                         title=f"Total: {total_co2:,.0f} kg")
        st.plotly_chart(fig_pie, use_container_width=True)

    with res_col2:
        # RESTORED: AI SUGGESTIONS
        st.markdown("#### 💡 AI Recommendations")
        best_transport = min(FACTORS["transport"], key=FACTORS["transport"].get)
        if transport_mode != best_transport:
            savings = (logistics_co2 - (cargo_weight * distance * FACTORS["transport"][best_transport]))
            st.warning(f"⚠️ Switch logistics to **{best_transport}** to save **{savings:,.0f} kg CO2**.")
        else:
            st.success("✅ Logistics are optimized!")

st.markdown("---")
st.markdown("Developed for AI for Climate Change Hackathon")




