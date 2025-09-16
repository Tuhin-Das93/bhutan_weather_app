import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit.components.v1 import html
import requests
import os
import plotly.express as px
from geopy.distance import distance
from datetime import datetime, timedelta

#=====================================
# Bucket S3
s3_bucket_name = os.getenv("S3_BUCKET_NAME")
s3_region = os.getenv("S3_REGION")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
#=======================================
print(s3_bucket_name)
print(s3_region)
print(aws_access_key_id)
print(aws_secret_access_key)

# ==========================
# Load CSV from a specific directory
# ==========================
@st.cache_data
def load_data(directory="csv_files"):
    csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    if not csv_files:
        st.error(f"No CSV files found in {directory}")
        return pd.DataFrame()
    
    csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))
    file_path = os.path.join(directory, csv_files[0])
    
    df = pd.read_csv(file_path)

    # Ensure forecast_date column is datetime
    if 'forecast_date' in df.columns:
        df['forecast_date'] = pd.to_datetime(df['forecast_date'])
    else:
        st.warning("forecast_date column missing in CSV. Using today's date instead.")
        df['forecast_date'] = pd.Timestamp.today().normalize()
    
    return df

df = load_data() ## change here

# ==========================
# Geocode location
# ==========================
def geocode_location(locality, gewog_thromde, dzongkhag):
    url = "https://nominatim.openstreetmap.org/search"
    query = f"{locality}, {gewog_thromde}, {dzongkhag}, Bhutan"
    params = {"q": query, "format": "json"}
    try:
        response = requests.get(url, params=params, headers={'User-Agent': 'forecast-app'})
        if response.status_code == 200 and response.json():
            data = response.json()[0]
            return float(data['lat']), float(data['lon'])
    except Exception as e:
        st.error(f"Geocoding error: {e}")
    return None, None

# ==========================
# Bilinear interpolation helpers
# ==========================
def find_surrounding_points(df, lat, lon, param, time_col):
    df_param = df[df['param'] == param]
    latitudes = np.sort(df_param['latitude'].unique())
    longitudes = np.sort(df_param['longitude'].unique())

    lat_below = latitudes[latitudes <= lat].max() if np.any(latitudes <= lat) else None
    lat_above = latitudes[latitudes >= lat].min() if np.any(latitudes >= lat) else None
    lon_left = longitudes[longitudes <= lon].max() if np.any(longitudes <= lon) else None
    lon_right = longitudes[longitudes >= lon].min() if np.any(longitudes >= lon) else None

    if lat_below is None or lat_above is None or lon_left is None or lon_right is None:
        return None

    Q11 = df_param[(df_param['latitude'] == lat_below) & (df_param['longitude'] == lon_left)][time_col].values
    Q21 = df_param[(df_param['latitude'] == lat_below) & (df_param['longitude'] == lon_right)][time_col].values
    Q12 = df_param[(df_param['latitude'] == lat_above) & (df_param['longitude'] == lon_left)][time_col].values
    Q22 = df_param[(df_param['latitude'] == lat_above) & (df_param['longitude'] == lon_right)][time_col].values

    if len(Q11) == 0 or len(Q21) == 0 or len(Q12) == 0 or len(Q22) == 0:
        return None

    return {
        'lat_below': lat_below,
        'lat_above': lat_above,
        'lon_left': lon_left,
        'lon_right': lon_right,
        'Q11': Q11[0],
        'Q21': Q21[0],
        'Q12': Q12[0],
        'Q22': Q22[0]
    }

def bilinear_interpolation(data, lat, lon):
    x1, x2 = data['lon_left'], data['lon_right']
    y1, y2 = data['lat_below'], data['lat_above']
    x, y = lon, lat

    Q11 = data['Q11']
    Q21 = data['Q21']
    Q12 = data['Q12']
    Q22 = data['Q22']

    denom = (x2 - x1) * (y2 - y1)
    if denom < 1e-4:
        return np.mean([Q11, Q21, Q12, Q22])

    term1 = Q11 * (x2 - x) * (y2 - y)
    term2 = Q21 * (x - x1) * (y2 - y)
    term3 = Q12 * (x2 - x) * (y - y1)
    term4 = Q22 * (x - x1) * (y - y1)

    return (term1 + term2 + term3 + term4) / denom

# ==========================
# Total precipitation helpers
# ==========================
def radius_in_degrees(lat_center, radius_km=10):
    lat_deg = radius_km / 111
    lon_deg = radius_km / (111 * np.cos(np.radians(lat_center)))
    return lat_deg, lon_deg

def points_within_radius(df, lat_center, lon_center, param, time_col, radius_km=10):
    lat_radius, lon_radius = radius_in_degrees(lat_center, radius_km)
    df_param = df[df['param'] == param]

    df_box = df_param[
        (df_param['latitude'] >= lat_center - lat_radius) &
        (df_param['latitude'] <= lat_center + lat_radius) &
        (df_param['longitude'] >= lon_center - lon_radius) &
        (df_param['longitude'] <= lon_center + lon_radius)
    ]

    selected_points = []
    for _, row in df_box.iterrows():
        dist = distance((lat_center, lon_center), (row['latitude'], row['longitude'])).km
        if dist <= radius_km:
            selected_points.append(row[time_col])
    
    return selected_points

def total_precipitation(selected_points):
    if not selected_points:
        return None
    return sum(selected_points)

# ==========================
# Initialize session_state
# ==========================
if 'forecast_clicked' not in st.session_state:
    st.session_state.forecast_clicked = False
if 'lat' not in st.session_state:
    st.session_state.lat = None
if 'lon' not in st.session_state:
    st.session_state.lon = None
if 'selected_param' not in st.session_state:
    st.session_state.selected_param = None

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(layout="wide")

st.markdown("""<style>
[data-testid="stAppViewContainer"] { padding-top:0 !important; }
.block-container { padding-top:0 !important; margin-top:0 !important; }
h1,h2,h3,h4,h5,h6 { margin-top:0 !important; padding-top:0 !important; }
[data-testid="stAppViewContainer"] { background-color: #f9f8f4; }
[data-testid="stSidebar"] { background-color: #0077cc; color: white; }
[data-testid="stSidebar"] * { color: white; }
div.stButton > button {
    background: linear-gradient(to bottom, #ffffff 0%, #e6e6e6 100%);
    border: 2px solid #999999; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    color: #000000; padding: 12px 28px; font-size: 20px; font-weight: bold; cursor: pointer; transition: all 0.2s ease-in-out;
}
div.stButton > button:hover {
    background: linear-gradient(to bottom, #f0f0f0 0%, #cccccc 100%);
    box-shadow: 0 8px 20px rgba(0,0,0,0.4);
}
div.stButton > button:active {
    box-shadow: inset 0 3px 8px rgba(0,0,0,0.5);
    background: linear-gradient(to bottom, #cccccc 0%, #f0f0f0 100%);
}
</style>""", unsafe_allow_html=True)

# --------------------
# Header
# --------------------
st.markdown("""<div style="background-color:f9f8f4; padding: 10px 5px 5px 5px; border-bottom: 2px solid #f9f8f4;">
    <h1 style="text-align: left; font-size: 60px; color: black; font-style: Calibri;">
        འབྲུག་ཆུ་རུད་ཀྱི་རྐྱེན་ངན་ཉེན་བརྡའི་དྲ་ངོས།<br>
        <span style="font-size: 30px;">Bhutan Weather Portal</span>
    </h1>
</div>""", unsafe_allow_html=True)

st.sidebar.title(" ")

tab_weather_forecast, = st.tabs(["Weather Forecast"])

with tab_weather_forecast:
    col1, col2, col3 = st.columns(3)
    with col1:
        locality = st.text_input("Locality", value="Changzamtog")
    with col2:
        gewog_thromde = st.text_input("Gewog or Thromde", value="Thimphu Thromde")
    with col3:
        dzongkhag = st.text_input("Dzongkhag", value="Thimphu")

    if st.button("Get Forecast", key="forecast_button"):
        lat, lon = geocode_location(locality, gewog_thromde, dzongkhag)
        if lat is None or lon is None:
            st.error("Location not found.")
            st.session_state.forecast_clicked = False
        else:
            st.session_state.lat = lat
            st.session_state.lon = lon
            st.session_state.forecast_clicked = True

    if st.session_state.forecast_clicked:
        expected_params = ["temperature_celcius", "precipitation", "surface_area"]
        params = [p for p in expected_params if p in df['param'].unique()]
        time_cols = [c for c in df.columns if 'h' in c]

        if st.session_state.selected_param is None:
            st.session_state.selected_param = "temperature_celcius" if "temperature_celcius" in params else params[0]

        col_map, col_chart = st.columns([1, 1])

        with col_map:
            st.markdown(f"""
            <div style="
                background-color: #005fa3;
                color: white;
                padding: 7px 0px;
                border-radius: 5px 5px 0 0;
                font-size: 18px;
                text-align: center;
                width: 100%;
                margin: 0;">
                📍 Location: Latitude {st.session_state.lat:.4f}, Longitude {st.session_state.lon:.4f}
            </div>""", unsafe_allow_html=True)

            m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=10)
            folium.Circle(
                location=[st.session_state.lat, st.session_state.lon],
                radius=10000,
                color="blue",
                weight=1,
                fill=True,
                fill_color="blue",
                fill_opacity=0.2,
                popup="10 km radius"
            ).add_to(m)
            html(m._repr_html_(), height=500)

            # Total precipitation box
            if "precipitation" in df['param'].unique():
                total_precip_list = []
                for time_col in time_cols:
                    nearby_points = points_within_radius(df, st.session_state.lat, st.session_state.lon,
                                                         "precipitation", time_col, radius_km=10)
                    total_precip_list.append(total_precipitation(nearby_points))
                st.markdown(f"""
                <div style="
                    background-color:#e0f7fa;
                    color:#0077cc;
                    padding:10px;
                    border-radius:5px;
                    font-size:18px;
                    text-align:center;
                    margin-top:5px;
                    margin-bottom:5px;">
                    🌧 Total Precipitation within 10 km radius (per forecast time): <b>{', '.join([str(round(v,2))+' mm' if v is not None else 'N/A' for v in total_precip_list])}</b>
                </div>
                """, unsafe_allow_html=True)

        # ---------------------
        # Line chart with proper dates
        # ---------------------
        with col_chart:
            st.markdown(f"""
            <div style="
                background-color: #005fa3;
                color: white;
                padding: 7px 0px;
                border-radius: 5px 5px 0 0;
                font-size: 18px;
                text-align: center;
                width: 100%;
                margin: 1;
            ">
                  4-day Weather forecast for {locality}, {gewog_thromde}, {dzongkhag}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <style>
            div[role="radiogroup"] { margin-top: 0px !important; margin-bottom: 0px !important; padding-top: 0px !important; padding-bottom: 0px !important; }
            div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] { color: black !important; font-weight: bold; }
            </style>
            """, unsafe_allow_html=True)

            param_labels = {
                "temperature_celcius": "Temperature (°C)",
                "precipitation": "Precipitation (mm)",
                "surface_area": "Surface Runoff"
            }
            selected_param = st.radio(
                label="",
                options=params,
                format_func=lambda x: param_labels[x],
                horizontal=True
            )
            st.session_state.selected_param = selected_param

            results = []
            for time in time_cols:
                surrounding = find_surrounding_points(df, st.session_state.lat, st.session_state.lon,
                                                      st.session_state.selected_param, time)
                if surrounding:
                    value = bilinear_interpolation(surrounding, st.session_state.lat, st.session_state.lon)
                    if st.session_state.selected_param == "surface_area" and (value is None or value < 0.01):
                        value = 0
                    results.append((time, round(value, 2)))
                else:
                    results.append((time, "Insufficient data"))

            # Add proper datetime for x-axis
            forecast_date = df['forecast_date'].iloc[0]
            times_for_plot = []
            for t, val in results:
                hour = int(t.replace('h',''))
                dt = forecast_date + timedelta(hours=hour)
                times_for_plot.append(dt)

            result_df = pd.DataFrame({
                'Forecast Time': times_for_plot,
                'Interpolated Value': [val for _, val in results]
            })
            plot_df = result_df[result_df['Interpolated Value'].apply(lambda x: isinstance(x, (int, float)))]

            if not plot_df.empty:
                fig = px.line(
                    plot_df,
                    x='Forecast Time',
                    y='Interpolated Value',
                    markers=True,
                    labels={'y': 'Interpolated Value', 'Forecast Time': 'Date & Time'}
                )
                fig.update_traces(
                    text=plot_df['Interpolated Value'],
                    textposition='top center',
                    mode='lines+markers+text',
                    line=dict(color='black', width=2),
                    marker=dict(color='black', size=8),
                    textfont=dict(color='black')
                )
                fig.update_layout(
                    height=335,
                    margin=dict(l=40, r=20, t=30, b=40),
                    paper_bgcolor='lightgrey',
                    plot_bgcolor='lightgrey',
                    xaxis=dict(
                        tickfont=dict(color='black'), 
                        title_font=dict(color='black'),
                        showgrid=False,
                        tickformat="%d %b %Hh"
                    ),
                    yaxis=dict(
                        tickfont=dict(color='black'), 
                        title_font=dict(color='black'),
                        showticklabels=False,
                        gridcolor='rgba(0,0,0,0.2)',
                        griddash='dot',
                        gridwidth=1
                    )
                )
                st.plotly_chart(fig, use_container_width=True)


        # ==========================
        # Nearby places using Overpass API (4-hour forecast table)
        # ==========================
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<h3 style="color:black;">Geographic location around the selected location - 10 Km radius</h3>', unsafe_allow_html=True)

        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json][timeout:25];
        (
          node["place"="city"](around:10000,{st.session_state.lat},{st.session_state.lon});
          node["place"="town"](around:10000,{st.session_state.lat},{st.session_state.lon});
          node["place"="village"](around:10000,{st.session_state.lat},{st.session_state.lon});
          node["place"="hamlet"](around:10000,{st.session_state.lat},{st.session_state.lon});
        );
        out body;
        >;
        out skel qt;
        """
        try:
            response = requests.get(overpass_url, params={'data': overpass_query})
            data = response.json()
            unique_places = []
            for element in data.get('elements', []):
                if 'tags' in element and 'name' in element['tags']:
                    unique_places.append({'name': element['tags']['name'], 'lat': element['lat'], 'lon': element['lon']})
            unique_places = unique_places[:10]
        except Exception as e:
            st.error(f"Overpass API error: {e}")
            unique_places = []

        if unique_places:
            parameters = ["temperature_celcius", "precipitation", "surface_area"]
            param_labels = {
                "temperature_celcius": "Temperature (°C)",
                "precipitation": "Precipitation (mm)",
                "surface_area": "Surface Runoff"
            }

            table_data = {place['name']: [] for place in unique_places}
            row_index = []

            for param in parameters:
                for time in time_cols:  # all 4 hours
                    row_index.append(f"{param_labels[param]} – {time}")
                    for place in unique_places:
                        lat_pt, lon_pt = place['lat'], place['lon']
                        interpol_data = find_surrounding_points(df, lat_pt, lon_pt, param, time)
                        if interpol_data:
                            value = bilinear_interpolation(interpol_data, lat_pt, lon_pt)
                            if param == "surface_area" and (value is None or value < 0.01):
                                value = 0
                            table_data[place['name']].append(round(value, 2) if value is not None else None)
                        else:
                            table_data[place['name']].append(None)

            df_places = pd.DataFrame(table_data, index=row_index)
            st.dataframe(df_places)
        else:
            st.info("No geographical places found within 10 km.")

