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
import io
import base64

#=====================================
# Bucket S3
#s3_bucket_name = os.getenv("S3_BUCKET_NAME")
#s3_region = os.getenv("S3_REGION")
#aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
#aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
#=======================================
#st.write(f"s3_bucket_name: {s3_bucket_name}")
#st.write(f"s3_region: {s3_region}")
#st.write(f"aws_access_key_id: {aws_access_key_id}")
#st.write(f"aws_secret_access_key: {aws_secret_access_key}")


#print(s3_bucket_name)
#print(s3_region)
#print(aws_access_key_id)
#print(aws_secret_access_key)

# ==========================
# Load CSVs from a specific directory (_1 and _2) and merge forecast columns
# ==========================
@st.cache_data
def load_data(directory="csv_files"):
    import re

    # Get all CSVs in the folder
    csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    if not csv_files:
        st.error(f"No CSV files found in {directory}")
        return pd.DataFrame()

    # Sort CSVs by numeric suffix (e.g., _1, _2, _3)
    def get_suffix_num(filename):
        match = re.search(r'_(\d+)\.csv$', filename)
        return int(match.group(1)) if match else 0

    csv_files.sort(key=get_suffix_num)

    # Load and concatenate all CSVs in order
    df_final = None
    for i, file in enumerate(csv_files):
        df = pd.read_csv(os.path.join(directory, file))

        # Ensure forecast_date is datetime
        if 'forecast_date' in df.columns:
            df['forecast_date'] = pd.to_datetime(df['forecast_date'], errors='coerce')
        else:
            df['forecast_date'] = pd.Timestamp.today().normalize()

        if df_final is None:
            df_final = df
        else:
            # Exclude common columns from the right df to avoid duplication
            common_cols = ['longitude', 'latitude', 'forecast_date', 'param', 'param_tag']
            new_cols = [c for c in df.columns if c not in common_cols]
            df_final = pd.concat([df_final.reset_index(drop=True), df[new_cols].reset_index(drop=True)], axis=1)

    return df_final


df = load_data()  # load combined data

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
st.markdown("""
<style>
/* Target the input label container and make it bold, black, larger */
div.stTextInput label > div[data-testid="stMarkdownContainer"],
div.stTextInput label {
    color: black !important;
    font-weight: 700 !important; /* Strong bold */
    font-size: 18px !important;
}

/* Remove margin and padding to reduce space between label and input */
div.stTextInput {
    margin-bottom: 0px !important;
    padding-bottom: 0px !important;
}

div.stTextInput > div {
    margin-bottom: 4px !important;
    padding-bottom: 0px !important;
}
            
/* Change the background color, border, and text color of input boxes */
div.stTextInput input {
    background-color: #ebebf0 !important;  /* light blue background */
    border: 2px solid #210307 !important;  /* blue border */
    color: #000000 !important;             /* text color black */
    border-radius: 8px !important;
    padding: 6px 10px !important;
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""<style>
/* Ensure background and container styling */
[data-testid="stAppViewContainer"] { padding-top:0 !important; }
.block-container { padding-top:0 !important; margin-top:0 !important; }
h1,h2,h3,h4,h5,h6 { margin-top:0 !important; padding-top:0 !important; }
[data-testid="stAppViewContainer"] { background-color: #f9f8f4; }
[data-testid="stSidebar"] { background-color: #0077cc; color: white; }
[data-testid="stSidebar"] * { color: white; }

/* Button styling */
div.stButton > button {
    background: linear-gradient(to bottom, #ffffff 0%, #e6e6e6 100%);
    border: 2px solid #006400; 
    border-radius: 10px; 
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    color: #006400;
    padding: 12px 28px; 
    font-size: 24px; 
    font-weight: bold; 
    cursor: pointer; 
    transition: all 0.3s ease-in-out;
}
div.stButton > button:hover {
    background: linear-gradient(to bottom, #006400 0%, #006400 100%);
    color: white;
    box-shadow: 0 8px 20px rgba(0,0,0,0.4);
}
div.stButton > button:active {
    box-shadow: inset 0 3px 8px rgba(0,0,0,0.5);
    background: linear-gradient(to bottom, #006400 0%, #006400 100%);
    color: white;
}

</style>""", unsafe_allow_html=True)



# --- Header ---
st.markdown("""<div style="background-color:f9f8f4; padding: 10px 5px 5px 5px; border-bottom: 2px solid #f9f8f4;">
    <h1 style="text-align: left; font-size: 60px; color: black; font-style: Calibri;">
        འབྲུག་ཆུ་རུད་ཀྱི་རྐྱེན་ངན་ཉེན་བརྡའི་དྲ་ངོས།<br>
        <span style="font-size: 30px;">Bhutan Weather Portal</span>
    </h1>
</div>""", unsafe_allow_html=True)

st.sidebar.title(" ")

tab_weather_forecast, = st.tabs(["Weather Forecast"])

# ==========================
# Live Rainfall Alert Banner - only for predefined cities/villages in Bhutan
# ==========================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<h3 style="color:black;">Live Rainfall Alert</h3>', unsafe_allow_html=True)

# Predefined list of top cities and villages
BHUTAN_LOCATIONS = [
    # Top Cities
    {"name": "Thimphu", "lat": 27.4728, "lon": 89.6393},
    {"name": "Phuntsholing", "lat": 26.8574, "lon": 89.3886},
    {"name": "Paro", "lat": 27.4305, "lon": 89.4134},
    {"name": "Gelephu", "lat": 26.8725, "lon": 90.4927},
    {"name": "Samdrup Jongkhar", "lat": 26.8000, "lon": 91.5000},
    {"name": "Wangdue Phodrang", "lat": 27.4167, "lon": 89.9000},
    {"name": "Punakha", "lat": 27.5833, "lon": 89.8667},
    {"name": "Jakar", "lat": 27.5492, "lon": 90.7525},
    {"name": "Nganglam", "lat": 26.7833, "lon": 91.2500},
    {"name": "Samtse", "lat": 26.8990, "lon": 89.0995},

    # Top Villages
    {"name": "Sakteng", "lat": 27.3833, "lon": 91.8667},
    {"name": "Merak", "lat": 27.2493, "lon": 91.9085},
    {"name": "Gangtey", "lat": 27.5000, "lon": 90.1667},
    {"name": "Khoma", "lat": 27.8245, "lon": 91.3281},
    {"name": "Talo", "lat": 27.5223, "lon": 89.9408},
    {"name": "Wochu", "lat": 27.4410, "lon": 89.3920},
    {"name": "Rinchengang", "lat": 27.4667, "lon": 89.3833},
    {"name": "Ura", "lat": 27.4167, "lon": 90.9167},
    {"name": "Rukubji", "lat": 27.5333, "lon": 89.9667},
    {"name": "Khamaed", "lat": 27.4667, "lon": 89.8833}
]

# Find places with heavy rainfall based on current forecast data (using bilinear interpolation)
if not df.empty and "precipitation" in df['param'].unique():
    time_cols = [c for c in df.columns if 'h' in c]
    heavy_rain_places = []

    for place in BHUTAN_LOCATIONS:
        total_precip = 0
        for time_col in time_cols:
            interpol_data = find_surrounding_points(df, place['lat'], place['lon'], "precipitation", time_col)
            if interpol_data:
                value = bilinear_interpolation(interpol_data, place['lat'], place['lon'])
                if value is not None:
                    total_precip += value

        if total_precip > 0.20:  # Threshold for at least moderate rainfall
            # Determine alert level and color
            if total_precip >= 0.50:
                alert_level = "Very High Rainfall ⚠️"
                color = "#ff4c4c"  # Red
            elif total_precip >= 0.30:
                alert_level = "High Rainfall ⚠️"
                color = "#ff9800"  # Orange
            else:
                alert_level = "Moderate Rainfall ⚡"
                color = "#ffcc00"  # Yellow

            heavy_rain_places.append({
                'name': place['name'],
                'precip': round(total_precip, 2),
                'alert_level': alert_level,
                'color': color
            })

    # Sort by precipitation and limit to top 30
    heavy_rain_places = sorted(heavy_rain_places, key=lambda x: x['precip'], reverse=True)[:30]

    if heavy_rain_places:
        # Build scrolling text with color-coded alerts
        places_text = "   ".join([
            f"<span style='color:{place['color']}; font-weight:bold;'>{place['name']} ({place['precip']} mm) - {place['alert_level']}</span>"
            for place in heavy_rain_places
        ])
        st.markdown(f"""
        <style>
        .scroll-container {{
            overflow: hidden;
            white-space: nowrap;
            box-sizing: border-box;
            border-radius: 8px;
            padding: 10px 0;
            background-color:#111112;
        }}
        .scroll-text {{
            display: inline-block;
            padding-left: 100%;
            animation: scroll-left 40s linear infinite;
            font-size: 18px;
        }}
        @keyframes scroll-left {{
            0% {{ transform: translateX(0); }}
            100% {{ transform: translateX(-100%); }}
        }}
        </style>
        <div class="scroll-container">
            <div class="scroll-text">{places_text}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            background-color:#d4edda;
            color:#155724;
            padding:15px;
            border-radius:8px;
            font-size:18px;
            text-align:center;">
            ✅ No places with significant rainfall at the moment.
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style="
        background-color:#f8d7da;
        color:#721c24;
        padding:15px;
        border-radius:8px;
        font-size:18px;
        text-align:center;">
        ⚠️ Precipitation data not available.
    </div>
    """, unsafe_allow_html=True)

with tab_weather_forecast:
    col1, col2, col3 = st.columns(3)
    with col1:
        locality = st.text_input("Specify Locality", value="Changzamtog")
    with col2:
        gewog_thromde = st.text_input("Specify Gewog or Thromde", value="Thimphu Thromde")
    with col3:
        dzongkhag = st.text_input("Specify Dzongkhag", value="Thimphu")

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
        # --- High Rainfall Alert ---
        high_rainfall = False
        if "precipitation" in df['param'].unique():
            for time_col in time_cols:
                nearby_points = points_within_radius(df, st.session_state.lat, st.session_state.lon,
                                                     "precipitation", time_col, radius_km=10)
                total_precip = total_precipitation(nearby_points)
                if total_precip is not None and total_precip > 0.01:
                    high_rainfall = True
                    break

        if high_rainfall:
            st.markdown(f"""
            <div style="
                background-color:#ff4c4c;
                color:white;
                padding:15px;
                border-radius:8px;
                font-size:20px;
                text-align:center;
                margin-bottom:10px;">
                ⚠️ <b>High Rainfall Alert!</b> Precipitation exceeds 10mm within a 10 km radius.
            </div>
            """, unsafe_allow_html=True)


        if st.session_state.selected_param is None:
            st.session_state.selected_param = "temperature_celcius" if "temperature_celcius" in params else params[0]

        col_map, col_chart = st.columns([1, 1])

        # --- Map and precipitation ---
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
                📍 Selected Location: Latitude {st.session_state.lat:.4f}, Longitude {st.session_state.lon:.4f}
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

            # Add a dot/marker at the epicenter
            # Add a sleek red pin marker at the epicenter
            folium.Marker(
                    location=[st.session_state.lat, st.session_state.lon],
                    icon=folium.Icon(color="red", icon="glyphicon-map-marker"),  # modern pin
                    popup="Selected Location"
            ).add_to(m)
            html(m._repr_html_(), height=500)
        # --- Line chart ---
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
                  4-Day Weather Forecast for {locality}, {gewog_thromde}, {dzongkhag}
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
                        tickformat="%I%p %d %b"),
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
# Nearby places using Overpass API (grouped forecast times by actual date)
# ==========================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<h3 style="color:black;">Nearby locations within 10 km of the selected point</h3>', unsafe_allow_html=True)

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
        "temperature_celcius": "Temperature",
        "precipitation": "Precipitation",
        "surface_area": "Surface Runoff"
    }

    # Group forecast times by "day" (every 3 time columns)
    time_groups = [time_cols[i:i+3] for i in range(0, len(time_cols), 3)]
    
    # Generate actual sequential dates for each group
    forecast_start = df['forecast_date'].iloc[0] if not df.empty else pd.Timestamp.today()
    date_labels = [(forecast_start + timedelta(days=i)).strftime("%d %b %Y") for i in range(len(time_groups))]

    # Dropdown to select forecast date
    selected_date_idx = st.selectbox("Select forecast date", options=range(len(date_labels)), format_func=lambda x: date_labels[x])
    selected_times = time_groups[selected_date_idx]

    all_rows = []
    for place in unique_places:
        row = {"Location": place['name']}
        for param in parameters:
            for time in selected_times:
                lat_pt, lon_pt = place['lat'], place['lon']
                interpol_data = find_surrounding_points(df, lat_pt, lon_pt, param, time)
                if interpol_data:
                    value = bilinear_interpolation(interpol_data, lat_pt, lon_pt)
                    if param == "surface_area" and (value is None or value < 0.01):
                        value = 0
                    value = round(value, 2)
                else:
                    value = None
                row[f"{param_labels[param]} ({time})"] = value
        all_rows.append(row)

    df_places = pd.DataFrame(all_rows)
    df_places.set_index("Location", inplace=True)
    st.dataframe(df_places)
else:
    st.info("No geographical places found within 10 km.")

TOP_CITIES = [
    {"name": "Thimphu", "lat": 27.4728, "lon": 89.6393},
    {"name": "Phuntsholing", "lat": 26.8574, "lon": 89.3886},
    {"name": "Paro", "lat": 27.4305, "lon": 89.4134},
    {"name": "Gelephu", "lat": 26.8725, "lon": 90.4927},
    {"name": "Samdrup Jongkhar", "lat": 26.8000, "lon": 91.5000},
    {"name": "Wangdue Phodrang", "lat": 27.4167, "lon": 89.9000},
    {"name": "Punakha", "lat": 27.5833, "lon": 89.8667},
    {"name": "Jakar", "lat": 27.5492, "lon": 90.7525},
    {"name": "Nganglam", "lat": 26.7833, "lon": 91.2500},
    {"name": "Samtse", "lat": 26.8990, "lon": 89.0995}
]

# Sidebar city selection and weather display
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown('<h3 style="color:white;">📍 Select City to View Weather</h3>', unsafe_allow_html=True)

if not df.empty:
    city_names = [city['name'] for city in TOP_CITIES]
    selected_city = st.sidebar.selectbox("Choose a city", options=city_names)

    # Find the city details
    city = next((c for c in TOP_CITIES if c['name'] == selected_city), None)

    if city:
        parameters = ["temperature_celcius", "precipitation", "surface_area"]
        param_labels = {
            "temperature_celcius": "Temperature (°C)",
            "precipitation": "Precipitation (mm)",
            "surface_area": "Surface Runoff"
        }
        time_cols = [c for c in df.columns if 'h' in c]
        
        # Generate datetime labels for each time column
        forecast_start = df['forecast_date'].iloc[0] if not df.empty else pd.Timestamp.today()
        time_labels = []
        for time in time_cols:
            hour = int(time.replace('h',''))
            dt = forecast_start + timedelta(hours=hour)
            label = dt.strftime("%d %b %Y %I%p")
            time_labels.append((time, label))

        st.sidebar.markdown(f"<h4 style='color:white;'>Weather in {city['name']}</h4>", unsafe_allow_html=True)

        for param in parameters:
            with st.sidebar.expander(f"{param_labels[param]}", expanded=False):
                for time, label in time_labels:
                    interpol_data = find_surrounding_points(df, city['lat'], city['lon'], param, time)
                    if interpol_data:
                        value = bilinear_interpolation(interpol_data, city['lat'], city['lon'])
                        if param == "surface_area" and (value is None or value < 0.01):
                            value = 0
                        value = round(value, 2)
                        display_value = f"{value}"
                    else:
                        display_value = "Data not available"

                    st.markdown(f"**{label}**: {display_value}")

        st.sidebar.markdown(f"""
            <div style="
                background-color:#004d99;
                color:white;
                padding:10px;
                border-radius:8px;
                text-align:center;
                margin-top:10px;">
                📍 Location: {city['name']}<br>
                Latitude: {city['lat']}<br>
                Longitude: {city['lon']}
            </div>
        """, unsafe_allow_html=True)

        # ---------------- Bhutan Flag at bottom ----------------
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("""
    <div style="text-align:center; padding-bottom:20px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Flag_of_Bhutan.svg" 
             alt="Bhutan Flag" 
             style="width:80%; border-radius:0px;"/>
        <p style="color:white; font-weight:bold; margin-top:5px;">Omdena Bhutan</p>
    </div>
""", unsafe_allow_html=True)



