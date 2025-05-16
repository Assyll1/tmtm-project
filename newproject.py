import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("📦 AI + IoT + ETA + Геокарта: Жүк қауіпін болжау және ТМТМ маршруты")

# --- 1. Генерация IoT-данных ---
np.random.seed(0)
df = pd.DataFrame({
    "температура_C": np.random.uniform(15, 35, 100),
    "ылғалдылық_пайыз": np.random.uniform(30, 90, 100),
    "вибрация_GPS": np.random.uniform(0, 2, 100),
    "риск_повреждения": np.random.choice([0, 1], 100)
})
df['номер_контейнера'] = [f"Контейнер {i+1}" for i in range(len(df))]

# --- 2. Обучение модели риска ---
X = df.drop(columns=["риск_повреждения", "номер_контейнера"])
y = df["риск_повреждения"]
risk_model = RandomForestClassifier()
risk_model.fit(X, y)
df["болжау_қауіп"] = risk_model.predict(X)

# --- 3. Геолокации контейнеров ---
coords_tmtn = [
    [36.0611, 103.8343], [44.2000, 80.4000], [43.2389, 76.8897],
    [43.6520, 51.1970], [40.4093, 49.8671], [41.7151, 44.8271],
    [40.6030, 43.0970], [36.8000, 34.6333]
]
lats, lons = zip(*coords_tmtn)
df["lat"] = np.random.choice(lats, size=len(df))
df["lon"] = np.random.choice(lons, size=len(df))

# --- 4. Модель ETA ---
eta_data = pd.DataFrame({
    "қашықтық_км": np.random.uniform(100, 5000, 200),
    "жылдамдық_км_сағ": np.random.uniform(20, 80, 200),
    "кеден_кешігуі_сағат": np.random.uniform(0, 48, 200),
    "порт_кешігуі": np.random.uniform(0, 24, 200)
})
eta_data["ETA_сағат"] = (
    eta_data["қашықтық_км"] / eta_data["жылдамдық_км_сағ"] +
    eta_data["кеден_кешігуі_сағат"] + eta_data["порт_кешігуі"]
)
eta_model = LinearRegression()
eta_model.fit(eta_data.drop(columns=["ETA_сағат"]), eta_data["ETA_сағат"])

# --- 5. ETA болжау формасы ---
st.subheader("⏱️ ETA болжау")

col1, col2, col3, col4 = st.columns(4)
қашықтық = col1.number_input("Қашықтық (км)", 100, 10000, 1200)
жылдамдық = col2.number_input("Жылдамдық (км/сағ)", 10, 100, 60)
кеден = col3.number_input("Кеден кешігуі (сағ)", 0, 72, 5)
порт = col4.number_input("Порт кешігуі (сағ)", 0, 48, 3)

if "eta_result" not in st.session_state:
    st.session_state.eta_result = None

if st.button("ETA есептеу"):
    eta_input = pd.DataFrame({
        "қашықтық_км": [қашықтық],
        "жылдамдық_км_сағ": [жылдамдық],
        "кеден_кешігуі_сағат": [кеден],
        "порт_кешігуі": [порт]
    })
    eta_result = eta_model.predict(eta_input)[0]
    st.session_state.eta_result = eta_result

if st.session_state.eta_result is not None:
    st.success(f"📦 Контейнердің жету уақыты: {st.session_state.eta_result:.1f} сағат")

# --- 6. Выбор маршрутов ---
st.subheader("🧭 Таңдалған маршрут учаскелері")

# Сегменты через Қорғас
tmtn_segments_qorgas = [
    {"from": "Ланьчжоу", "to": "Қорғас", "coords": [[36.0611, 103.8343], [44.2000, 80.4000]]},
    {"from": "Қорғас", "to": "Алматы", "coords": [[44.2000, 80.4000], [43.2389, 76.8897]]},
    {"from": "Алматы", "to": "Ақтау", "coords": [[43.2389, 76.8897], [43.6520, 51.1970]]},
    {"from": "Ақтау", "to": "Баку", "coords": [[43.6520, 51.1970], [40.4093, 49.8671]]},
    {"from": "Баку", "to": "Тбилиси", "coords": [[40.4093, 49.8671], [41.7151, 44.8271]]},
    {"from": "Тбилиси", "to": "Карс", "coords": [[41.7151, 44.8271], [40.6030, 43.0970]]},
    {"from": "Карс", "to": "Мерсин", "coords": [[40.6030, 43.0970], [36.8000, 34.6333]]}
]

# Сегменты через Үрімші
tmtn_segments_urumqi = [
    {"from": "Ланьчжоу", "to": "Үрімші", "coords": [[36.0611, 103.8343], [43.8, 87.6]]},
    {"from": "Үрімші", "to": "Алматы", "coords": [[43.8, 87.6], [43.2389, 76.8897]]},
    {"from": "Алматы", "to": "Ақтау", "coords": [[43.2389, 76.8897], [43.6520, 51.1970]]},
    {"from": "Ақтау", "to": "Баку", "coords": [[43.6520, 51.1970], [40.4093, 49.8671]]},
    {"from": "Баку", "to": "Тбилиси", "coords": [[40.4093, 49.8671], [41.7151, 44.8271]]},
    {"from": "Тбилиси", "to": "Карс", "coords": [[41.7151, 44.8271], [40.6030, 43.0970]]},
    {"from": "Карс", "to": "Мерсин", "coords": [[40.6030, 43.0970], [36.8000, 34.6333]]}
]

# Переменная для хранения выбранных сегментов
selected_segments = []

# Выбор маршрута через Қорғас
if st.checkbox("Маршрут через Қорғас", value=True, key="route_qorgas"):
    selected_segments.extend(tmtn_segments_qorgas)

# Выбор маршрута через Үрімші
if st.checkbox("Маршрут через Үрімші", value=True, key="route_urumqi"):
    selected_segments.extend(tmtn_segments_urumqi)

# --- 7. ETA для выбранных участков ---
segments = []
for seg in selected_segments:
    dist = np.random.uniform(400, 1000)
    speed = np.random.uniform(35, 65)
    delay1 = np.random.uniform(1, 10)
    delay2 = np.random.uniform(1, 5)
    eta = dist / speed + delay1 + delay2
    segments.append({
        "Участок": f"{seg['from']} → {seg['to']}",
        "Қашықтық (км)": round(dist, 1),
        "Жылдамдық (км/сағ)": round(speed, 1),
        "Кеден кешігуі (сағ)": round(delay1, 1),
        "Порт кешігуі (сағ)": round(delay2, 1),
        "ETA (сағат)": round(eta, 1),
        "coords": seg["coords"]
    })

eta_total = sum(s["ETA (сағат)"] for s in segments)
df_segments = pd.DataFrame(segments).drop(columns=["coords"])
st.success(f"🔄 Жалпы ETA таңдалған маршрутқа: {eta_total:.1f} сағат")
st.dataframe(df_segments)
# --- 8. ETA графиктері (обновлённый) ---
st.subheader("📈 ETA талдауы")

df_eta = pd.DataFrame(segments)

fig1, ax1 = plt.subplots(figsize=(8, 4))
sns.histplot(df_eta["ETA (сағат)"], bins=10, kde=True, ax=ax1, color='skyblue')
ax1.set_title("ETA таралуы (таңдалған учаскелер бойынша)")
ax1.set_xlabel("ETA (сағат)")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.scatterplot(
    data=df_eta,
    x="Қашықтық (км)",
    y="ETA (сағат)",
    hue="Жылдамдық (км/сағ)",
    palette="viridis",
    ax=ax2
)
ax2.set_title("ETA vs Қашықтық (таңдалған учаскелер бойынша)")
st.pyplot(fig2)

# --- 9. Карта маршрута ---
st.subheader("🗺️ Маршрут картасы (таңдалған учаскелер)")

map_route = folium.Map(location=[43.5, 70.0], zoom_start=4)

# Добавляем маршруты на карту
for seg in segments:
    folium.PolyLine(
        locations=seg["coords"],
        tooltip=f"{seg['Участок']}, ETA: {seg['ETA (сағат)']} сағ",
        color="blue", weight=4, opacity=0.7
    ).add_to(map_route)

st_folium(map_route, width=1000, height=600)
# --- 10. Контейнер фильтрациясы и карта ---
st.subheader("📅 Контейнер фильтрациясы")
df["дата_жеткізу"] = pd.date_range(start="2025-01-01", periods=len(df), freq="D")

col_d1, col_d2 = st.columns(2)
start_date = col_d1.date_input("Бастапқы дата", df["дата_жеткізу"].min().date())
end_date = col_d2.date_input("Соңғы дата", df["дата_жеткізу"].max().date())

filter_risk = st.checkbox("🛑 Тек қауіпті контейнерлерді көрсету", value=False)
search_query = st.text_input("🔍 Контейнерді іздеу (мысалы: Контейнер 12)", "")

filtered_df = df[
    (df["дата_жеткізу"].dt.date >= start_date) &
    (df["дата_жеткізу"].dt.date <= end_date)
]
if filter_risk:
    filtered_df = filtered_df[filtered_df["болжау_қауіп"] == 1]
if search_query:
    filtered_df = filtered_df[filtered_df["номер_контейнера"].str.contains(search_query, case=False)]

st.write(f"Көрсетілген контейнерлер саны: {len(filtered_df)}")

# --- Контейнер картасы ---
st.subheader("🛰️ Контейнер картасы")

m = folium.Map(location=[43.5, 70.0], zoom_start=4)

for _, row in filtered_df.iterrows():
    color = "red" if row["болжау_қауіп"] == 1 else "green"
    popup_info = (
        f"<b>{row['номер_контейнера']}</b><br>"
        f"Температура: {row['температура_C']:.1f}°C<br>"
        f"Ылғалдылық: {row['ылғалдылық_пайыз']:.1f}%<br>"
        f"Вибрация: {row['вибрация_GPS']:.2f} G<br>"
        f"Қауіп: {'Иә' if row['болжау_қауіп'] else 'Жоқ'}<br>"
        f"Жеткізу күні: {row['дата_жеткізу'].strftime('%Y-%m-%d')}"
    )
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=6,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=popup_info
    ).add_to(m)

st_folium(m, width=1000, height=600)