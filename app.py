import streamlit as st
import math

# 1. Page Configuration MUST be the very first Streamlit command executed
st.set_page_config(page_title="Planetary Gearbox Calculator", layout="wide")

st.title("⚙️ Planetary Gearbox Geometry & CAD Calculator")
st.write("Input your gear parameters below to check assembly alignment and mechanical metrics.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Core Parameters")
module = st.sidebar.number_input("Module (mm)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
pressure_angle = st.sidebar.slider("Pressure Angle (°)", min_value=14.5, max_value=25.0, value=20.0, step=0.5)

st.sidebar.header("2. Tooth Configuration")
N_sun = st.sidebar.number_input("Sun Gear Teeth (Ns)", min_value=10, max_value=100, value=12, step=1)
N_planet = st.sidebar.number_input("Planet Gear Teeth (Np)", min_value=10, max_value=100, value=42, step=1)

# Automatic calculation of Ring Gear teeth for clean alignment
N_ring = N_sun + (2 * N_planet)
num_planets = st.sidebar.number_input("Number of Planets (P)", min_value=2, max_value=6, value=3, step=1)

# --- CALCULATIONS ---
d_sun = module * N_sun
d_planet = module * N_planet
d_ring = module * N_ring
center_dist = (d_sun + d_planet) / 2
gear_ratio = 1 + (N_ring / N_sun)

# Assembly Rules Verification
assembly_factor = (N_sun + N_ring) / num_planets
is_assembly_valid = assembly_factor.is_integer()

# --- INTERFACE DISPLAY ---
col1, col2, col3 = st.columns(3)
col1.metric("Calculated Ring Teeth (Nr)", int(N_ring))
col2.metric("Gear Ratio (Fixed Ring)", f"{gear_ratio:.2f}:1")
col3.metric("Center Distance", f"{center_dist:.2f} mm")

st.subheader("📋 Component Specifications Table")
geo_data = {
    "Component": ["Sun Gear", "Planet Gear", "Ring Gear"],
    "Teeth Count": [N_sun, N_planet, int(N_ring)],
    "Pitch Diameter (mm)": [d_sun, d_planet, d_ring],
    "Outer Diameter (mm)": [d_sun + (2*module), d_planet + (2*module), d_ring - (2*module)]
}
st.table(geo_data)

st.subheader("🧩 Assembly Condition Check")
if is_assembly_valid:
    st.success(f"✔️ Assembly Config Valid! Index factor is a whole integer: {int(assembly_factor)}")
else:
    st.error(f"❌ Index Check Failed! Factor is {assembly_factor:.2f}. Gears will jam unless spacing is adjusted manually.")

st.subheader("🛠️ CAD Export Profile Variables")
st.info(f"Use these parameters in your CAD Tool (SolidWorks/Fusion 360):\n"
        f"- Module: {module} mm\n"
        f"- Pressure Angle: {pressure_angle}°\n"
        f"- Mapped Center Distance Circle: {center_dist} mm")


