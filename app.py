"""
================================================================================
 PLANETARY GEARBOX — COMPLETE DESIGN & STRESS CALCULATOR
================================================================================
Covers, for a 3-planet single-stage epicyclic gearbox:
  1. Tooth-count synthesis (Sun / Planet / Ring) for the target ratio
  2. Full gear geometry (pitch / base / addendum / dedendum diameters, etc.)
  3. Kinematics (speed of every member, incl. planet spin relative to carrier)
  4. Mesh force analysis (tangential, radial, normal, resultant planet-pin load)
  5. Bending (root) and contact (flank) stress per ISO 6336 - simplified form,
     for Sun-Planet mesh, Ring-Planet mesh and the combined planet tooth load
  6. Input & output shaft sizing (ASME combined torsion+bending code)
  7. Planet pin sizing (bending, double shear, bearing/bush pressure)
  8. Rolling bearing life (L10) for main shaft bearings and planet bearing
  9. 2D schematic layout of the gear set
  10. Consolidated PASS/FAIL design-check dashboard

Run with:   streamlit run planetary_gearbox_app.py
Requires :  streamlit, numpy, matplotlib, pandas
--------------------------------------------------------------------------------
NOTE ON ENGINEERING RIGOUR
This tool uses simplified/representative formulas (ISO 6336 lite, ASME shaft
code, Lundberg-Palmgren bearing life). It is meant as a first-pass sizing and
learning aid. For a certified/production design, verify every result against
the full ISO 6336 / AGMA 2001 / ISO 281 / bearing-manufacturer catalogues.
================================================================================
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ================================================================
# 1. FIXED PROJECT CONSTANTS
# ================================================================
TARGET_RATIO   = 9.0          # Target transmission ratio (1:9)
MAX_OD_MM      = 200.0        # Maximum outer (ring) diameter (mm)
N_PLANETS      = 3            # 3-planet configuration
EFFICIENCY     = 0.97         # Mesh efficiency per stage
PRESSURE_ANGLE = 20.0         # Normal pressure angle (deg)
HELIX_ANGLE    = 0.0          # Spur gears -> beta = 0
MODULE_LIST    = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
DESIGN_OUT_TQ  = 75.0         # Max design output torque (N.m) - stress basis

RATED_T_IN_NM  = 5.0
RATED_N_IN_RPM = 1500.0
MOTOR_POWER_W  = (2.0 * math.pi * RATED_N_IN_RPM / 60.0) * \
                 (DESIGN_OUT_TQ / (TARGET_RATIO * EFFICIENCY))

# Material properties: shear-allow (shafts/keys), Young's modulus, Poisson,
# gear bending-fatigue limit, gear contact-fatigue limit, pin bending-allow.
# Values are representative handbook figures (MPa) - not certified data.
MATERIAL_PROPS = {
    '17CrNiMo6 / 18CrNiMo7-6 (Case Carburized)': {
        'tau': 240.0, 'E': 210000.0, 'nu': 0.30,
        'sigmaF_lim': 430.0, 'sigmaH_lim': 1500.0, 'sigma_allow_bend': 380.0},
    'Alloy Steel (EN24 / 4340 Hardened)': {
        'tau': 150.0, 'E': 206000.0, 'nu': 0.30,
        'sigmaF_lim': 310.0, 'sigmaH_lim': 1150.0, 'sigma_allow_bend': 260.0},
    'Case Carburized Steel (20MnCr5 / 16MnCr5)': {
        'tau': 140.0, 'E': 210000.0, 'nu': 0.30,
        'sigmaF_lim': 380.0, 'sigmaH_lim': 1350.0, 'sigma_allow_bend': 320.0},
    'Stainless Steel (316)': {
        'tau': 50.0, 'E': 193000.0, 'nu': 0.31,
        'sigmaF_lim': 170.0, 'sigmaH_lim': 600.0, 'sigma_allow_bend': 140.0},
    'Mild Steel (AISI 1020)': {
        'tau': 40.0, 'E': 200000.0, 'nu': 0.29,
        'sigmaF_lim': 140.0, 'sigmaH_lim': 450.0, 'sigma_allow_bend': 110.0},
    'Custom': {
        'tau': 240.0, 'E': 200000.0, 'nu': 0.30,
        'sigmaF_lim': 300.0, 'sigmaH_lim': 1200.0, 'sigma_allow_bend': 250.0},
}

# ASME shaft-design shock/fatigue factors: {label: (Kb, Kt)}
ASME_FACTORS = {
    'Gradually applied / steady load':        (1.5, 1.0),
    'Minor shocks (typical machine drive)':    (1.5, 1.2),
    'Heavy shocks / frequent starts':          (2.0, 1.5),
}


# ================================================================
# 2. TOOTH-COUNT SYNTHESIS
# ================================================================
def find_teeth_combo(fixed_case, target_ratio, n_planets):
    """Search S (sun) and P (planet) teeth for the closest match to the
    target ratio, subject to the epicyclic assembly condition and physical
    (non-interference) clearance between adjacent planets."""
    best_err = float('inf')
    best_combo = (0, 0, 0, 0.0, False)

    for S in range(15, 91):
        for P in range(15, 91):
            R = S + 2 * P
            assembly_ok = (S + R) % n_planets == 0
            clearance_ok = (S + P) * math.sin(math.radians(180 / n_planets)) > (P + 2)
            if not (assembly_ok and clearance_ok):
                continue

            if fixed_case == 'Ring Fixed':
                ratio = (S + R) / S
            elif fixed_case == 'Sun Fixed':
                ratio = (S + R) / R
            else:  # Carrier Fixed
                ratio = R / S

            err = abs(ratio - target_ratio)
            if err < best_err:
                best_err = err
                best_combo = (S, P, R, ratio, True)
                if err < 1e-9:
                    return best_combo
    return best_combo


# ================================================================
# 3. GEAR GEOMETRY
# ================================================================
def gear_geometry(S, P, R, m, alpha_n_deg, beta_deg=0.0):
    """Full geometric parameter set for the sun / planet / ring gear."""
    alpha_n = math.radians(alpha_n_deg)
    beta = math.radians(beta_deg)
    alpha_t = math.atan(math.tan(alpha_n) / math.cos(beta))

    def ext_gear(z):
        d = z * m / math.cos(beta)
        return {
            'z': z, 'd_pitch': d, 'd_base': d * math.cos(alpha_t),
            'd_tip': d + 2 * m, 'd_root': d - 2.5 * m,
        }

    def int_gear(z):  # internal (ring) gear - addendum/dedendum reversed
        d = z * m / math.cos(beta)
        return {
            'z': z, 'd_pitch': d, 'd_base': d * math.cos(alpha_t),
            'd_tip': d - 2 * m, 'd_root': d + 2.5 * m,
        }

    sun = ext_gear(S)
    planet = ext_gear(P)
    ring = int_gear(R)

    a_sun_planet = (S + P) * m / (2 * math.cos(beta))
    a_ring_planet = (R - P) * m / (2 * math.cos(beta))
    circular_pitch = math.pi * m

    return {
        'alpha_t_deg': math.degrees(alpha_t),
        'sun': sun, 'planet': planet, 'ring': ring,
        'center_dist_sun_planet': a_sun_planet,
        'center_dist_ring_planet': a_ring_planet,
        'circular_pitch': circular_pitch,
    }


# ================================================================
# 4. KINEMATICS (speed of every member)
# ================================================================
def compute_kinematics_speeds(fixed_case, S, P, n_in_rpm, ratio_actual):
    """Returns absolute rpm of sun, ring, carrier and the PLANET SPIN SPEED
    relative to the carrier (the speed that matters for planet-bearing life)."""
    output_speed = n_in_rpm / ratio_actual

    if fixed_case == 'Ring Fixed':          # input = Sun, output = Carrier
        n_sun, n_carrier, n_ring = n_in_rpm, output_speed, 0.0
    elif fixed_case == 'Sun Fixed':         # input = Ring, output = Carrier
        n_sun, n_carrier, n_ring = 0.0, output_speed, n_in_rpm
    else:                                   # Carrier Fixed: input=Sun, output=Ring
        n_sun, n_carrier, n_ring = n_in_rpm, 0.0, output_speed

    n_planet_spin_rel = abs(n_sun - n_carrier) * (S / P)   # about its own axis
    return {
        'n_sun': n_sun, 'n_ring': n_ring, 'n_carrier': n_carrier,
        'n_planet_spin_rel_carrier': n_planet_spin_rel,
        'output_speed': output_speed,
    }


# ================================================================
# 5. GEAR TOOTH FORCE & STRESS ANALYSIS  (ISO 6336 - simplified form)
# ================================================================
def planetary_gear_stress_3planets(params):
    zS, zP, zR = params['zS'], params['zP'], params['zR']
    mn, alpha_n, beta = params['mn'], params['alpha_n'], params['beta']
    b, E1, E2 = params['b'], params['E1'], params['E2']
    nu1, nu2 = params['nu1'], params['nu2']
    TS, KA, KV = params['TS'], params['KA'], params['KV']
    KFbeta, KFalpha = params['KFbeta'], params['KFalpha']
    KHbeta, KHalpha = params['KHbeta'], params['KHalpha']
    Kp = params['Kp']
    YFa_SP, YFa_RP = params['YFa']['S'], params['YFa']['P']
    YSa_SP, YSa_RP = params['YSa']['S'], params['YSa']['P']
    Yeps, Ybeta = params['Yeps'], params['Ybeta']
    ZH, Zeps, Zbeta = params['ZH'], params['Zeps'], params['Zbeta']
    ZR, YR = params['ZR'], params['YR']
    theta_deg = params['theta_deg']

    alpha_n_rad = math.radians(alpha_n)
    beta_rad = math.radians(beta)
    alpha_t = math.atan(math.tan(alpha_n_rad) / math.cos(beta_rad))

    rS = (zS * mn) / (2 * math.cos(beta_rad))
    rP = (zP * mn) / (2 * math.cos(beta_rad))
    rR = (zR * mn) / (2 * math.cos(beta_rad))

    rbS = rS * math.cos(alpha_t)
    rbP = rP * math.cos(alpha_t)
    rbR = rR * math.cos(alpha_t)

    dS, dP = 2 * rS, 2 * rP
    np_planets = 3

    # ---- Tangential / normal / radial mesh forces ----
    Ft_SP = (TS / (np_planets * rbS)) * Kp
    Ft_RP = Ft_SP * (rbS / rbR)
    Fn_SP = Ft_SP / math.cos(alpha_t)
    Fn_RP = Ft_RP / math.cos(alpha_t)
    Fr_SP = Ft_SP * math.tan(alpha_t)
    Fr_RP = Ft_RP * math.tan(alpha_t)

    # ---- Root bending stress (each mesh) ----
    num_common = Ft_SP * KA * KV * KFbeta * KFalpha
    den_common = b * mn
    sigmaF_SP = (num_common / den_common) * YFa_SP * YSa_SP * Yeps * Ybeta

    num_RP = Ft_RP * KA * KV * KFbeta * KFalpha
    sigmaF_RP = (num_RP / den_common) * YFa_RP * YSa_RP * Yeps * Ybeta

    theta = math.radians(theta_deg)
    sigmaF_planet = math.sqrt(sigmaF_SP**2 + sigmaF_RP**2
                               - 2 * sigmaF_SP * sigmaF_RP * math.cos(theta))

    # Resultant force on the planet pin/bearing - same vector combination
    F_pin = math.sqrt(Fn_SP**2 + Fn_RP**2 - 2 * Fn_SP * Fn_RP * math.cos(theta))

    # ---- Contact (flank) stress ----
    ZE = math.sqrt(1.0 / (math.pi * (((1 - nu1**2) / E1) + ((1 - nu2**2) / E2))))

    u_SP = zP / zS
    u_RP = zR / zP

    term_SP = (Ft_SP * KA * KV * KHbeta * KHalpha) / (b * dS) * (u_SP + 1) / u_SP
    sigmaH_SP = ZH * ZE * Zeps * Zbeta * math.sqrt(term_SP)

    term_RP = (Ft_RP * KA * KV * KHbeta * KHalpha) / (b * dP) * (u_RP - 1) / u_RP
    sigmaH_RP = ZH * ZE * Zeps * Zbeta * math.sqrt(term_RP)

    sigmaH_ring = sigmaH_RP * ZR
    sigmaF_ring = sigmaF_RP * YR

    return {
        'Ft_SP': Ft_SP, 'Ft_RP': Ft_RP, 'Fr_SP': Fr_SP, 'Fr_RP': Fr_RP,
        'Fn_SP': Fn_SP, 'Fn_RP': Fn_RP, 'F_pin': F_pin,
        'sigmaF_SP': sigmaF_SP, 'sigmaF_RP': sigmaF_RP,
        'sigmaF_planet': sigmaF_planet, 'sigmaH_SP': sigmaH_SP,
        'sigmaH_RP': sigmaH_RP, 'sigmaH_ring': sigmaH_ring,
        'sigmaF_ring': sigmaF_ring,
    }


# ================================================================
# 6. SHAFT SIZING  (ASME combined torsion + bending code)
# ================================================================
def shaft_diameter_asme(T_nmm, M_nmm, tau_allow_mpa, Kb, Kt, Kw=1.0):
    """Solid round-shaft diameter from combined torsion + bending.
    T, M in N.mm; tau_allow in MPa; Kw = keyway stress-concentration factor
    (applied to the torque term, a common conservative practice)."""
    Te = math.sqrt((Kb * M_nmm)**2 + (Kt * Kw * T_nmm)**2)
    d = (16.0 * Te / (math.pi * tau_allow_mpa)) ** (1.0 / 3.0)
    return d, Te


# ================================================================
# 7. PLANET PIN SIZING  (bending + double shear + bearing pressure)
# ================================================================
def design_planet_pin(F_pin_N, span_mm, face_width_mm, sigma_allow_bend,
                       tau_allow_shear, allow_bearing_pressure_mpa):
    """Planet pin treated as a simply-supported beam (fixed in both carrier
    plates) carrying the resultant mesh force F_pin at mid-span -> double
    shear at the supports, max bending moment at the centre."""
    M_max = F_pin_N * span_mm / 4.0          # simply supported, central load
    V_support = F_pin_N / 2.0                # double shear

    d_bend = (32.0 * M_max / (math.pi * sigma_allow_bend)) ** (1.0 / 3.0)
    d_shear = math.sqrt(4.0 * V_support / (math.pi * tau_allow_shear))
    d_pin = max(d_bend, d_shear)

    bearing_pressure = F_pin_N / (d_pin * face_width_mm)
    pressure_ok = bearing_pressure <= allow_bearing_pressure_mpa

    return {
        'M_max_Nmm': M_max, 'V_support_N': V_support,
        'd_bend_mm': d_bend, 'd_shear_mm': d_shear, 'd_pin_mm': d_pin,
        'bearing_pressure_mpa': bearing_pressure, 'pressure_ok': pressure_ok,
    }


# ================================================================
# 8. ROLLING BEARING LIFE (Lundberg-Palmgren, L10)
# ================================================================
def bearing_L10_life(C_dyn_N, P_equiv_N, n_rpm, bearing_type='Ball'):
    """L10 basic rating life in millions of revolutions and in hours."""
    p = 3.0 if bearing_type == 'Ball' else 10.0 / 3.0
    if P_equiv_N <= 0 or n_rpm <= 0:
        return {'L10_Mrev': float('inf'), 'L10_h': float('inf'), 'p': p}
    L10_Mrev = (C_dyn_N / P_equiv_N) ** p
    L10_h = (L10_Mrev * 1.0e6) / (60.0 * n_rpm)
    return {'L10_Mrev': L10_Mrev, 'L10_h': L10_h, 'p': p}


# ================================================================
# 9. 2D SCHEMATIC LAYOUT
# ================================================================
def generate_gear_outline(N, m, r_pitch, phase_angle, is_internal):
    addendum = m
    dedendum = 1.25 * m
    if is_internal:
        r_outer = r_pitch - addendum
        r_inner = r_pitch + dedendum
    else:
        r_outer = r_pitch + addendum
        r_inner = r_pitch - dedendum

    pts_per_tooth = 4
    total_pts = N * pts_per_tooth
    angles = np.linspace(0, 2 * np.pi, total_pts, endpoint=False) + phase_angle

    r = np.zeros(total_pts)
    for i in range(N):
        idx = i * pts_per_tooth
        r[idx] = r_inner
        r[idx + 1] = r_outer
        r[idx + 2] = r_outer
        r[idx + 3] = r_inner

    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return x, y


def create_gearbox_plot(S, P, R, m, n_planets, fixed_case, tS, d_pin_mm):
    fig, ax = plt.subplots(figsize=(6, 6))

    rS = (S * m) / 2.0
    rP = (P * m) / 2.0
    rR = (R * m) / 2.0
    rCarrier = rS + rP

    if fixed_case == 'Ring Fixed':
        tC = tS * (S / (S + R))
        tP = -tS * (S / P) + tC * (1 + S / P)
        thetaSunActual = tS
    elif fixed_case == 'Sun Fixed':
        tC = tS * (R / (S + R))
        tP = tC * (1 + S / P)
        thetaSunActual = 0.0
    else:  # Carrier Fixed
        tC = 0.0
        tP = -tS * (S / P)
        thetaSunActual = tS

    # Sun
    xS, yS = generate_gear_outline(S, m, rS, thetaSunActual, False)
    ax.fill(xS, yS, color='#D9531E', edgecolor='k', linewidth=1, label='Sun')

    # Planets + carrier arms + pin circle
    carrierX, carrierY = [], []
    xP_base, yP_base = generate_gear_outline(P, m, rP, 0, False)
    pin_r = max(d_pin_mm / 2.0, 0.5)

    for k in range(n_planets):
        angleP = tC + k * (2 * np.pi / n_planets)
        pX = rCarrier * math.cos(angleP)
        pY = rCarrier * math.sin(angleP)
        carrierX.append(pX)
        carrierY.append(pY)

        curAngle = tP + angleP
        cosA, sinA = math.cos(curAngle), math.sin(curAngle)
        xRot = xP_base * cosA - yP_base * sinA
        yRot = xP_base * sinA + yP_base * cosA

        ax.fill(xRot + pX, yRot + pY, color='#EDB120', edgecolor='k',
                 label='Planet' if k == 0 else "")
        # planet pin (small circle at planet centre)
        pin_th = np.linspace(0, 2 * np.pi, 40)
        ax.fill(pX + pin_r * np.cos(pin_th), pY + pin_r * np.sin(pin_th),
                 color='#3B3B3B', label='Planet Pin' if k == 0 else "")
        ax.plot([0, pX], [0, pY], 'b-', linewidth=2)

    ax.plot(carrierX, carrierY, 'bo', markersize=4, label='Carrier Pin Centre')

    # Ring
    xR_in, yR_in = generate_gear_outline(R, m, rR, 0, True)
    rOuter = rR + 2.5 * m
    thArr = np.linspace(0, 2 * np.pi, 120)
    xR_out = rOuter * np.cos(thArr)
    yR_out = rOuter * np.sin(thArr)

    xR_all = np.concatenate([xR_out, xR_in[::-1]])
    yR_all = np.concatenate([yR_out, yR_in[::-1]])
    ax.fill(xR_all, yR_all, color='gray', alpha=0.4, edgecolor='k', label='Ring Gear')

    ax.plot(0, 0, 'k+', markersize=10, markeredgewidth=1.5)
    limitVal = rOuter * 1.15
    ax.set_xlim([-limitVal, limitVal])
    ax.set_ylim([-limitVal, limitVal])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax.set_title(f"Planetary Stage (S:{S} | P:{P} | R:{R} | m:{m:.2f}mm)")

    return fig


# ================================================================
# 10. STREAMLIT INTERFACE
# ================================================================
st.set_page_config(page_title="Planetary Gearbox Calculator", layout="wide")
st.title("⚙️ Planetary Gearbox — Full Design & Stress Calculator")
st.caption(
    f"Ratio (FIXED): 1:{TARGET_RATIO:.0f}  |  Motor Power (FIXED): "
    f"{MOTOR_POWER_W:.1f} W  |  Max OD: {MAX_OD_MM:.0f} mm  |  Planets: {N_PLANETS}"
)

# ---------------- Sidebar : all inputs ----------------
with st.sidebar:
    st.header("Operating Conditions")
    t_in_nm = st.number_input("Input Torque (N.m):", min_value=0.01, max_value=1000.0,
                               value=RATED_T_IN_NM, step=0.1)
    n_in_rpm = st.number_input("Input Speed (RPM):", min_value=1.0, max_value=20000.0,
                                value=RATED_N_IN_RPM, step=50.0)
    fixed_case = st.selectbox("Fixed Member:", ['Ring Fixed', 'Sun Fixed', 'Carrier Fixed'])

    st.header("Material")
    selected_mat = st.selectbox("Gear / Shaft / Pin Material:", list(MATERIAL_PROPS.keys()))
    mat_data = dict(MATERIAL_PROPS[selected_mat])  # copy so custom edits don't mutate original
    if selected_mat == 'Custom':
        with st.expander("Custom material properties", expanded=True):
            mat_data['E'] = st.number_input("Young's Modulus E (MPa):", value=mat_data['E'])
            mat_data['nu'] = st.number_input("Poisson's Ratio:", value=mat_data['nu'], step=0.01)
            mat_data['sigmaF_lim'] = st.number_input("Bending Fatigue Limit σF (MPa):", value=mat_data['sigmaF_lim'])
            mat_data['sigmaH_lim'] = st.number_input("Contact Fatigue Limit σH (MPa):", value=mat_data['sigmaH_lim'])
            mat_data['sigma_allow_bend'] = st.number_input("Pin Allowable Bending Stress (MPa):", value=mat_data['sigma_allow_bend'])

    tau = st.number_input("Allowable Shear Stress τ (MPa) — shafts/pin:",
                           min_value=1.0, max_value=1000.0, value=mat_data['tau'])
    kw = st.number_input("Keyway Stress-Concentration Factor Kw:",
                          min_value=1.0, max_value=2.0, value=1.3, step=0.05)
    shock_label = st.selectbox("Shaft Loading Condition (ASME Kb/Kt):", list(ASME_FACTORS.keys()))
    Kb, Kt = ASME_FACTORS[shock_label]

    st.header("Main Bearings (Sun/Ring shaft)")
    sf = st.number_input("Bearing Service Factor (SF):", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
    cdyn = st.number_input("Main Bearing Dynamic Capacity Cdyn (N):",
                            min_value=1.0, max_value=100000.0, value=12000.0, step=500.0)
    main_bearing_type = st.selectbox("Main Bearing Type:", ['Ball', 'Roller'])

    st.header("Planet Pin / Bearing")
    pin_span_mm = st.number_input("Pin Support Span between carrier plates (mm):",
                                   min_value=5.0, max_value=200.0, value=30.0, step=1.0)
    pin_support = st.selectbox("Planet Pin Support Type:", ['Needle Roller Bearing', 'Plain Bronze Bush'])
    if pin_support == 'Needle Roller Bearing':
        allow_bearing_pressure = st.number_input("Allowable Dynamic Pressure (MPa):", value=25.0)
        cdyn_planet = st.number_input("Planet Bearing Dynamic Capacity Cdyn (N):", value=6000.0, step=250.0)
    else:
        allow_bearing_pressure = st.number_input("Allowable Static Bush Pressure (MPa):", value=10.0)
        cdyn_planet = None

    st.header("Layout")
    t_sun_phase = st.slider("Sun Rotation Angle (rad) — visual only:", 0.0, 2 * np.pi, 0.0, step=0.05)
    theta_deg = st.slider("Angle between Sun-mesh & Ring-mesh force lines on planet (deg):",
                           60.0, 180.0, 120.0, step=1.0)

# ================================================================
# CALCULATIONS
# ================================================================
S, P, R, ratio_actual, found = find_teeth_combo(fixed_case, TARGET_RATIO, N_PLANETS)

if not found:
    st.error("No valid tooth combination found for the target ratio. Try adjusting parameters.")
    st.stop()

if fixed_case == 'Ring Fixed':
    input_member, output_member = 'Sun', 'Carrier'
elif fixed_case == 'Sun Fixed':
    input_member, output_member = 'Ring', 'Carrier'
else:
    input_member, output_member = 'Sun', 'Ring'

output_speed = n_in_rpm / ratio_actual
output_torque_nominal = t_in_nm * ratio_actual * EFFICIENCY

# ---- Module selection to satisfy max OD ----
m_use = MODULE_LIST[0]
for m in reversed(MODULE_LIST):
    if (R + 2.5) * m <= MAX_OD_MM:
        m_use = m
        break
est_od = (R + 2.5) * m_use
od_fits = est_od <= MAX_OD_MM

# ---- Full gear geometry ----
geom = gear_geometry(S, P, R, m_use, PRESSURE_ANGLE, HELIX_ANGLE)
d_sun, d_ring = geom['sun']['d_pitch'], geom['ring']['d_pitch']

# ---- Kinematics ----
kin = compute_kinematics_speeds(fixed_case, S, P, n_in_rpm, ratio_actual)

# ---- Shaft sizing (ASME combined torsion + bending; bending optional) ----
st_sidebar_M_in = 0.0   # simplifying assumption: negligible external bending at gear shafts
st_sidebar_M_out = 0.0
T_in_design_Nmm = (DESIGN_OUT_TQ / (ratio_actual * EFFICIENCY)) * 1000.0   # worst-case input torque
T_out_design_Nmm = DESIGN_OUT_TQ * 1000.0

d_shaft_in, Te_in = shaft_diameter_asme(T_in_design_Nmm, st_sidebar_M_in, tau, Kb, Kt, kw)
d_shaft_out, Te_out = shaft_diameter_asme(T_out_design_Nmm, st_sidebar_M_out, tau, Kb, Kt, kw)

# ---- Main bearing check ----
d_mesh = d_sun if input_member == 'Sun' else d_ring
ft_design = 2 * ((DESIGN_OUT_TQ / ratio_actual) * 1000) / d_mesh
f_res = ft_design / math.cos(math.radians(PRESSURE_ANGLE))
f_design = f_res * sf
bearing_pass = f_design <= cdyn
main_bearing_speed = n_in_rpm if input_member != 'Carrier' else kin['n_carrier']
main_bearing_life = bearing_L10_life(cdyn, f_res, max(main_bearing_speed, 1e-6), main_bearing_type)

# ---- Assembly checks ----
assembly_ok = (S + R) % N_PLANETS == 0
clearance_ok = (S + P) * math.sin(math.radians(180 / N_PLANETS)) > (P + 2)

# ---- Gear tooth stress analysis ----
t_design_in_nmm = (DESIGN_OUT_TQ / (ratio_actual * EFFICIENCY)) * 1000
stress_params = {
    'zS': S, 'zP': P, 'zR': R, 'mn': m_use,
    'alpha_n': PRESSURE_ANGLE, 'beta': HELIX_ANGLE, 'b': 16 * m_use,
    'E1': mat_data['E'], 'E2': mat_data['E'], 'nu1': mat_data['nu'], 'nu2': mat_data['nu'],
    'TS': t_design_in_nmm, 'KA': 1.25, 'KV': 1.15, 'KFbeta': 1.2, 'KFalpha': 1.0,
    'KHbeta': 1.25, 'KHalpha': 1.0, 'Kp': 1.05,
    'YFa': {'S': 2.8, 'P': 2.5, 'R': 2.2},
    'YSa': {'S': 1.5, 'P': 1.6, 'R': 1.7},
    'Yeps': 0.85, 'Ybeta': 1.0, 'ZH': 2.5, 'Zeps': 0.9, 'Zbeta': 1.0,
    'ZR': 1.0, 'YR': 1.0, 'theta_deg': theta_deg,
}
stress_res = planetary_gear_stress_3planets(stress_params)

sfF_SP = mat_data['sigmaF_lim'] / stress_res['sigmaF_SP']
sfF_RP = mat_data['sigmaF_lim'] / stress_res['sigmaF_RP']
sfH_SP = mat_data['sigmaH_lim'] / stress_res['sigmaH_SP']
sfH_RP = mat_data['sigmaH_lim'] / stress_res['sigmaH_RP']

# ---- Planet pin design ----
face_width = 16 * m_use
pin_res = design_planet_pin(stress_res['F_pin'], pin_span_mm, face_width,
                             mat_data['sigma_allow_bend'], tau, allow_bearing_pressure)

if pin_support == 'Needle Roller Bearing' and cdyn_planet:
    planet_bearing_life = bearing_L10_life(
        cdyn_planet, stress_res['F_pin'],
        max(kin['n_planet_spin_rel_carrier'], 1e-6), 'Roller')
else:
    planet_bearing_life = None

# ================================================================
# DISPLAY
# ================================================================
col_plot, col_dash = st.columns([1, 1.3])

with col_plot:
    fig = create_gearbox_plot(S, P, R, m_use, N_PLANETS, fixed_case, t_sun_phase, pin_res['d_pin_mm'])
    st.pyplot(fig)

    overall_ok = (od_fits and assembly_ok and clearance_ok and bearing_pass
                  and pin_res['pressure_ok'] and sfF_SP >= 1 and sfF_RP >= 1
                  and sfH_SP >= 1 and sfH_RP >= 1)
    st.metric("Overall Design Status", "PASS ✅" if overall_ok else "CHECK REQUIRED ⚠️")

with col_dash:
    tabs = st.tabs(["Kinematics", "Gear Geometry", "Forces & Stresses",
                     "Shafts", "Planet Pin", "Bearings"])

    with tabs[0]:
        st.subheader("Kinematics")
        kin_df = pd.DataFrame({
            'Parameter': ['Configuration', 'Input Member', 'Output Member',
                          'Achieved Ratio', 'Target Ratio', 'Input Speed (rpm)',
                          'Output Speed (rpm)', 'Sun Speed (rpm)', 'Ring Speed (rpm)',
                          'Carrier Speed (rpm)', 'Planet Spin Speed rel. Carrier (rpm)'],
            'Value': [fixed_case, input_member, output_member,
                      f"{ratio_actual:.3f}", f"{TARGET_RATIO:.0f}", f"{n_in_rpm:.1f}",
                      f"{output_speed:.2f}", f"{kin['n_sun']:.2f}", f"{kin['n_ring']:.2f}",
                      f"{kin['n_carrier']:.2f}", f"{kin['n_planet_spin_rel_carrier']:.2f}"]
        })
        st.dataframe(kin_df, hide_index=True, use_container_width=True)
        st.caption(f"Motor Power (fixed design point): {MOTOR_POWER_W:.1f} W  |  "
                   f"Nominal Output Torque: {output_torque_nominal:.2f} N.m  |  "
                   f"Design Output Torque: {DESIGN_OUT_TQ:.2f} N.m")

    with tabs[1]:
        st.subheader("Tooth Counts & Gear Geometry")
        st.write(f"**Teeth:** Sun z={S} | Planet z={P} | Ring z={R}   "
                 f"(Assembly: {'OK' if assembly_ok else 'FAIL'}, "
                 f"Clearance: {'OK' if clearance_ok else 'FAIL'})")
        st.write(f"**Module:** m = {m_use:.2f} mm  |  **Outer Dia (est.):** "
                 f"{est_od:.1f} mm ≤ {MAX_OD_MM:.0f} mm → {'PASS' if od_fits else 'FAIL'}")
        st.write(f"**Working Transverse Pressure Angle:** {geom['alpha_t_deg']:.2f}°  |  "
                 f"**Circular Pitch:** {geom['circular_pitch']:.2f} mm")

        geo_rows = []
        for label, g in [('Sun', geom['sun']), ('Planet', geom['planet']), ('Ring', geom['ring'])]:
            geo_rows.append({
                'Gear': label, 'Teeth z': g['z'],
                'Pitch Dia (mm)': f"{g['d_pitch']:.2f}",
                'Base Dia (mm)': f"{g['d_base']:.2f}",
                'Tip Dia (mm)': f"{g['d_tip']:.2f}",
                'Root Dia (mm)': f"{g['d_root']:.2f}",
            })
        st.dataframe(pd.DataFrame(geo_rows), hide_index=True, use_container_width=True)
        st.write(f"**Centre Distance (Sun-Planet):** {geom['center_dist_sun_planet']:.2f} mm  |  "
                 f"**Centre Distance (Ring-Planet):** {geom['center_dist_ring_planet']:.2f} mm  |  "
                 f"**Face Width b:** {face_width:.1f} mm")

    with tabs[2]:
        st.subheader("Mesh Forces")
        force_df = pd.DataFrame({
            'Quantity': ['Tangential Ft (Sun-Planet)', 'Tangential Ft (Ring-Planet)',
                         'Radial Fr (Sun-Planet)', 'Radial Fr (Ring-Planet)',
                         'Normal Fn (Sun-Planet)', 'Normal Fn (Ring-Planet)',
                         'Resultant Planet-Pin Load'],
            'Value (N)': [f"{stress_res['Ft_SP']:.1f}", f"{stress_res['Ft_RP']:.1f}",
                          f"{stress_res['Fr_SP']:.1f}", f"{stress_res['Fr_RP']:.1f}",
                          f"{stress_res['Fn_SP']:.1f}", f"{stress_res['Fn_RP']:.1f}",
                          f"{stress_res['F_pin']:.1f}"]
        })
        st.dataframe(force_df, hide_index=True, use_container_width=True)

        st.subheader(f"Stresses at Design Load ({DESIGN_OUT_TQ:.0f} N.m) — Material: {selected_mat}")
        stress_df = pd.DataFrame([
            {'Check': 'Bending σF — Sun/Planet mesh', 'Actual (MPa)': f"{stress_res['sigmaF_SP']:.1f}",
             'Allowable (MPa)': f"{mat_data['sigmaF_lim']:.0f}", 'Safety Factor': f"{sfF_SP:.2f}",
             'Status': 'PASS' if sfF_SP >= 1 else 'FAIL'},
            {'Check': 'Bending σF — Ring/Planet mesh', 'Actual (MPa)': f"{stress_res['sigmaF_RP']:.1f}",
             'Allowable (MPa)': f"{mat_data['sigmaF_lim']:.0f}", 'Safety Factor': f"{sfF_RP:.2f}",
             'Status': 'PASS' if sfF_RP >= 1 else 'FAIL'},
            {'Check': 'Combined Planet Root σF', 'Actual (MPa)': f"{stress_res['sigmaF_planet']:.1f}",
             'Allowable (MPa)': f"{mat_data['sigmaF_lim']:.0f}",
             'Safety Factor': f"{mat_data['sigmaF_lim']/stress_res['sigmaF_planet']:.2f}",
             'Status': 'PASS' if mat_data['sigmaF_lim']/stress_res['sigmaF_planet'] >= 1 else 'FAIL'},
            {'Check': 'Contact σH — Sun/Planet mesh', 'Actual (MPa)': f"{stress_res['sigmaH_SP']:.1f}",
             'Allowable (MPa)': f"{mat_data['sigmaH_lim']:.0f}", 'Safety Factor': f"{sfH_SP:.2f}",
             'Status': 'PASS' if sfH_SP >= 1 else 'FAIL'},
            {'Check': 'Contact σH — Ring/Planet mesh', 'Actual (MPa)': f"{stress_res['sigmaH_RP']:.1f}",
             'Allowable (MPa)': f"{mat_data['sigmaH_lim']:.0f}", 'Safety Factor': f"{sfH_RP:.2f}",
             'Status': 'PASS' if sfH_RP >= 1 else 'FAIL'},
            {'Check': 'Ring Gear Adjusted σH', 'Actual (MPa)': f"{stress_res['sigmaH_ring']:.1f}",
             'Allowable (MPa)': f"{mat_data['sigmaH_lim']:.0f}",
             'Safety Factor': f"{mat_data['sigmaH_lim']/stress_res['sigmaH_ring']:.2f}", 'Status': '—'},
            {'Check': 'Ring Gear Adjusted σF', 'Actual (MPa)': f"{stress_res['sigmaF_ring']:.1f}",
             'Allowable (MPa)': f"{mat_data['sigmaF_lim']:.0f}",
             'Safety Factor': f"{mat_data['sigmaF_lim']/stress_res['sigmaF_ring']:.2f}", 'Status': '—'},
        ])
        st.dataframe(stress_df, hide_index=True, use_container_width=True)

    with tabs[3]:
        st.subheader("Shaft Sizing (ASME combined torsion + bending code)")
        st.caption(f"Loading condition: {shock_label}  (Kb={Kb}, Kt={Kt})  |  Keyway factor Kw={kw}. "
                   f"External bending moment on the gear shafts is assumed negligible "
                   f"(gears close-coupled to bearings); add a bending moment input if an "
                   f"overhung coupling/pulley is present.")
        shaft_df = pd.DataFrame([
            {'Shaft': 'Input', 'Design Torque (N.m)': f"{T_in_design_Nmm/1000:.2f}",
             'Equivalent Torque Te (N.m)': f"{Te_in/1000:.2f}", 'Required Dia (mm)': f"{d_shaft_in:.2f}"},
            {'Shaft': 'Output', 'Design Torque (N.m)': f"{T_out_design_Nmm/1000:.2f}",
             'Equivalent Torque Te (N.m)': f"{Te_out/1000:.2f}", 'Required Dia (mm)': f"{d_shaft_out:.2f}"},
        ])
        st.dataframe(shaft_df, hide_index=True, use_container_width=True)

    with tabs[4]:
        st.subheader("Planet Pin Design")
        st.caption("Pin modelled as a simply-supported beam spanning the two carrier "
                   "plates, loaded at mid-span by the resultant mesh force → double "
                   "shear at the supports, peak bending at the centre.")
        pin_df = pd.DataFrame({
            'Quantity': ['Resultant Mesh Load on Pin', 'Support Span', 'Max Bending Moment',
                         'Support Shear Force (each)', 'Dia. required (bending)',
                         'Dia. required (shear)', 'Design Pin Diameter',
                         'Bearing/Bush Pressure', 'Allowable Pressure', 'Pressure Check'],
            'Value': [f"{stress_res['F_pin']:.1f} N", f"{pin_span_mm:.1f} mm",
                      f"{pin_res['M_max_Nmm']:.1f} N.mm", f"{pin_res['V_support_N']:.1f} N",
                      f"{pin_res['d_bend_mm']:.2f} mm", f"{pin_res['d_shear_mm']:.2f} mm",
                      f"{pin_res['d_pin_mm']:.2f} mm", f"{pin_res['bearing_pressure_mpa']:.2f} MPa",
                      f"{allow_bearing_pressure:.2f} MPa",
                      'PASS' if pin_res['pressure_ok'] else 'FAIL']
        })
        st.dataframe(pin_df, hide_index=True, use_container_width=True)

        if planet_bearing_life is not None:
            st.write(f"**Planet Needle-Bearing L10 Life:** "
                     f"{planet_bearing_life['L10_Mrev']:.1f} million rev  ≈ "
                     f"{planet_bearing_life['L10_h']:.0f} hours "
                     f"(at {kin['n_planet_spin_rel_carrier']:.1f} rpm relative spin)")

    with tabs[5]:
        st.subheader("Main Shaft Bearing Check")
        st.write(f"Mesh point diameter used: **{d_mesh:.2f} mm** (on {input_member} shaft)")
        bearing_df = pd.DataFrame({
            'Quantity': ['Design Tangential Force', 'Resultant Radial Force',
                         'Design Load (×SF)', 'Dynamic Capacity Cdyn', 'Check'],
            'Value': [f"{ft_design:.1f} N", f"{f_res:.1f} N", f"{f_design:.1f} N",
                      f"{cdyn:.1f} N", 'PASS' if bearing_pass else 'FAIL']
        })
        st.dataframe(bearing_df, hide_index=True, use_container_width=True)
        st.write(f"**L10 Life:** {main_bearing_life['L10_Mrev']:.1f} million rev ≈ "
                 f"{main_bearing_life['L10_h']:.0f} hours (at {main_bearing_speed:.1f} rpm, "
                 f"{main_bearing_type} bearing)")

st.divider()
with st.expander("📋 Plain-text Design Summary (copy/export)"):
    summary_text = f"""===== PLANETARY GEARBOX — FULL DESIGN SUMMARY =====
Configuration       : {fixed_case}
Input / Output      : {input_member} -> {output_member}
Achieved Ratio      : {ratio_actual:.3f}  (Target 1:{TARGET_RATIO:.0f})
Input Torque/Speed  : {t_in_nm:.3f} N.m @ {n_in_rpm:.1f} rpm
Output Speed        : {output_speed:.2f} rpm
Nominal Out Torque  : {output_torque_nominal:.2f} N.m
Design Out Torque   : {DESIGN_OUT_TQ:.2f} N.m
Motor Power (fixed) : {MOTOR_POWER_W:.1f} W

Teeth Counts        : S={S} | P={P} | R={R}
Module / Outer Dia  : m={m_use:.2f} mm | OD={est_od:.1f} mm <= {MAX_OD_MM:.0f} mm -> {'PASS' if od_fits else 'FAIL'}
Assembly / Clearance: {'OK' if assembly_ok else 'FAIL'} / {'OK' if clearance_ok else 'FAIL'}
Face Width          : {face_width:.1f} mm

--- Gear Stresses @ Design Load ---
Bending SP / RP     : {stress_res['sigmaF_SP']:.1f} / {stress_res['sigmaF_RP']:.1f} MPa  (SF {sfF_SP:.2f} / {sfF_RP:.2f})
Combined Planet SigF: {stress_res['sigmaF_planet']:.2f} MPa
Contact SP / RP     : {stress_res['sigmaH_SP']:.1f} / {stress_res['sigmaH_RP']:.1f} MPa  (SF {sfH_SP:.2f} / {sfH_RP:.2f})
Ring Adj. SigH/SigF : {stress_res['sigmaH_ring']:.1f} / {stress_res['sigmaF_ring']:.1f} MPa

--- Shafts (ASME) ---
Input Shaft Dia     : {d_shaft_in:.2f} mm
Output Shaft Dia    : {d_shaft_out:.2f} mm

--- Planet Pin ---
Resultant Pin Load  : {stress_res['F_pin']:.1f} N
Pin Diameter        : {pin_res['d_pin_mm']:.2f} mm (bend {pin_res['d_bend_mm']:.2f} / shear {pin_res['d_shear_mm']:.2f})
Bearing Pressure    : {pin_res['bearing_pressure_mpa']:.2f} MPa vs {allow_bearing_pressure:.2f} MPa -> {'PASS' if pin_res['pressure_ok'] else 'FAIL'}

--- Main Bearing ---
Design Load / Cdyn  : {f_design:.1f} N / {cdyn:.1f} N -> {'PASS' if bearing_pass else 'FAIL'}
L10 Life            : {main_bearing_life['L10_h']:.0f} hours
"""
    st.code(summary_text, language='text')
    st.caption("Simplified sizing tool (ISO 6336-lite / ASME shaft code / Lundberg-Palmgren "
               "bearing life). Verify against full standards before production release.")
