import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ==========================================
# 1. FIXED PROJECT CONSTANTS & SETUP
# ==========================================
TARGET_RATIO = 9.0        # Target transmission ratio (1:9)
MAX_OD_MM = 200.0         # Maximum outer diameter (mm)
N_PLANETS = 3             # 3-planet setup
EFFICIENCY = 0.97         # Mesh efficiency per stage
PRESSURE_ANGLE = 20.0     # Pressure angle (degrees)
MODULE_LIST = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
DESIGN_OUT_TQ = 75.0      # Maximum Design Output Torque (N.m)

# Rated point for Motor Power calculation
RATED_T_IN_NM = 5.0
RATED_N_IN_RPM = 1500.0
MOTOR_POWER_W = (2.0 * math.pi * RATED_N_IN_RPM / 60.0) * (DESIGN_OUT_TQ / (TARGET_RATIO * EFFICIENCY))

# Material properties mapping: (Shear Stress Tau, E1/E2, Poisson nu)
MATERIAL_PROPS = {
    '17CrNiMo6 / 18CrNiMo7-6 (Case Carburized)': {'tau': 240.0, 'E': 210000.0, 'nu': 0.30},
    'Alloy Steel (EN24 / 4340 Hardened)':       {'tau': 150.0, 'E': 206000.0, 'nu': 0.30},
    'Case Carburized Steel (20MnCr5 / 16MnCr5)': {'tau': 140.0, 'E': 210000.0, 'nu': 0.30},
    'Stainless Steel (316)':                    {'tau': 50.0,  'E': 193000.0, 'nu': 0.31},
    'Mild Steel (AISI 1020)':                   {'tau': 40.0,  'E': 200000.0, 'nu': 0.29},
    'Custom':                                   {'tau': 240.0, 'E': 200000.0, 'nu': 0.30}
}

# ==========================================
# 2. CALCULATION ENGINES
# ==========================================
def find_teeth_combo(fixed_case, target_ratio, n_planets):
    best_err = float('inf')
    best_combo = (0, 0, 0, 0.0, False)
    
    for S in range(15, 91):
        for P in range(15, 91):
            R = S + 2 * P
            # Assembly mesh criteria & physical clearance between planets
            if not ((S + R) % n_planets == 0 and (S + P) * math.sin(math.radians(180 / n_planets)) > (P + 2)):
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
    
    Ft_SP = (TS / (np_planets * rbS)) * Kp
    Ft_RP = Ft_SP * (rbS / rbR)
    
    num_common = Ft_SP * KA * KV * KFbeta * KFalpha
    den_common = b * mn
    sigmaF_SP = (num_common / den_common) * YFa_SP * YSa_SP * Yeps * Ybeta
    
    num_RP = Ft_RP * KA * KV * KFbeta * KFalpha
    sigmaF_RP = (num_RP / den_common) * YFa_RP * YSa_RP * Yeps * Ybeta
    
    theta = math.radians(theta_deg)
    sigmaF_planet = math.sqrt(sigmaF_SP**2 + sigmaF_RP**2 - 2 * sigmaF_SP * sigmaF_RP * math.cos(theta))
    
    ZE = math.sqrt(1.0 / (math.pi * (((1 - nu1**2) / E1) + ((1 - nu2**2) / E2))))
    
    u_SP = zP / zS
    u_RP = zR / zP
    
    d1_SP = dS
    term_SP = (Ft_SP * KA * KV * KHbeta * KHalpha) / (b * d1_SP) * (u_SP + 1) / u_SP
    sigmaH_SP = ZH * ZE * Zeps * Zbeta * math.sqrt(term_SP)
    
    d1_RP = dP
    term_RP = (Ft_RP * KA * KV * KHbeta * KHalpha) / (b * d1_RP) * (u_RP - 1) / u_RP
    sigmaH_RP = ZH * ZE * Zeps * Zbeta * math.sqrt(term_RP)
    
    sigmaH_ring = sigmaH_RP * ZR
    sigmaF_ring = sigmaF_RP * YR
    
    return {
        'Ft_SP': Ft_SP, 'Ft_RP': Ft_RP,
        'sigmaF_SP': sigmaF_SP, 'sigmaF_RP': sigmaF_RP,
        'sigmaF_planet': sigmaF_planet, 'sigmaH_SP': sigmaH_SP,
        'sigmaH_RP': sigmaH_RP, 'sigmaH_ring': sigmaH_ring,
        'sigmaF_ring': sigmaF_ring
    }

# ==========================================
# 3. GRAPHICS GENERATION ENGINE
# ==========================================
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

def create_gearbox_plot(S, P, R, m, n_planets, fixed_case, tS):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    rS = (S * m) / 2.0
    rP = (P * m) / 2.0
    rR = (R * m) / 2.0
    rCarrier = rS + rP
    
    # Kinematics
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

    # 1. Sun Gear
    xS, yS = generate_gear_outline(S, m, rS, thetaSunActual, False)
    ax.fill(xS, yS, color='#D9531E', edgecolor='k', linewidth=1, label='Sun')

    # 2. Planet Gears & Carrier Arms
    carrierX, carrierY = [], []
    xP_base, yP_base = generate_gear_outline(P, m, rP, 0, False)
    
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
        
        ax.fill(xRot + pX, yRot + pY, color='#EDB120', edgecolor='k', label='Planet' if k == 0 else "")
        ax.plot([0, pX], [0, pY], 'b-', linewidth=2)

    ax.plot(carrierX, carrierY, 'bo', markersize=6, label='Carrier Pins')
    
    # 3. Ring Gear
    xR_in, yR_in = generate_gear_outline(R, m, rR, 0, True)
    rOuter = rR + 2.5 * m
    thArr = np.linspace(0, 2 * np.pi, 120)
    xR_out = rOuter * np.cos(thArr)
    yR_out = rOuter * np.sin(thArr)
    
    # Combine ring outer and inner boundaries
    xR_all = np.concatenate([xR_out, xR_in[::-1]])
    yR_all = np.concatenate([yR_out, yR_in[::-1]])
    ax.fill(xR_all, yR_all, color='gray', alpha=0.4, edgecolor='k', label='Ring Gear')

    ax.plot(0, 0, 'k+', markersize=10, markeredgewidth=1.5)
    limitVal = rOuter * 1.15
    ax.set_xlim([-limitVal, limitVal])
    ax.set_ylim([-limitVal, limitVal])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.set_title(f"Planetary Stage (S:{S} | P:{P} | R:{R} | m:{m:.2f}mm)")
    
    return fig

# ==========================================
# 4. STREAMLIT INTERFACE
# ==========================================
st.set_page_config(page_title="Planetary Gearbox Calculator", layout="wide")

st.title("⚙️ Planetary Gearbox Calculator & Stress Engine")
st.caption(f"Ratio (FIXED): 1:{TARGET_RATIO:.0f} | Motor Power (FIXED): {MOTOR_POWER_W:.1f} W | Max OD: {MAX_OD_MM:.0f} mm | Planets: {N_PLANETS}")

# Layout columns
col_controls, col_display = st.columns([1, 1.2])

with col_controls:
    st.subheader("Operating & Material Inputs")
    
    c1, c2 = st.columns(2)
    with c1:
        t_in_nm = st.number_input("Input Torque (N.m):", min_value=0.01, max_value=1000.0, value=RATED_T_IN_NM, step=0.1)
    with c2:
        n_in_rpm = st.number_input("Input Speed (RPM):", min_value=1.0, max_value=20000.0, value=RATED_N_IN_RPM, step=50.0)

    fixed_case = st.selectbox("Fixed Member:", ['Ring Fixed', 'Sun Fixed', 'Carrier Fixed'])
    
    selected_mat = st.selectbox("Gear Material:", list(MATERIAL_PROPS.keys()))
    default_tau = MATERIAL_PROPS[selected_mat]['tau']
    
    c3, c4 = st.columns(2)
    with c3:
        tau = st.number_input("Allowable Shear Tau (MPa):", min_value=1.0, max_value=1000.0, value=default_tau)
        kw = st.number_input("Keyway Factor (Kw):", min_value=1.0, max_value=2.0, value=1.3, step=0.05)
    with c4:
        sf = st.number_input("Bearing Service Factor (SF):", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
        cdyn = st.number_input("Bearing Capacity (N):", min_value=1.0, max_value=100000.0, value=12000.0, step=500.0)

    # Optional Rotation Slider to simulate tooth positioning
    t_sun_phase = st.slider("Sun Rotation Angle (Rad):", 0.0, 2 * np.pi, 0.0, step=0.05)

# --- CALCULATIONS ---
S, P, R, ratio_actual, found = find_teeth_combo(fixed_case, TARGET_RATIO, N_PLANETS)

if not found:
    st.error("No valid tooth combination found. Try adjusting parameters.")
else:
    # Kinematics
    if fixed_case == 'Ring Fixed':
        input_member, output_member = 'Sun', 'Carrier'
    elif fixed_case == 'Sun Fixed':
        input_member, output_member = 'Ring', 'Carrier'
    else:
        input_member, output_member = 'Sun', 'Ring'

    output_speed = n_in_rpm / ratio_actual
    output_torque_nominal = t_in_nm * ratio_actual * EFFICIENCY
    
    # Module Selection
    m_use = MODULE_LIST[0]
    for m in reversed(MODULE_LIST):
        if (R + 2.5) * m <= MAX_OD_MM:
            m_use = m
            break
            
    est_od = (R + 2.5) * m_use
    od_fits = est_od <= MAX_OD_MM
    d_sun = S * m_use
    d_ring = R * m_use
    
    # Shaft Sizing (Evaluated at Max Design Torque Capacity 75 N.m)
    d_shaft_in = ((16 * (t_in_nm * 1000) * kw) / (math.pi * tau)) ** (1/3)
    d_shaft_out = ((16 * (DESIGN_OUT_TQ * 1000) * kw) / (math.pi * tau)) ** (1/3)
    
    # Bearing Check
    d_mesh = d_sun if input_member == 'Sun' else d_ring
    ft_design = 2 * ((DESIGN_OUT_TQ / ratio_actual) * 1000) / d_mesh
    f_res = ft_design / math.cos(math.radians(PRESSURE_ANGLE))
    f_design = f_res * sf
    bearing_pass = f_design <= cdyn
    
    # Assembly Check
    assembly_ok = (S + R) % N_PLANETS == 0
    clearance_ok = (S + P) * math.sin(math.radians(180 / N_PLANETS)) > (P + 2)
    
    # Stress Parameters Evaluation
    mat_data = MATERIAL_PROPS[selected_mat]
    t_design_in_nmm = (DESIGN_OUT_TQ / (ratio_actual * EFFICIENCY)) * 1000
    stress_params = {
        'zS': S, 'zP': P, 'zR': R, 'mn': m_use,
        'alpha_n': PRESSURE_ANGLE, 'beta': 0, 'b': 16 * m_use,
        'E1': mat_data['E'], 'E2': mat_data['E'], 'nu1': mat_data['nu'], 'nu2': mat_data['nu'],
        'TS': t_design_in_nmm, 'KA': 1.25, 'KV': 1.15, 'KFbeta': 1.2, 'KFalpha': 1.0,
        'KHbeta': 1.25, 'KHalpha': 1.0, 'Kp': 1.05,
        'YFa': {'S': 2.8, 'P': 2.5, 'R': 2.2},
        'YSa': {'S': 1.5, 'P': 1.6, 'R': 1.7},
        'Yeps': 0.85, 'Ybeta': 1.0, 'ZH': 2.5, 'Zeps': 0.9, 'Zbeta': 1.0,
        'ZR': 1.0, 'YR': 1.0, 'theta_deg': 120
    }
    stress_res = planetary_gear_stress_3planets(stress_params)

    with col_display:
        # Plot rendering
        fig = create_gearbox_plot(S, P, R, m_use, N_PLANETS, fixed_case, t_sun_phase)
        st.pyplot(fig)

    # Output Console View
    st.subheader("Design & Stress Results")
    
    results_text = f"""===== PLANETARY GEARBOX DESIGN RESULT =====
Configuration      : {fixed_case}
Input / Output     : {input_member} -> {output_member}
Achieved Ratio     : {ratio_actual:.3f} (Target 1:{TARGET_RATIO:.0f}, FIXED)
Input Torque/Speed : {t_in_nm:.3f} N.m @ {n_in_rpm:.1f} rpm (User Set)
Nominal Out Torque : {output_torque_nominal:.2f} N.m
Design Out Torque  : {DESIGN_OUT_TQ:.2f} N.m (Max Load Capacity)
Output Speed       : {output_speed:.2f} rpm
Motor Power (FIXED): {MOTOR_POWER_W:.1f} W

Teeth Counts       : S={S} | P={P} | R={R}
Assembly / Clear.  : {'OK' if assembly_ok else 'FAIL'} / {'OK' if clearance_ok else 'FAIL'}
Module / Outer Dia : m={m_use:.2f} mm | OD={est_od:.1f} mm <= {MAX_OD_MM:.0f}mm ({'PASS' if od_fits else 'FAIL'})

Shaft Dia (In/Out) : {d_shaft_in:.2f} mm / {d_shaft_out:.2f} mm
Bearing Load Check : {f_design:.1f} N vs Cap {cdyn:.1f} N -> {'PASS' if bearing_pass else 'FAIL'}

===== STRESS ANALYSIS AT {DESIGN_OUT_TQ:.0f} N.m DESIGN LOAD =====
Material Selected   : {selected_mat}
Tangential Force SP : {stress_res['Ft_SP']:.1f} N
Bending Stress SP   : {stress_res['sigmaF_SP']:.2f} MPa
Bending Stress RP   : {stress_res['sigmaF_RP']:.2f} MPa
Combined Planet SigF: {stress_res['sigmaF_planet']:.2f} MPa
Contact Stress SP   : {stress_res['sigmaH_SP']:.2f} MPa
Contact Stress RP   : {stress_res['sigmaH_RP']:.2f} MPa
Ring Gear Adj SigH  : {stress_res['sigmaH_ring']:.2f} MPa
"""
    st.code(results_text, language='text')
