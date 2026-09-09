"""
================================================================================
 PLANETARY GEARBOX — FULL CUSTOM DESIGN & STRESS CALCULATOR (v2)
================================================================================
Single-stage epicyclic gearbox, target ratio 1:9, design output torque locked
to the 65-75 N.m band, max ring OD 200 mm (all still adjustable in the UI).

Everything that was requested is here:
  - Manual OR auto-search of Sun / Planet / Ring teeth and module
  - Input torque customizable from 5 N.m up to a user ceiling
  - Input speed customizable from 1500 rpm up to a user ceiling
  - Design (worst-case) torque locked to 65-75 N.m via a slider
  - Motor power is NEVER a user input — it is always a *result*, computed
    from torque, speed, ratio and efficiency
  - Full gear-tooth geometry (pitch / base / tip / root dia) for Sun,
    Planet, Ring
  - Tooth forces, tooth bending (root) stress, contact/pitting stress
  - Planet load-sharing factor
  - Planet pin (bending + shear + bearing pressure)
  - Carrier arm bending
  - Carrier plate bending
  - Ring gear rim strength (AGMA-style rim-thickness factor)
  - Sun shaft & Carrier/output shaft sizing (ASME combined torsion+bending)
  - Keys / splines sizing (shear + bearing/crushing)
  - Torsional shaft deflection + approximate mesh (tooth) deflection
  - Rolling-bearing L10 life (main shafts + planet bearing)
  - Thermal / lubrication check (power loss, temperature rise, pitch-line
    velocity vs splash-lubrication limit)
  - Component dimension summary — inner/outer dia, thickness/height, for
    every physical part
  - Consolidated overall PASS / FAIL dashboard

Run with:   streamlit run planetary_gearbox_app.py
Requires :  streamlit, numpy, matplotlib, pandas
--------------------------------------------------------------------------------
NOTE ON ENGINEERING RIGOUR
This tool uses simplified/representative formulas (ISO 6336-lite, ASME shaft
code, AGMA-style rim-thickness factor, Lundberg-Palmgren bearing life, basic
beam/plate-bending approximations for the carrier and simple lumped-parameter
thermal balance). It is a first-pass sizing and learning aid, not a certified
design. Verify every result against full ISO 6336 / AGMA 2001 / ISO 281 /
bearing- and lubricant-manufacturer data before production release.
================================================================================
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ================================================================
# 1. FIXED PROJECT CONSTANTS (defaults — most are adjustable in the UI)
# ================================================================
TARGET_RATIO_DEFAULT   = 9.0     # Target transmission ratio (1:9)
MAX_OD_DEFAULT_MM      = 200.0   # Max outer (ring) diameter (mm)
N_PLANETS_DEFAULT      = 3
EFFICIENCY_DEFAULT     = 0.97    # Mesh efficiency per stage
PRESSURE_ANGLE         = 20.0    # Normal pressure angle (deg) - standard
HELIX_ANGLE            = 0.0     # Spur gears -> beta = 0
MODULE_LIST            = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

DESIGN_OUT_TQ_MIN = 65.0
DESIGN_OUT_TQ_MAX = 75.0

RATED_T_IN_NM_MIN  = 5.0
RATED_N_IN_RPM_MIN = 1500.0

MATERIAL_PROPS = {
    '17CrNiMo6 / 18CrNiMo7-6 (Case Carburized)': {
        'tau': 240.0, 'E': 210000.0, 'nu': 0.30,
        'sigmaF_lim': 430.0, 'sigmaH_lim': 1500.0, 'sigma_allow_bend': 380.0,
        'sigma_allow_bearing': 480.0},
    'Alloy Steel (EN24 / 4340 Hardened)': {
        'tau': 150.0, 'E': 206000.0, 'nu': 0.30,
        'sigmaF_lim': 310.0, 'sigmaH_lim': 1150.0, 'sigma_allow_bend': 260.0,
        'sigma_allow_bearing': 320.0},
    'Case Carburized Steel (20MnCr5 / 16MnCr5)': {
        'tau': 140.0, 'E': 210000.0, 'nu': 0.30,
        'sigmaF_lim': 380.0, 'sigmaH_lim': 1350.0, 'sigma_allow_bend': 320.0,
        'sigma_allow_bearing': 400.0},
    'Stainless Steel (316)': {
        'tau': 50.0, 'E': 193000.0, 'nu': 0.31,
        'sigmaF_lim': 170.0, 'sigmaH_lim': 600.0, 'sigma_allow_bend': 140.0,
        'sigma_allow_bearing': 180.0},
    'Mild Steel (AISI 1020)': {
        'tau': 40.0, 'E': 200000.0, 'nu': 0.29,
        'sigmaF_lim': 140.0, 'sigmaH_lim': 450.0, 'sigma_allow_bend': 110.0,
        'sigma_allow_bearing': 140.0},
    'Custom': {
        'tau': 240.0, 'E': 200000.0, 'nu': 0.30,
        'sigmaF_lim': 300.0, 'sigmaH_lim': 1200.0, 'sigma_allow_bend': 250.0,
        'sigma_allow_bearing': 300.0},
}

ASME_FACTORS = {
    'Gradually applied / steady load':      (1.5, 1.0),
    'Minor shocks (typical machine drive)': (1.5, 1.2),
    'Heavy shocks / frequent starts':       (2.0, 1.5),
}


# ================================================================
# 2. TOOTH-COUNT SYNTHESIS
# ================================================================
def find_teeth_combo(fixed_case, target_ratio, n_planets, s_range=(15, 91), p_range=(15, 91)):
    best_err = float('inf')
    best_combo = (0, 0, 0, 0.0, False)
    for S in range(s_range[0], s_range[1]):
        for P in range(p_range[0], p_range[1]):
            R = S + 2 * P
            assembly_ok = (S + R) % n_planets == 0
            clearance_ok = (S + P) * math.sin(math.radians(180 / n_planets)) > (P + 2)
            if not (assembly_ok and clearance_ok):
                continue
            if fixed_case == 'Ring Fixed':
                ratio = (S + R) / S
            elif fixed_case == 'Sun Fixed':
                ratio = (S + R) / R
            else:
                ratio = R / S
            err = abs(ratio - target_ratio)
            if err < best_err:
                best_err = err
                best_combo = (S, P, R, ratio, True)
                if err < 1e-9:
                    return best_combo
    return best_combo


def evaluate_manual_teeth(fixed_case, S, P, n_planets, target_ratio):
    R = S + 2 * P
    assembly_ok = (S + R) % n_planets == 0
    clearance_ok = (S + P) * math.sin(math.radians(180 / n_planets)) > (P + 2)
    if fixed_case == 'Ring Fixed':
        ratio = (S + R) / S
    elif fixed_case == 'Sun Fixed':
        ratio = (S + R) / R
    else:
        ratio = R / S
    return R, ratio, assembly_ok, clearance_ok, abs(ratio - target_ratio)


# ================================================================
# 3. GEAR GEOMETRY
# ================================================================
def gear_geometry(S, P, R, m, alpha_n_deg, beta_deg=0.0):
    alpha_n = math.radians(alpha_n_deg)
    beta = math.radians(beta_deg)
    alpha_t = math.atan(math.tan(alpha_n) / math.cos(beta))

    def ext_gear(z):
        d = z * m / math.cos(beta)
        return {'z': z, 'd_pitch': d, 'd_base': d * math.cos(alpha_t),
                'd_tip': d + 2 * m, 'd_root': d - 2.5 * m}

    def int_gear(z):
        d = z * m / math.cos(beta)
        return {'z': z, 'd_pitch': d, 'd_base': d * math.cos(alpha_t),
                'd_tip': d - 2 * m, 'd_root': d + 2.5 * m}

    sun, planet, ring = ext_gear(S), ext_gear(P), int_gear(R)
    a_sun_planet = (S + P) * m / (2 * math.cos(beta))
    a_ring_planet = (R - P) * m / (2 * math.cos(beta))
    circular_pitch = math.pi * m
    return {'alpha_t_deg': math.degrees(alpha_t), 'sun': sun, 'planet': planet, 'ring': ring,
            'center_dist_sun_planet': a_sun_planet, 'center_dist_ring_planet': a_ring_planet,
            'circular_pitch': circular_pitch}


# ================================================================
# 4. KINEMATICS
# ================================================================
def compute_kinematics_speeds(fixed_case, S, P, n_in_rpm, ratio_actual):
    output_speed = n_in_rpm / ratio_actual
    if fixed_case == 'Ring Fixed':
        n_sun, n_carrier, n_ring = n_in_rpm, output_speed, 0.0
    elif fixed_case == 'Sun Fixed':
        n_sun, n_carrier, n_ring = 0.0, output_speed, n_in_rpm
    else:
        n_sun, n_carrier, n_ring = n_in_rpm, 0.0, output_speed
    n_planet_spin_rel = abs(n_sun - n_carrier) * (S / P)
    return {'n_sun': n_sun, 'n_ring': n_ring, 'n_carrier': n_carrier,
            'n_planet_spin_rel_carrier': n_planet_spin_rel, 'output_speed': output_speed}


# ================================================================
# 5. GEAR TOOTH FORCE & STRESS ANALYSIS (ISO 6336 - simplified)
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
    n_planets = params.get('n_planets', 3)

    alpha_n_rad = math.radians(alpha_n)
    beta_rad = math.radians(beta)
    alpha_t = math.atan(math.tan(alpha_n_rad) / math.cos(beta_rad))

    rS = (zS * mn) / (2 * math.cos(beta_rad))
    rP = (zP * mn) / (2 * math.cos(beta_rad))
    rR = (zR * mn) / (2 * math.cos(beta_rad))
    rbS, rbP, rbR = rS * math.cos(alpha_t), rP * math.cos(alpha_t), rR * math.cos(alpha_t)
    dS, dP = 2 * rS, 2 * rP

    Ft_SP = (TS / (n_planets * rbS)) * Kp
    Ft_RP = Ft_SP * (rbS / rbR)
    Fn_SP = Ft_SP / math.cos(alpha_t)
    Fn_RP = Ft_RP / math.cos(alpha_t)
    Fr_SP = Ft_SP * math.tan(alpha_t)
    Fr_RP = Ft_RP * math.tan(alpha_t)

    num_common = Ft_SP * KA * KV * KFbeta * KFalpha
    den_common = b * mn
    sigmaF_SP = (num_common / den_common) * YFa_SP * YSa_SP * Yeps * Ybeta
    num_RP = Ft_RP * KA * KV * KFbeta * KFalpha
    sigmaF_RP = (num_RP / den_common) * YFa_RP * YSa_RP * Yeps * Ybeta

    theta = math.radians(theta_deg)
    sigmaF_planet = math.sqrt(sigmaF_SP**2 + sigmaF_RP**2 - 2 * sigmaF_SP * sigmaF_RP * math.cos(theta))
    F_pin = math.sqrt(Fn_SP**2 + Fn_RP**2 - 2 * Fn_SP * Fn_RP * math.cos(theta))

    ZE = math.sqrt(1.0 / (math.pi * (((1 - nu1**2) / E1) + ((1 - nu2**2) / E2))))
    u_SP = zP / zS
    u_RP = zR / zP
    term_SP = (Ft_SP * KA * KV * KHbeta * KHalpha) / (b * dS) * (u_SP + 1) / u_SP
    sigmaH_SP = ZH * ZE * Zeps * Zbeta * math.sqrt(term_SP)
    term_RP = (Ft_RP * KA * KV * KHbeta * KHalpha) / (b * dP) * (u_RP - 1) / u_RP
    sigmaH_RP = ZH * ZE * Zeps * Zbeta * math.sqrt(term_RP)
    sigmaH_ring = sigmaH_RP * ZR
    sigmaF_ring = sigmaF_RP * YR

    return {'Ft_SP': Ft_SP, 'Ft_RP': Ft_RP, 'Fr_SP': Fr_SP, 'Fr_RP': Fr_RP,
            'Fn_SP': Fn_SP, 'Fn_RP': Fn_RP, 'F_pin': F_pin,
            'sigmaF_SP': sigmaF_SP, 'sigmaF_RP': sigmaF_RP, 'sigmaF_planet': sigmaF_planet,
            'sigmaH_SP': sigmaH_SP, 'sigmaH_RP': sigmaH_RP,
            'sigmaH_ring': sigmaH_ring, 'sigmaF_ring': sigmaF_ring,
            'rS': rS, 'rP': rP, 'rR': rR, 'ZE': ZE}


# ================================================================
# 6. SHAFT SIZING (ASME combined torsion + bending)
# ================================================================
def shaft_diameter_asme(T_nmm, M_nmm, tau_allow_mpa, Kb, Kt, Kw=1.0):
    Te = math.sqrt((Kb * M_nmm) ** 2 + (Kt * Kw * T_nmm) ** 2)
    d = (16.0 * Te / (math.pi * tau_allow_mpa)) ** (1.0 / 3.0)
    return d, Te


# ================================================================
# 7. PLANET PIN SIZING
# ================================================================
def design_planet_pin(F_pin_N, span_mm, face_width_mm, sigma_allow_bend,
                       tau_allow_shear, allow_bearing_pressure_mpa):
    M_max = F_pin_N * span_mm / 4.0
    V_support = F_pin_N / 2.0
    d_bend = (32.0 * M_max / (math.pi * sigma_allow_bend)) ** (1.0 / 3.0)
    d_shear = math.sqrt(4.0 * V_support / (math.pi * tau_allow_shear))
    d_pin = max(d_bend, d_shear)
    bearing_pressure = F_pin_N / (d_pin * face_width_mm)
    pressure_ok = bearing_pressure <= allow_bearing_pressure_mpa
    return {'M_max_Nmm': M_max, 'V_support_N': V_support, 'd_bend_mm': d_bend,
            'd_shear_mm': d_shear, 'd_pin_mm': d_pin,
            'bearing_pressure_mpa': bearing_pressure, 'pressure_ok': pressure_ok}


# ================================================================
# 8. ROLLING BEARING LIFE (Lundberg-Palmgren, L10)
# ================================================================
def bearing_L10_life(C_dyn_N, P_equiv_N, n_rpm, bearing_type='Ball'):
    p = 3.0 if bearing_type == 'Ball' else 10.0 / 3.0
    if P_equiv_N <= 0 or n_rpm <= 0:
        return {'L10_Mrev': float('inf'), 'L10_h': float('inf'), 'p': p}
    L10_Mrev = (C_dyn_N / P_equiv_N) ** p
    L10_h = (L10_Mrev * 1.0e6) / (60.0 * n_rpm)
    return {'L10_Mrev': L10_Mrev, 'L10_h': L10_h, 'p': p}


# ================================================================
# 8b. CARRIER, RING RIM, KEYS/SPLINES, DEFLECTION, THERMAL  (NEW)
# ================================================================
def carrier_arm_bending(F_pin_N, arm_length_mm, arm_width_mm, arm_thickness_mm, sigma_allow_mpa):
    """Carrier arm = short cantilever beam from the hub/plate out to the pin
    centre, rectangular cross-section, loaded by the resultant pin force at
    its tip."""
    M = F_pin_N * arm_length_mm
    Z = arm_width_mm * arm_thickness_mm ** 2 / 6.0
    sigma = M / Z if Z > 0 else float('inf')
    sf = sigma_allow_mpa / sigma if sigma > 0 else float('inf')
    return {'M_Nmm': M, 'sigma_mpa': sigma, 'sf': sf, 'pass': sf >= 1.0}


def carrier_plate_bending(F_pin_N, n_planets, r_pin_circle_mm, r_bore_mm,
                           plate_thickness_mm, sigma_allow_mpa):
    """Simplified sector-cantilever model: the plate material between the
    bore (clamped, keyed to the output/input shaft) and the pin circle is
    treated as a cantilevered sector beam of tributary width = pin-circle
    circumference / n_planets, loaded by one pin force at its tip."""
    tributary_width = (2 * math.pi * r_pin_circle_mm) / n_planets
    arm = max(r_pin_circle_mm - r_bore_mm, 1e-6)
    M = F_pin_N * arm
    Z = tributary_width * plate_thickness_mm ** 2 / 6.0
    sigma = M / Z if Z > 0 else float('inf')
    sf = sigma_allow_mpa / sigma if sigma > 0 else float('inf')
    return {'M_Nmm': M, 'tributary_width_mm': tributary_width, 'sigma_mpa': sigma,
            'sf': sf, 'pass': sf >= 1.0}


def ring_rim_strength(sigmaF_ring_mpa, m_use_mm, d_root_ring_mm, d_od_ring_mm, sigmaF_lim_mpa):
    """AGMA-style rim-thickness factor Y_B. Rim that is too thin behind the
    root diameter lets the whole ring flex/crack instead of just the tooth
    root bending as assumed by the standard formula, so a thin rim is
    penalised with an extra multiplier on the calculated bending stress."""
    ht = 2.25 * m_use_mm                      # approx whole tooth depth
    rim_thickness = (d_od_ring_mm - d_root_ring_mm) / 2.0
    mB = rim_thickness / ht if ht > 0 else 0.0
    if mB >= 1.2:
        YB = 1.0
    elif mB > 0:
        YB = 1.6 * math.log(2.242 / mB)
    else:
        YB = float('inf')
    sigmaF_ring_adj = sigmaF_ring_mpa * YB
    sf = sigmaF_lim_mpa / sigmaF_ring_adj if sigmaF_ring_adj > 0 else float('inf')
    return {'rim_thickness_mm': rim_thickness, 'ht_mm': ht, 'mB': mB, 'YB': YB,
            'sigmaF_ring_adj_mpa': sigmaF_ring_adj, 'sf': sf, 'pass': sf >= 1.0}


def key_sizing(T_Nmm, d_shaft_mm, key_width_mm, key_height_mm, key_length_mm,
               tau_allow_mpa, sigma_allow_bearing_mpa):
    """Standard flat/parallel key: tangential force at the shaft surface,
    checked in shear across the key section and in bearing (crushing) on
    the key/keyway flank."""
    F_key = 2.0 * T_Nmm / d_shaft_mm
    tau_key = F_key / (key_width_mm * key_length_mm)
    sigma_bearing = F_key / (0.5 * key_height_mm * key_length_mm)
    sf_shear = tau_allow_mpa / tau_key if tau_key > 0 else float('inf')
    sf_bearing = sigma_allow_bearing_mpa / sigma_bearing if sigma_bearing > 0 else float('inf')
    return {'F_key_N': F_key, 'tau_key_mpa': tau_key, 'sigma_bearing_mpa': sigma_bearing,
            'sf_shear': sf_shear, 'sf_bearing': sf_bearing,
            'pass': (sf_shear >= 1.0 and sf_bearing >= 1.0)}


def torsional_deflection_deg(T_Nmm, L_mm, E_mpa, nu, d_mm):
    """Angular twist of a solid round shaft under torque T over length L."""
    G = E_mpa / (2.0 * (1.0 + nu))
    J = math.pi * d_mm ** 4 / 32.0
    theta_rad = T_Nmm * L_mm / (G * J)
    theta_deg = math.degrees(theta_rad)
    return {'G_mpa': G, 'J_mm4': J, 'theta_deg': theta_deg}


def mesh_tooth_deflection_um(Fn_N, b_mm, E_mpa=210000.0):
    """Very approximate single-tooth-pair mesh stiffness deflection using a
    typical ISO 6336 mesh-stiffness constant c' ~ 0.8*E(steel-on-steel,
    N/mm per micron per mm face width scaled) -- representative only,
    intended to flag gross under-stiffness, not for fatigue design."""
    c_prime = 0.04 * E_mpa / 210000.0 * 20.0   # N/mm per micron, per mm face width (~20 for steel)
    stiffness_N_per_mm = c_prime * b_mm * 1000.0   # convert micron->mm basis
    delta_mm = Fn_N / stiffness_N_per_mm if stiffness_N_per_mm > 0 else float('inf')
    return {'stiffness_N_per_mm': stiffness_N_per_mm, 'delta_um': delta_mm * 1000.0}


def thermal_lubrication_check(power_loss_w, ring_od_mm, face_width_mm,
                               ambient_c=25.0, h_conv=15.0, d_sun_mm=None, n_in_rpm=None):
    """Lumped-parameter housing heat balance (natural convection off an
    approximated cylindrical housing) + pitch-line velocity check to decide
    splash vs. forced/spray lubrication."""
    housing_od_m = (ring_od_mm + 20.0) / 1000.0     # + wall allowance
    housing_len_m = (face_width_mm + 40.0) / 1000.0  # + end-cover allowance
    area_m2 = math.pi * housing_od_m * housing_len_m + 2 * (math.pi * (housing_od_m / 2) ** 2)
    delta_T = power_loss_w / (h_conv * area_m2) if area_m2 > 0 else float('inf')
    steady_state_temp = ambient_c + delta_T

    pitch_line_velocity = None
    if d_sun_mm is not None and n_in_rpm is not None:
        pitch_line_velocity = (math.pi * d_sun_mm / 1000.0 * n_in_rpm) / 60.0  # m/s

    splash_ok = (pitch_line_velocity is not None) and (pitch_line_velocity <= 15.0)
    lube_recommendation = "Splash / bath lubrication acceptable" if splash_ok else \
        "Pitch-line velocity high — consider forced/spray or synthetic low-viscosity lubrication"

    return {'housing_area_m2': area_m2, 'delta_T_c': delta_T, 'steady_state_temp_c': steady_state_temp,
            'pitch_line_velocity_mps': pitch_line_velocity, 'splash_ok': splash_ok,
            'lube_recommendation': lube_recommendation, 'temp_ok': steady_state_temp <= 90.0}


# ================================================================
# 9. 2D SCHEMATIC LAYOUT
# ================================================================
def generate_gear_outline(N, m, r_pitch, phase_angle, is_internal):
    addendum, dedendum = m, 1.25 * m
    if is_internal:
        r_outer, r_inner = r_pitch - addendum, r_pitch + dedendum
    else:
        r_outer, r_inner = r_pitch + addendum, r_pitch - dedendum
    pts_per_tooth = 4
    total_pts = N * pts_per_tooth
    angles = np.linspace(0, 2 * np.pi, total_pts, endpoint=False) + phase_angle
    r = np.zeros(total_pts)
    for i in range(N):
        idx = i * pts_per_tooth
        r[idx], r[idx + 1], r[idx + 2], r[idx + 3] = r_inner, r_outer, r_outer, r_inner
    return r * np.cos(angles), r * np.sin(angles)


def create_gearbox_plot(S, P, R, m, n_planets, fixed_case, tS, d_pin_mm):
    fig, ax = plt.subplots(figsize=(6, 6))
    rS, rP, rR = (S * m) / 2.0, (P * m) / 2.0, (R * m) / 2.0
    rCarrier = rS + rP

    if fixed_case == 'Ring Fixed':
        tC = tS * (S / (S + R)); tP = -tS * (S / P) + tC * (1 + S / P); thetaSunActual = tS
    elif fixed_case == 'Sun Fixed':
        tC = tS * (R / (S + R)); tP = tC * (1 + S / P); thetaSunActual = 0.0
    else:
        tC = 0.0; tP = -tS * (S / P); thetaSunActual = tS

    xS, yS = generate_gear_outline(S, m, rS, thetaSunActual, False)
    ax.fill(xS, yS, color='#D9531E', edgecolor='k', linewidth=1, label='Sun')

    carrierX, carrierY = [], []
    xP_base, yP_base = generate_gear_outline(P, m, rP, 0, False)
    pin_r = max(d_pin_mm / 2.0, 0.5)

    for k in range(n_planets):
        angleP = tC + k * (2 * np.pi / n_planets)
        pX, pY = rCarrier * math.cos(angleP), rCarrier * math.sin(angleP)
        carrierX.append(pX); carrierY.append(pY)
        curAngle = tP + angleP
        cosA, sinA = math.cos(curAngle), math.sin(curAngle)
        xRot = xP_base * cosA - yP_base * sinA
        yRot = xP_base * sinA + yP_base * cosA
        ax.fill(xRot + pX, yRot + pY, color='#EDB120', edgecolor='k', label='Planet' if k == 0 else "")
        pin_th = np.linspace(0, 2 * np.pi, 40)
        ax.fill(pX + pin_r * np.cos(pin_th), pY + pin_r * np.sin(pin_th),
                color='#3B3B3B', label='Planet Pin' if k == 0 else "")
        ax.plot([0, pX], [0, pY], 'b-', linewidth=2)

    ax.plot(carrierX, carrierY, 'bo', markersize=4, label='Carrier Pin Centre')

    xR_in, yR_in = generate_gear_outline(R, m, rR, 0, True)
    rOuter = rR + 2.5 * m
    thArr = np.linspace(0, 2 * np.pi, 120)
    xR_out, yR_out = rOuter * np.cos(thArr), rOuter * np.sin(thArr)
    xR_all = np.concatenate([xR_out, xR_in[::-1]])
    yR_all = np.concatenate([yR_out, yR_in[::-1]])
    ax.fill(xR_all, yR_all, color='gray', alpha=0.4, edgecolor='k', label='Ring Gear')

    ax.plot(0, 0, 'k+', markersize=10, markeredgewidth=1.5)
    limitVal = rOuter * 1.15
    ax.set_xlim([-limitVal, limitVal]); ax.set_ylim([-limitVal, limitVal])
    ax.set_aspect('equal', adjustable='box'); ax.grid(True)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax.set_title(f"Planetary Stage (S:{S} | P:{P} | R:{R} | m:{m:.2f}mm)")
    return fig


# ================================================================
# 10. STREAMLIT INTERFACE
# ================================================================
st.set_page_config(page_title="Planetary Gearbox Calculator", layout="wide")
st.title("⚙️ Planetary Gearbox — Full Custom Design & Stress Calculator")

# ---------------- Sidebar : all inputs ----------------
with st.sidebar:
    st.header("1. Ratio / Envelope / Planets")
    target_ratio = st.number_input("Target Ratio (1:x):", min_value=2.0, max_value=20.0,
                                    value=TARGET_RATIO_DEFAULT, step=0.5)
    max_od_mm = st.number_input("Max Ring OD (mm):", min_value=50.0, max_value=1000.0,
                                 value=MAX_OD_DEFAULT_MM, step=5.0)
    n_planets = st.number_input("Number of Planets:", min_value=3, max_value=6,
                                 value=N_PLANETS_DEFAULT, step=1)
    efficiency = st.number_input("Mesh Efficiency (per stage):", min_value=0.80, max_value=0.99,
                                  value=EFFICIENCY_DEFAULT, step=0.01)
    fixed_case = st.selectbox("Fixed Member:", ['Ring Fixed', 'Sun Fixed', 'Carrier Fixed'])

    st.header("2. Teeth & Module")
    manual_teeth = st.checkbox("Manually set Sun / Planet teeth & module", value=False)
    if manual_teeth:
        S_manual = st.number_input("Sun teeth (Zs):", min_value=10, max_value=150, value=20, step=1)
        P_manual = st.number_input("Planet teeth (Zp):", min_value=10, max_value=150, value=32, step=1)
        m_manual = st.selectbox("Module (mm):", MODULE_LIST, index=2)
    else:
        st.caption("Sun/Planet teeth found automatically for the closest match "
                   "to the target ratio; Ring = Sun + 2×Planet (coaxiality).")

    st.header("3. Operating Conditions (customisable range)")
    t_in_max = st.number_input("Max Input Torque available (N.m):", min_value=RATED_T_IN_NM_MIN,
                                max_value=500.0, value=20.0, step=0.5)
    t_in_nm = st.slider("Input Torque (N.m):", min_value=RATED_T_IN_NM_MIN, max_value=t_in_max,
                         value=RATED_T_IN_NM_MIN, step=0.1)
    n_in_max = st.number_input("Max Input Speed available (rpm):", min_value=RATED_N_IN_RPM_MIN,
                                max_value=20000.0, value=3000.0, step=50.0)
    n_in_rpm = st.slider("Input Speed (rpm):", min_value=RATED_N_IN_RPM_MIN, max_value=n_in_max,
                          value=RATED_N_IN_RPM_MIN, step=10.0)

    st.header("4. Design (worst-case) Torque — locked 65-75 N.m")
    design_out_tq = st.slider("Design Output Torque (N.m):", min_value=DESIGN_OUT_TQ_MIN,
                               max_value=DESIGN_OUT_TQ_MAX, value=70.0, step=0.5)

    st.header("5. Material")
    selected_mat = st.selectbox("Gear / Shaft / Pin / Carrier Material:", list(MATERIAL_PROPS.keys()))
    mat_data = dict(MATERIAL_PROPS[selected_mat])
    if selected_mat == 'Custom':
        with st.expander("Custom material properties", expanded=True):
            mat_data['E'] = st.number_input("Young's Modulus E (MPa):", value=mat_data['E'])
            mat_data['nu'] = st.number_input("Poisson's Ratio:", value=mat_data['nu'], step=0.01)
            mat_data['sigmaF_lim'] = st.number_input("Bending Fatigue Limit σF (MPa):", value=mat_data['sigmaF_lim'])
            mat_data['sigmaH_lim'] = st.number_input("Contact Fatigue Limit σH (MPa):", value=mat_data['sigmaH_lim'])
            mat_data['sigma_allow_bend'] = st.number_input("Allowable Bending Stress (MPa):", value=mat_data['sigma_allow_bend'])
            mat_data['sigma_allow_bearing'] = st.number_input("Allowable Bearing/Crush Stress (MPa):", value=mat_data['sigma_allow_bearing'])

    tau = st.number_input("Allowable Shear Stress τ (MPa) — shafts/pin/keys:",
                           min_value=1.0, max_value=1000.0, value=mat_data['tau'])
    kw = st.number_input("Keyway Stress-Concentration Factor Kw:", min_value=1.0, max_value=2.0,
                          value=1.3, step=0.05)
    shock_label = st.selectbox("Shaft Loading Condition (ASME Kb/Kt):", list(ASME_FACTORS.keys()))
    Kb, Kt = ASME_FACTORS[shock_label]

    st.header("6. Main Bearings (Sun/Ring shaft)")
    sf_bearing = st.number_input("Bearing Service Factor (SF):", min_value=1.0, max_value=3.0,
                                  value=1.5, step=0.1)
    cdyn = st.number_input("Main Bearing Dynamic Capacity Cdyn (N):", min_value=1.0,
                            max_value=100000.0, value=12000.0, step=500.0)
    main_bearing_type = st.selectbox("Main Bearing Type:", ['Ball', 'Roller'])

    st.header("7. Planet Pin / Bearing")
    pin_span_mm = st.number_input("Pin Support Span between carrier plates (mm):",
                                   min_value=5.0, max_value=200.0, value=30.0, step=1.0)
    pin_support = st.selectbox("Planet Pin Support Type:", ['Needle Roller Bearing', 'Plain Bronze Bush'])
    if pin_support == 'Needle Roller Bearing':
        allow_bearing_pressure = st.number_input("Allowable Dynamic Pressure (MPa):", value=25.0)
        cdyn_planet = st.number_input("Planet Bearing Dynamic Capacity Cdyn (N):", value=6000.0, step=250.0)
    else:
        allow_bearing_pressure = st.number_input("Allowable Static Bush Pressure (MPa):", value=10.0)
        cdyn_planet = None

    st.header("8. Carrier Geometry")
    arm_length_mm = st.number_input("Carrier Arm Length (hub-face to pin centre, mm):",
                                     min_value=5.0, max_value=200.0, value=25.0, step=1.0)
    arm_width_mm = st.number_input("Carrier Arm Width (mm):", min_value=5.0, max_value=100.0,
                                    value=18.0, step=1.0)
    arm_thickness_mm = st.number_input("Carrier Arm Thickness (mm):", min_value=3.0, max_value=60.0,
                                        value=10.0, step=1.0)
    plate_thickness_mm = st.number_input("Carrier Plate Thickness (mm):", min_value=3.0, max_value=60.0,
                                          value=12.0, step=1.0)
    plate_bore_margin_mm = st.number_input("Carrier Plate Bore Margin over shaft dia (mm):",
                                            min_value=0.0, max_value=20.0, value=2.0, step=0.5)

    st.header("9. Keys / Splines")
    key_width_mm = st.number_input("Key Width w (mm):", min_value=2.0, max_value=40.0, value=6.0, step=0.5)
    key_height_mm = st.number_input("Key Height h (mm):", min_value=2.0, max_value=40.0, value=6.0, step=0.5)
    key_length_mm = st.number_input("Key Length l (mm):", min_value=5.0, max_value=150.0, value=20.0, step=1.0)

    st.header("10. Shaft Lengths (deflection)")
    shaft_len_in_mm = st.number_input("Input Shaft Length (mm):", min_value=10.0, max_value=500.0,
                                       value=60.0, step=5.0)
    shaft_len_out_mm = st.number_input("Output Shaft Length (mm):", min_value=10.0, max_value=500.0,
                                        value=60.0, step=5.0)

    st.header("11. Thermal")
    ambient_temp_c = st.number_input("Ambient Temperature (°C):", min_value=-10.0, max_value=60.0,
                                      value=25.0, step=1.0)
    h_conv = st.number_input("Convective Coefficient h (W/m².K):", min_value=5.0, max_value=60.0,
                              value=15.0, step=1.0)

    st.header("12. Layout")
    t_sun_phase = st.slider("Sun Rotation Angle (rad) — visual only:", 0.0, 2 * np.pi, 0.0, step=0.05)
    theta_deg = st.slider("Angle between Sun-mesh & Ring-mesh force lines on planet (deg):",
                           60.0, 180.0, 120.0, step=1.0)

# ================================================================
# CALCULATIONS
# ================================================================
if manual_teeth:
    R, ratio_actual, assembly_ok, clearance_ok, ratio_err = evaluate_manual_teeth(
        fixed_case, S_manual, P_manual, n_planets, target_ratio)
    S, P, m_use, found = S_manual, P_manual, m_manual, True
else:
    S, P, R, ratio_actual, found = find_teeth_combo(fixed_case, target_ratio, n_planets)
    assembly_ok = (S + R) % n_planets == 0
    clearance_ok = (S + P) * math.sin(math.radians(180 / n_planets)) > (P + 2)

if not found or S <= 0:
    st.error("No valid tooth combination found. Adjust the ratio / planet count / manual teeth.")
    st.stop()

if fixed_case == 'Ring Fixed':
    input_member, output_member = 'Sun', 'Carrier'
elif fixed_case == 'Sun Fixed':
    input_member, output_member = 'Ring', 'Carrier'
else:
    input_member, output_member = 'Sun', 'Ring'

output_speed = n_in_rpm / ratio_actual
output_torque_nominal = t_in_nm * ratio_actual * efficiency

# ---- Module selection (auto mode only) to satisfy max OD ----
if not manual_teeth:
    m_use = MODULE_LIST[0]
    for m in reversed(MODULE_LIST):
        if (R + 2.5) * m <= max_od_mm:
            m_use = m
            break
est_od = (R + 2.5) * m_use
od_fits = est_od <= max_od_mm

# ---- Motor power — ALWAYS a computed result, never a direct input ----
motor_power_operating_w = (2.0 * math.pi * n_in_rpm / 60.0) * t_in_nm
motor_power_design_w = (2.0 * math.pi * n_in_rpm / 60.0) * (design_out_tq / (ratio_actual * efficiency))

# ---- Full gear geometry ----
geom = gear_geometry(S, P, R, m_use, PRESSURE_ANGLE, HELIX_ANGLE)
d_sun, d_ring = geom['sun']['d_pitch'], geom['ring']['d_pitch']
face_width = 16 * m_use

# ---- Kinematics ----
kin = compute_kinematics_speeds(fixed_case, S, P, n_in_rpm, ratio_actual)

# ---- Shaft sizing (ASME) ----
T_in_design_Nmm = (design_out_tq / (ratio_actual * efficiency)) * 1000.0
T_out_design_Nmm = design_out_tq * 1000.0
d_shaft_in, Te_in = shaft_diameter_asme(T_in_design_Nmm, 0.0, tau, Kb, Kt, kw)
d_shaft_out, Te_out = shaft_diameter_asme(T_out_design_Nmm, 0.0, tau, Kb, Kt, kw)

# ---- Main bearing check ----
d_mesh = d_sun if input_member == 'Sun' else d_ring
ft_design = 2 * ((design_out_tq / ratio_actual) * 1000) / d_mesh
f_res = ft_design / math.cos(math.radians(PRESSURE_ANGLE))
f_design = f_res * sf_bearing
bearing_pass = f_design <= cdyn
main_bearing_speed = n_in_rpm if input_member != 'Carrier' else kin['n_carrier']
main_bearing_life = bearing_L10_life(cdyn, f_res, max(main_bearing_speed, 1e-6), main_bearing_type)

# ---- Gear tooth stress analysis ----
t_design_in_nmm = T_in_design_Nmm
stress_params = {
    'zS': S, 'zP': P, 'zR': R, 'mn': m_use, 'alpha_n': PRESSURE_ANGLE, 'beta': HELIX_ANGLE,
    'b': face_width, 'E1': mat_data['E'], 'E2': mat_data['E'], 'nu1': mat_data['nu'], 'nu2': mat_data['nu'],
    'TS': t_design_in_nmm, 'KA': 1.25, 'KV': 1.15, 'KFbeta': 1.2, 'KFalpha': 1.0,
    'KHbeta': 1.25, 'KHalpha': 1.0, 'Kp': 1.05,
    'YFa': {'S': 2.8, 'P': 2.5, 'R': 2.2}, 'YSa': {'S': 1.5, 'P': 1.6, 'R': 1.7},
    'Yeps': 0.85, 'Ybeta': 1.0, 'ZH': 2.5, 'Zeps': 0.9, 'Zbeta': 1.0,
    'ZR': 1.0, 'YR': 1.0, 'theta_deg': theta_deg, 'n_planets': n_planets,
}
stress_res = planetary_gear_stress_3planets(stress_params)

sfF_SP = mat_data['sigmaF_lim'] / stress_res['sigmaF_SP']
sfF_RP = mat_data['sigmaF_lim'] / stress_res['sigmaF_RP']
sfH_SP = mat_data['sigmaH_lim'] / stress_res['sigmaH_SP']
sfH_RP = mat_data['sigmaH_lim'] / stress_res['sigmaH_RP']

# ---- Planet pin design ----
pin_res = design_planet_pin(stress_res['F_pin'], pin_span_mm, face_width,
                             mat_data['sigma_allow_bend'], tau, allow_bearing_pressure)

if pin_support == 'Needle Roller Bearing' and cdyn_planet:
    planet_bearing_life = bearing_L10_life(cdyn_planet, stress_res['F_pin'],
                                            max(kin['n_planet_spin_rel_carrier'], 1e-6), 'Roller')
else:
    planet_bearing_life = None

# ---- Carrier arm / plate ----
r_pin_circle = geom['sun']['d_pitch'] / 2 + geom['planet']['d_pitch'] / 2
r_bore_carrier = (d_shaft_out / 2.0) + plate_bore_margin_mm
carrier_arm_res = carrier_arm_bending(stress_res['F_pin'], arm_length_mm, arm_width_mm,
                                       arm_thickness_mm, mat_data['sigma_allow_bend'])
carrier_plate_res = carrier_plate_bending(stress_res['F_pin'], n_planets, r_pin_circle,
                                           r_bore_carrier, plate_thickness_mm, mat_data['sigma_allow_bend'])
carrier_plate_od = 2 * (r_pin_circle + pin_res['d_pin_mm'] * 1.5)

# ---- Ring gear rim ----
ring_rim_res = ring_rim_strength(stress_res['sigmaF_ring'], m_use, geom['ring']['d_root'],
                                  est_od, mat_data['sigmaF_lim'])

# ---- Keys / splines (sun input shaft + carrier/ring output shaft) ----
key_sun_res = key_sizing(T_in_design_Nmm, d_shaft_in, key_width_mm, key_height_mm, key_length_mm,
                          tau, mat_data['sigma_allow_bearing'])
key_out_res = key_sizing(T_out_design_Nmm, d_shaft_out, key_width_mm, key_height_mm, key_length_mm,
                          tau, mat_data['sigma_allow_bearing'])

# ---- Deflection ----
defl_in = torsional_deflection_deg(T_in_design_Nmm, shaft_len_in_mm, mat_data['E'], mat_data['nu'], d_shaft_in)
defl_out = torsional_deflection_deg(T_out_design_Nmm, shaft_len_out_mm, mat_data['E'], mat_data['nu'], d_shaft_out)
mesh_defl_sp = mesh_tooth_deflection_um(stress_res['Fn_SP'], face_width, mat_data['E'])
mesh_defl_rp = mesh_tooth_deflection_um(stress_res['Fn_RP'], face_width, mat_data['E'])
defl_in_ok = defl_in['theta_deg'] <= 0.5
defl_out_ok = defl_out['theta_deg'] <= 0.5

# ---- Thermal / lubrication ----
power_loss_w = motor_power_design_w * (1.0 - efficiency)
thermal_res = thermal_lubrication_check(power_loss_w, est_od, face_width, ambient_temp_c, h_conv,
                                         d_sun_mm=d_sun, n_in_rpm=n_in_rpm)

# ---- Component dimensions summary ----
sun_bore_mm = d_shaft_in
planet_bore_mm = pin_res['d_pin_mm'] + 6.0   # + representative needle-bearing / bush radial section
carrier_id_mm = 2 * r_bore_carrier

# ================================================================
# DISPLAY
# ================================================================
c_top1, c_top2, c_top3, c_top4 = st.columns(4)
c_top1.metric("Achieved Ratio", f"1:{ratio_actual:.3f}")
c_top2.metric("Motor Power @ Design Torque", f"{motor_power_design_w:.1f} W")
c_top3.metric("Motor Power @ Operating Torque", f"{motor_power_operating_w:.1f} W")
c_top4.metric("Ring OD (est.)", f"{est_od:.1f} mm")

col_plot, col_dash = st.columns([1, 1.4])

with col_plot:
    fig = create_gearbox_plot(S, P, R, m_use, n_planets, fixed_case, t_sun_phase, pin_res['d_pin_mm'])
    st.pyplot(fig)

    overall_checks = {
        'Ring OD fits envelope': od_fits,
        'Assembly condition': assembly_ok,
        'Clearance condition': clearance_ok,
        'Sun/Planet bending SF≥1': sfF_SP >= 1,
        'Ring/Planet bending SF≥1': sfF_RP >= 1,
        'Sun/Planet contact SF≥1': sfH_SP >= 1,
        'Ring/Planet contact SF≥1': sfH_RP >= 1,
        'Planet pin pressure OK': pin_res['pressure_ok'],
        'Main bearing OK': bearing_pass,
        'Carrier arm bending OK': carrier_arm_res['pass'],
        'Carrier plate bending OK': carrier_plate_res['pass'],
        'Ring rim strength OK': ring_rim_res['pass'],
        'Sun key OK': key_sun_res['pass'],
        'Output key OK': key_out_res['pass'],
        'Input shaft twist ≤0.5°': defl_in_ok,
        'Output shaft twist ≤0.5°': defl_out_ok,
        'Thermal steady-state ≤90°C': thermal_res['temp_ok'],
    }
    overall_ok = all(overall_checks.values())
    st.metric("Overall Design Status", "PASS ✅" if overall_ok else "CHECK REQUIRED ⚠️")
    fails = [k for k, v in overall_checks.items() if not v]
    if fails:
        st.warning("Failing checks: " + ", ".join(fails))

with col_dash:
    tabs = st.tabs(["Kinematics", "Gear Geometry", "Tooth Forces & Bending", "Contact/Pitting",
                     "Planet Load Sharing", "Planet Pin", "Carrier", "Ring Rim",
                     "Shafts", "Keys/Splines", "Deflection", "Bearings", "Thermal/Lube",
                     "Component Dimensions", "Overall Dashboard"])

    with tabs[0]:
        st.subheader("Kinematics")
        kin_df = pd.DataFrame({
            'Parameter': ['Configuration', 'Input Member', 'Output Member', 'Achieved Ratio',
                          'Target Ratio', 'Input Speed (rpm)', 'Output Speed (rpm)',
                          'Sun Speed (rpm)', 'Ring Speed (rpm)', 'Carrier Speed (rpm)',
                          'Planet Spin Speed rel. Carrier (rpm)'],
            'Value': [fixed_case, input_member, output_member, f"{ratio_actual:.3f}",
                      f"{target_ratio:.2f}", f"{n_in_rpm:.1f}", f"{output_speed:.2f}",
                      f"{kin['n_sun']:.2f}", f"{kin['n_ring']:.2f}", f"{kin['n_carrier']:.2f}",
                      f"{kin['n_planet_spin_rel_carrier']:.2f}"]
        })
        st.dataframe(kin_df, hide_index=True, use_container_width=True)
        st.caption(f"Motor Power (operating point): {motor_power_operating_w:.1f} W  |  "
                   f"Motor Power (design point): {motor_power_design_w:.1f} W  |  "
                   f"Nominal Output Torque: {output_torque_nominal:.2f} N.m  |  "
                   f"Design Output Torque: {design_out_tq:.2f} N.m")

    with tabs[1]:
        st.subheader("Tooth Counts & Gear Geometry")
        st.write(f"**Teeth:** Sun z={S} | Planet z={P} | Ring z={R}   "
                 f"(Assembly: {'OK' if assembly_ok else 'FAIL'}, Clearance: {'OK' if clearance_ok else 'FAIL'})")
        st.write(f"**Module:** m = {m_use:.2f} mm  |  **Outer Dia (est.):** {est_od:.1f} mm ≤ "
                 f"{max_od_mm:.0f} mm → {'PASS' if od_fits else 'FAIL'}")
        st.write(f"**Working Transverse Pressure Angle:** {geom['alpha_t_deg']:.2f}°  |  "
                 f"**Circular Pitch:** {geom['circular_pitch']:.2f} mm")
        geo_rows = []
        for label, g in [('Sun', geom['sun']), ('Planet', geom['planet']), ('Ring', geom['ring'])]:
            geo_rows.append({'Gear': label, 'Teeth z': g['z'],
                              'Pitch Dia (mm)': f"{g['d_pitch']:.2f}", 'Base Dia (mm)': f"{g['d_base']:.2f}",
                              'Tip Dia (mm)': f"{g['d_tip']:.2f}", 'Root Dia (mm)': f"{g['d_root']:.2f}"})
        st.dataframe(pd.DataFrame(geo_rows), hide_index=True, use_container_width=True)
        st.write(f"**Centre Distance (Sun-Planet):** {geom['center_dist_sun_planet']:.2f} mm  |  "
                 f"**Centre Distance (Ring-Planet):** {geom['center_dist_ring_planet']:.2f} mm  |  "
                 f"**Face Width b:** {face_width:.1f} mm")

    with tabs[2]:
        st.subheader("Mesh (Tooth) Forces")
        force_df = pd.DataFrame({
            'Quantity': ['Tangential Ft (Sun-Planet)', 'Tangential Ft (Ring-Planet)',
                         'Radial Fr (Sun-Planet)', 'Radial Fr (Ring-Planet)',
                         'Normal Fn (Sun-Planet)', 'Normal Fn (Ring-Planet)', 'Resultant Planet-Pin Load'],
            'Value (N)': [f"{stress_res['Ft_SP']:.1f}", f"{stress_res['Ft_RP']:.1f}",
                          f"{stress_res['Fr_SP']:.1f}", f"{stress_res['Fr_RP']:.1f}",
                          f"{stress_res['Fn_SP']:.1f}", f"{stress_res['Fn_RP']:.1f}", f"{stress_res['F_pin']:.1f}"]
        })
        st.dataframe(force_df, hide_index=True, use_container_width=True)

        st.subheader(f"Root Bending Stress @ Design Load ({design_out_tq:.0f} N.m) — {selected_mat}")
        bend_df = pd.DataFrame([
            {'Check': 'Bending σF — Sun/Planet mesh', 'Actual (MPa)': f"{stress_res['sigmaF_SP']:.1f}",
             'Allowable (MPa)': f"{mat_data['sigmaF_lim']:.0f}", 'SF': f"{sfF_SP:.2f}",
             'Status': 'PASS' if sfF_SP >= 1 else 'FAIL'},
            {'Check': 'Bending σF — Ring/Planet mesh', 'Actual (MPa)': f"{stress_res['sigmaF_RP']:.1f}",
             'Allowable (MPa)': f"{mat_data['sigmaF_lim']:.0f}", 'SF': f"{sfF_RP:.2f}",
             'Status': 'PASS' if sfF_RP >= 1 else 'FAIL'},
            {'Check': 'Combined Planet Root σF', 'Actual (MPa)': f"{stress_res['sigmaF_planet']:.1f}",
             'Allowable (MPa)': f"{mat_data['sigmaF_lim']:.0f}",
             'SF': f"{mat_data['sigmaF_lim']/stress_res['sigmaF_planet']:.2f}",
             'Status': 'PASS' if mat_data['sigmaF_lim']/stress_res['sigmaF_planet'] >= 1 else 'FAIL'},
        ])
        st.dataframe(bend_df, hide_index=True, use_container_width=True)

    with tabs[3]:
        st.subheader("Contact / Pitting (Flank) Stress")
        cont_df = pd.DataFrame([
            {'Check': 'Contact σH — Sun/Planet mesh', 'Actual (MPa)': f"{stress_res['sigmaH_SP']:.1f}",
             'Allowable (MPa)': f"{mat_data['sigmaH_lim']:.0f}", 'SF': f"{sfH_SP:.2f}",
             'Status': 'PASS' if sfH_SP >= 1 else 'FAIL'},
            {'Check': 'Contact σH — Ring/Planet mesh', 'Actual (MPa)': f"{stress_res['sigmaH_RP']:.1f}",
             'Allowable (MPa)': f"{mat_data['sigmaH_lim']:.0f}", 'SF': f"{sfH_RP:.2f}",
             'Status': 'PASS' if sfH_RP >= 1 else 'FAIL'},
            {'Check': 'Ring Gear Adjusted σH', 'Actual (MPa)': f"{stress_res['sigmaH_ring']:.1f}",
             'Allowable (MPa)': f"{mat_data['sigmaH_lim']:.0f}",
             'SF': f"{mat_data['sigmaH_lim']/stress_res['sigmaH_ring']:.2f}", 'Status': '—'},
        ])
        st.dataframe(cont_df, hide_index=True, use_container_width=True)
        st.caption(f"Elasticity factor Z_E = {stress_res['ZE']:.2f} √MPa (from E={mat_data['E']:.0f} MPa, "
                   f"ν={mat_data['nu']:.2f})")

    with tabs[4]:
        st.subheader("Planet Load Sharing")
        st.write(f"**Load-sharing/imbalance factor Kp:** {stress_params['Kp']:.2f}  "
                 f"(inflates the worst-loaded mesh to account for unequal load sharing "
                 f"between the {n_planets} planets)")
        st.write(f"**Nominal load per planet (equal-share):** {t_design_in_nmm/(n_planets):.1f} N.mm portion of input torque")
        st.write(f"**Worst-case mesh tangential force (incl. Kp):** {stress_res['Ft_SP']:.1f} N")
        st.caption("Real planetary trains rarely share load perfectly; Kp>1 conservatively "
                   "loads one mesh more than the theoretical 1/n share. Increase Kp for looser "
                   "manufacturing tolerances or a floating (non-self-aligning) sun.")

    with tabs[5]:
        st.subheader("Planet Pin Design")
        st.caption("Pin modelled as a simply-supported beam spanning the two carrier plates, "
                   "loaded at mid-span by the resultant mesh force.")
        pin_df = pd.DataFrame({
            'Quantity': ['Resultant Mesh Load on Pin', 'Support Span', 'Max Bending Moment',
                         'Support Shear Force (each)', 'Dia. required (bending)', 'Dia. required (shear)',
                         'Design Pin Diameter', 'Bearing/Bush Pressure', 'Allowable Pressure', 'Pressure Check'],
            'Value': [f"{stress_res['F_pin']:.1f} N", f"{pin_span_mm:.1f} mm", f"{pin_res['M_max_Nmm']:.1f} N.mm",
                      f"{pin_res['V_support_N']:.1f} N", f"{pin_res['d_bend_mm']:.2f} mm",
                      f"{pin_res['d_shear_mm']:.2f} mm", f"{pin_res['d_pin_mm']:.2f} mm",
                      f"{pin_res['bearing_pressure_mpa']:.2f} MPa", f"{allow_bearing_pressure:.2f} MPa",
                      'PASS' if pin_res['pressure_ok'] else 'FAIL']
        })
        st.dataframe(pin_df, hide_index=True, use_container_width=True)
        if planet_bearing_life is not None:
            st.write(f"**Planet Needle-Bearing L10 Life:** {planet_bearing_life['L10_Mrev']:.1f} million rev "
                     f"≈ {planet_bearing_life['L10_h']:.0f} hours "
                     f"(at {kin['n_planet_spin_rel_carrier']:.1f} rpm relative spin)")

    with tabs[6]:
        st.subheader("Carrier — Arm Bending & Plate Bending")
        st.markdown("**Carrier Arm** (cantilever beam, hub to pin centre)")
        arm_df = pd.DataFrame({
            'Quantity': ['Resultant Pin Load', 'Arm Length', 'Arm Width', 'Arm Thickness',
                         'Bending Moment', 'Bending Stress', 'Allowable Stress', 'Safety Factor', 'Status'],
            'Value': [f"{stress_res['F_pin']:.1f} N", f"{arm_length_mm:.1f} mm", f"{arm_width_mm:.1f} mm",
                      f"{arm_thickness_mm:.1f} mm", f"{carrier_arm_res['M_Nmm']:.1f} N.mm",
                      f"{carrier_arm_res['sigma_mpa']:.1f} MPa", f"{mat_data['sigma_allow_bend']:.0f} MPa",
                      f"{carrier_arm_res['sf']:.2f}", 'PASS' if carrier_arm_res['pass'] else 'FAIL']
        })
        st.dataframe(arm_df, hide_index=True, use_container_width=True)

        st.markdown("**Carrier Plate** (sector-cantilever from bore to pin circle)")
        plate_df = pd.DataFrame({
            'Quantity': ['Pin Circle Radius', 'Bore Radius (incl. margin)', 'Plate Thickness',
                         'Tributary Width per Planet', 'Bending Moment', 'Bending Stress',
                         'Allowable Stress', 'Safety Factor', 'Status'],
            'Value': [f"{r_pin_circle:.2f} mm", f"{r_bore_carrier:.2f} mm", f"{plate_thickness_mm:.1f} mm",
                      f"{carrier_plate_res['tributary_width_mm']:.2f} mm", f"{carrier_plate_res['M_Nmm']:.1f} N.mm",
                      f"{carrier_plate_res['sigma_mpa']:.1f} MPa", f"{mat_data['sigma_allow_bend']:.0f} MPa",
                      f"{carrier_plate_res['sf']:.2f}", 'PASS' if carrier_plate_res['pass'] else 'FAIL']
        })
        st.dataframe(plate_df, hide_index=True, use_container_width=True)
        st.caption("Simplified beam-sector approximation — a real carrier should also be checked "
                   "by FEA for plate torsion and local pin-boss stress concentration.")

    with tabs[7]:
        st.subheader("Ring Gear Rim Strength")
        rim_df = pd.DataFrame({
            'Quantity': ['Whole Tooth Depth ht', 'Rim Thickness (OD − root radius)', 'Rim Ratio mB',
                         'Rim Thickness Factor YB', 'Ring σF (unadjusted)', 'Ring σF (rim-adjusted)',
                         'Allowable σF', 'Safety Factor', 'Status'],
            'Value': [f"{ring_rim_res['ht_mm']:.2f} mm", f"{ring_rim_res['rim_thickness_mm']:.2f} mm",
                      f"{ring_rim_res['mB']:.2f}", f"{ring_rim_res['YB']:.2f}",
                      f"{stress_res['sigmaF_ring']:.1f} MPa", f"{ring_rim_res['sigmaF_ring_adj_mpa']:.1f} MPa",
                      f"{mat_data['sigmaF_lim']:.0f} MPa", f"{ring_rim_res['sf']:.2f}",
                      'PASS' if ring_rim_res['pass'] else 'FAIL']
        })
        st.dataframe(rim_df, hide_index=True, use_container_width=True)
        st.caption("AGMA guidance recommends a rim-thickness ratio mB ≥ 1.2 (rim thickness ≥ "
                   "1.2× whole tooth depth) to avoid the ring flexing/cracking behind the teeth "
                   "instead of the tooth root governing.")

    with tabs[8]:
        st.subheader("Shaft Sizing (ASME combined torsion + bending)")
        st.caption(f"Loading condition: {shock_label} (Kb={Kb}, Kt={Kt}) | Keyway factor Kw={kw}. "
                   f"External bending assumed negligible (close-coupled bearings).")
        shaft_df = pd.DataFrame([
            {'Shaft': f'Input ({input_member})', 'Design Torque (N.m)': f"{T_in_design_Nmm/1000:.2f}",
             'Equivalent Torque Te (N.m)': f"{Te_in/1000:.2f}", 'Required Dia (mm)': f"{d_shaft_in:.2f}"},
            {'Shaft': f'Output ({output_member})', 'Design Torque (N.m)': f"{T_out_design_Nmm/1000:.2f}",
             'Equivalent Torque Te (N.m)': f"{Te_out/1000:.2f}", 'Required Dia (mm)': f"{d_shaft_out:.2f}"},
        ])
        st.dataframe(shaft_df, hide_index=True, use_container_width=True)

    with tabs[9]:
        st.subheader("Keys / Splines")
        key_df = pd.DataFrame([
            {'Location': f'Sun/Input shaft (⌀{d_shaft_in:.1f}mm)', 'Key Force (N)': f"{key_sun_res['F_key_N']:.1f}",
             'Shear Stress (MPa)': f"{key_sun_res['tau_key_mpa']:.1f}", 'SF Shear': f"{key_sun_res['sf_shear']:.2f}",
             'Bearing Stress (MPa)': f"{key_sun_res['sigma_bearing_mpa']:.1f}",
             'SF Bearing': f"{key_sun_res['sf_bearing']:.2f}", 'Status': 'PASS' if key_sun_res['pass'] else 'FAIL'},
            {'Location': f'Output shaft (⌀{d_shaft_out:.1f}mm)', 'Key Force (N)': f"{key_out_res['F_key_N']:.1f}",
             'Shear Stress (MPa)': f"{key_out_res['tau_key_mpa']:.1f}", 'SF Shear': f"{key_out_res['sf_shear']:.2f}",
             'Bearing Stress (MPa)': f"{key_out_res['sigma_bearing_mpa']:.1f}",
             'SF Bearing': f"{key_out_res['sf_bearing']:.2f}", 'Status': 'PASS' if key_out_res['pass'] else 'FAIL'},
        ])
        st.dataframe(key_df, hide_index=True, use_container_width=True)
        st.caption(f"Key section used: w={key_width_mm:.1f} mm × h={key_height_mm:.1f} mm × "
                   f"l={key_length_mm:.1f} mm (standard flat/parallel key, shear + crush checked).")

    with tabs[10]:
        st.subheader("Deflection")
        st.markdown("**Torsional shaft twist**")
        defl_df = pd.DataFrame([
            {'Shaft': 'Input', 'Length (mm)': f"{shaft_len_in_mm:.1f}", 'Dia (mm)': f"{d_shaft_in:.2f}",
             'G (MPa)': f"{defl_in['G_mpa']:.0f}", 'Twist (deg)': f"{defl_in['theta_deg']:.3f}",
             'Status (≤0.5°)': 'PASS' if defl_in_ok else 'FAIL'},
            {'Shaft': 'Output', 'Length (mm)': f"{shaft_len_out_mm:.1f}", 'Dia (mm)': f"{d_shaft_out:.2f}",
             'G (MPa)': f"{defl_out['G_mpa']:.0f}", 'Twist (deg)': f"{defl_out['theta_deg']:.3f}",
             'Status (≤0.5°)': 'PASS' if defl_out_ok else 'FAIL'},
        ])
        st.dataframe(defl_df, hide_index=True, use_container_width=True)

        st.markdown("**Approximate mesh (tooth-pair) deflection**")
        mesh_df = pd.DataFrame({
            'Mesh': ['Sun-Planet', 'Ring-Planet'],
            'Normal Force (N)': [f"{stress_res['Fn_SP']:.1f}", f"{stress_res['Fn_RP']:.1f}"],
            'Est. Deflection (µm)': [f"{mesh_defl_sp['delta_um']:.2f}", f"{mesh_defl_rp['delta_um']:.2f}"],
        })
        st.dataframe(mesh_df, hide_index=True, use_container_width=True)
        st.caption("Mesh deflection uses a representative single-pair mesh-stiffness constant — "
                   "for load-sharing/transmission-error work use the full ISO 6336 mesh-stiffness "
                   "method instead.")

    with tabs[11]:
        st.subheader("Main Shaft Bearing Check")
        st.write(f"Mesh point diameter used: **{d_mesh:.2f} mm** (on {input_member} shaft)")
        bearing_df = pd.DataFrame({
            'Quantity': ['Design Tangential Force', 'Resultant Radial Force', 'Design Load (×SF)',
                         'Dynamic Capacity Cdyn', 'Check'],
            'Value': [f"{ft_design:.1f} N", f"{f_res:.1f} N", f"{f_design:.1f} N", f"{cdyn:.1f} N",
                      'PASS' if bearing_pass else 'FAIL']
        })
        st.dataframe(bearing_df, hide_index=True, use_container_width=True)
        st.write(f"**L10 Life:** {main_bearing_life['L10_Mrev']:.1f} million rev ≈ "
                 f"{main_bearing_life['L10_h']:.0f} hours (at {main_bearing_speed:.1f} rpm, {main_bearing_type} bearing)")

    with tabs[12]:
        st.subheader("Thermal / Lubrication")
        th_df = pd.DataFrame({
            'Quantity': ['Power Loss (1 − η)', 'Approx. Housing Surface Area', 'Temperature Rise ΔT',
                         'Steady-State Housing Temp', 'Sun Pitch-Line Velocity', 'Splash Lubrication Suitable?'],
            'Value': [f"{power_loss_w:.1f} W", f"{thermal_res['housing_area_m2']:.3f} m²",
                      f"{thermal_res['delta_T_c']:.1f} °C", f"{thermal_res['steady_state_temp_c']:.1f} °C",
                      f"{thermal_res['pitch_line_velocity_mps']:.2f} m/s" if thermal_res['pitch_line_velocity_mps'] else "n/a",
                      'Yes' if thermal_res['splash_ok'] else 'No']
        })
        st.dataframe(th_df, hide_index=True, use_container_width=True)
        st.info(thermal_res['lube_recommendation'])
        st.caption("Lumped natural-convection housing model — for continuous duty at higher power, "
                   "verify against a full thermal network or add forced-air/oil cooling.")

    with tabs[13]:
        st.subheader("Component Dimensions — Inner/Outer Dia, Thickness, Height")
        comp_df = pd.DataFrame([
            {'Component': 'Sun Gear', 'Outer/Tip Dia (mm)': f"{geom['sun']['d_tip']:.2f}",
             'Root Dia (mm)': f"{geom['sun']['d_root']:.2f}", 'Bore/Inner Dia (mm)': f"{sun_bore_mm:.2f}",
             'Face Width/Height (mm)': f"{face_width:.1f}"},
            {'Component': 'Planet Gear', 'Outer/Tip Dia (mm)': f"{geom['planet']['d_tip']:.2f}",
             'Root Dia (mm)': f"{geom['planet']['d_root']:.2f}", 'Bore/Inner Dia (mm)': f"{planet_bore_mm:.2f}",
             'Face Width/Height (mm)': f"{face_width:.1f}"},
            {'Component': 'Ring Gear', 'Outer/Tip Dia (mm)': f"{est_od:.2f}",
             'Root Dia (mm)': f"{geom['ring']['d_root']:.2f}", 'Bore/Inner Dia (mm)': f"{geom['ring']['d_tip']:.2f}",
             'Face Width/Height (mm)': f"{face_width:.1f}"},
            {'Component': 'Carrier Plate', 'Outer/Tip Dia (mm)': f"{carrier_plate_od:.2f}",
             'Root Dia (mm)': "—", 'Bore/Inner Dia (mm)': f"{carrier_id_mm:.2f}",
             'Face Width/Height (mm)': f"{plate_thickness_mm:.1f}"},
            {'Component': 'Carrier Arm (section)', 'Outer/Tip Dia (mm)': f"w={arm_width_mm:.1f}",
             'Root Dia (mm)': "—", 'Bore/Inner Dia (mm)': "—",
             'Face Width/Height (mm)': f"t={arm_thickness_mm:.1f}, L={arm_length_mm:.1f}"},
            {'Component': 'Planet Pin', 'Outer/Tip Dia (mm)': f"{pin_res['d_pin_mm']:.2f}",
             'Root Dia (mm)': "—", 'Bore/Inner Dia (mm)': "—",
             'Face Width/Height (mm)': f"span={pin_span_mm:.1f}"},
            {'Component': 'Input Shaft', 'Outer/Tip Dia (mm)': f"{d_shaft_in:.2f}", 'Root Dia (mm)': "—",
             'Bore/Inner Dia (mm)': "—", 'Face Width/Height (mm)': f"L={shaft_len_in_mm:.1f}"},
            {'Component': 'Output Shaft', 'Outer/Tip Dia (mm)': f"{d_shaft_out:.2f}", 'Root Dia (mm)': "—",
             'Bore/Inner Dia (mm)': "—", 'Face Width/Height (mm)': f"L={shaft_len_out_mm:.1f}"},
        ])
        st.dataframe(comp_df, hide_index=True, use_container_width=True)
        st.caption("Planet bore assumes a representative +6 mm allowance over the pin diameter for "
                   "a needle bearing or bronze bush wall — confirm against the actual bearing/bush "
                   "catalogue part chosen.")

    with tabs[14]:
        st.subheader("Overall PASS / FAIL Dashboard")
        overall_df = pd.DataFrame({'Check': list(overall_checks.keys()),
                                    'Status': ['PASS' if v else 'FAIL' for v in overall_checks.values()]})
        st.dataframe(overall_df, hide_index=True, use_container_width=True)
        st.metric("Overall Design Status", "PASS ✅" if overall_ok else "CHECK REQUIRED ⚠️")

st.divider()
with st.expander("📋 Plain-text Design Summary (copy/export)"):
    summary_text = f"""===== PLANETARY GEARBOX — FULL DESIGN SUMMARY =====
Configuration       : {fixed_case}
Input / Output      : {input_member} -> {output_member}
Achieved Ratio      : {ratio_actual:.3f}  (Target 1:{target_ratio:.2f})
Input Torque/Speed  : {t_in_nm:.3f} N.m @ {n_in_rpm:.1f} rpm
Output Speed        : {output_speed:.2f} rpm
Nominal Out Torque  : {output_torque_nominal:.2f} N.m
Design Out Torque   : {design_out_tq:.2f} N.m  (locked 65-75 N.m band)
Motor Power (oper.) : {motor_power_operating_w:.1f} W   (computed, not a user input)
Motor Power (design): {motor_power_design_w:.1f} W   (computed, not a user input)

Teeth Counts        : S={S} | P={P} | R={R}   (mode: {'manual' if manual_teeth else 'auto-search'})
Module / Outer Dia  : m={m_use:.2f} mm | OD={est_od:.1f} mm <= {max_od_mm:.0f} mm -> {'PASS' if od_fits else 'FAIL'}
Assembly / Clearance: {'OK' if assembly_ok else 'FAIL'} / {'OK' if clearance_ok else 'FAIL'}
Face Width          : {face_width:.1f} mm

--- Gear Stresses @ Design Load ---
Bending SP / RP     : {stress_res['sigmaF_SP']:.1f} / {stress_res['sigmaF_RP']:.1f} MPa  (SF {sfF_SP:.2f} / {sfF_RP:.2f})
Combined Planet SigF: {stress_res['sigmaF_planet']:.2f} MPa
Contact SP / RP     : {stress_res['sigmaH_SP']:.1f} / {stress_res['sigmaH_RP']:.1f} MPa  (SF {sfH_SP:.2f} / {sfH_RP:.2f})

--- Shafts (ASME) ---
Input Shaft Dia     : {d_shaft_in:.2f} mm  (twist {defl_in['theta_deg']:.3f} deg)
Output Shaft Dia    : {d_shaft_out:.2f} mm  (twist {defl_out['theta_deg']:.3f} deg)

--- Planet Pin ---
Resultant Pin Load  : {stress_res['F_pin']:.1f} N
Pin Diameter        : {pin_res['d_pin_mm']:.2f} mm (bend {pin_res['d_bend_mm']:.2f} / shear {pin_res['d_shear_mm']:.2f})
Bearing Pressure    : {pin_res['bearing_pressure_mpa']:.2f} MPa vs {allow_bearing_pressure:.2f} MPa -> {'PASS' if pin_res['pressure_ok'] else 'FAIL'}

--- Carrier ---
Arm Bending SF      : {carrier_arm_res['sf']:.2f} -> {'PASS' if carrier_arm_res['pass'] else 'FAIL'}
Plate Bending SF    : {carrier_plate_res['sf']:.2f} -> {'PASS' if carrier_plate_res['pass'] else 'FAIL'}

--- Ring Rim ---
Rim Ratio mB        : {ring_rim_res['mB']:.2f}  (YB={ring_rim_res['YB']:.2f})  SF={ring_rim_res['sf']:.2f} -> {'PASS' if ring_rim_res['pass'] else 'FAIL'}

--- Keys ---
Sun Key SF (shear/bearing)   : {key_sun_res['sf_shear']:.2f} / {key_sun_res['sf_bearing']:.2f}
Output Key SF (shear/bearing): {key_out_res['sf_shear']:.2f} / {key_out_res['sf_bearing']:.2f}

--- Main Bearing ---
Design Load / Cdyn  : {f_design:.1f} N / {cdyn:.1f} N -> {'PASS' if bearing_pass else 'FAIL'}
L10 Life            : {main_bearing_life['L10_h']:.0f} hours

--- Thermal ---
Power Loss          : {power_loss_w:.1f} W
Steady-State Temp   : {thermal_res['steady_state_temp_c']:.1f} °C -> {'PASS' if thermal_res['temp_ok'] else 'FAIL'}
Lubrication         : {thermal_res['lube_recommendation']}

OVERALL STATUS      : {'PASS' if overall_ok else 'CHECK REQUIRED'}
"""
    st.code(summary_text, language='text')
    st.caption("Simplified sizing tool (ISO 6336-lite / ASME shaft code / AGMA-style rim factor / "
               "Lundberg-Palmgren bearing life / lumped thermal model). Verify against full "
               "standards before production release.")
