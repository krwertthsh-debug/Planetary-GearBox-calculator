"""
================================================================================
 PLANETARY GEARBOX DESIGNER - EDITION (v5)
================================================================================
 Single-stage, 3-planet epicyclic gearbox designer. Best-of-all-versions merge:

   - Tooth synthesis (manual + auto-search + exact-ratio candidate tables)
   - Full gear geometry (pitch/base/tip/root) for Sun/Planet/Ring (helix-aware)
   - Kinematics supporting Ring-Fixed, Sun-Fixed or Carrier-Fixed
   - ISO 6336-lite stress engine: real Lewis form-factor interpolation,
     zone factor, speed-dependent dynamic factor Kv, face/transverse load
     distribution factors, and life factors (ZN/YN) from a target duty life
   - ASME shaft sizing + torsional deflection
   - Planet pin sizing (bending + double shear + bearing/bush pressure)
   - Main + planet bearing L10 life (Lundberg-Palmgren)
   - Carrier arm & plate bending, AGMA-style ring-rim thickness factor,
     key/spline shear + bearing checks
   - Full component dimension table (incl. housing envelope)
   - Consolidated PASS/FAIL dashboard
   - True-involute 3D solids (Plotly) with explode slider + metallic CAD look
   - Parametric OpenSCAD export with every view / part
   - Full parameter glossary + JSON config export

 NOTE ON ENGINEERING RIGOUR
 This is a first-pass sizing/learning aid using simplified formulas
 (ISO 6336-lite, ASME shaft code, AGMA-style rim factor, Lundberg-Palmgren
 bearing life). The default inputs are chosen so the baseline design computes
 a genuine PASS -- but every check is computed live from your inputs, not
 hard-coded, so pushing torque/speed/envelope hard enough will show FAIL.

 Run with:   streamlit run planetary_gearbox_designer.py
 Requires :  streamlit, numpy, pandas, plotly
================================================================================
"""

import math
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

PI = math.pi

# ================================================================
# 1. CONSTANTS & DEFAULTS
# ================================================================
TARGET_RATIO_DEFAULT   = 9.0
MAX_OD_DEFAULT_MM       = 200.0
N_PLANETS               = 3           # fixed 3-planet configuration
EFFICIENCY_DEFAULT      = 0.97
PRESSURE_ANGLE_DEFAULT  = 20.0
HELIX_ANGLE_DEFAULT     = 0.0
MODULE_LIST             = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
DESIGN_TQ_MIN, DESIGN_TQ_MAX = 65.0, 75.0
FACE_WIDTH_FACTOR_DEFAULT = 16.0
RATED_T_IN_NM_MIN  = 1.0
RATED_N_IN_RPM_MIN = 100.0

MATERIALS = {
    '17CrNiMo6 / 18CrNiMo7-6 (Case Carburized)': {
        'E': 210000.0, 'nu': 0.30, 'tau': 240.0, 'sigmaF': 430.0, 'sigmaH': 1500.0,
        'sigma_allow_bend': 380.0, 'sigma_allow_bearing': 480.0, 'HB': 600, 'density': 7850.0},
    'Case Carburized Steel (20MnCr5 / 16MnCr5)': {
        'E': 210000.0, 'nu': 0.30, 'tau': 140.0, 'sigmaF': 380.0, 'sigmaH': 1350.0,
        'sigma_allow_bend': 320.0, 'sigma_allow_bearing': 400.0, 'HB': 550, 'density': 7850.0},
    'Alloy Steel (EN24 / 4340 Hardened)': {
        'E': 206000.0, 'nu': 0.30, 'tau': 150.0, 'sigmaF': 310.0, 'sigmaH': 1150.0,
        'sigma_allow_bend': 260.0, 'sigma_allow_bearing': 320.0, 'HB': 350, 'density': 7850.0},
    'SAE 6150 / 51CrV4 (Spring Steel)': {
        'E': 207000.0, 'nu': 0.30, 'tau': 170.0, 'sigmaF': 350.0, 'sigmaH': 1250.0,
        'sigma_allow_bend': 290.0, 'sigma_allow_bearing': 360.0, 'HB': 400, 'density': 7850.0},
    'Nitralloy 135M (Nitrided)': {
        'E': 205000.0, 'nu': 0.30, 'tau': 180.0, 'sigmaF': 450.0, 'sigmaH': 1250.0,
        'sigma_allow_bend': 320.0, 'sigma_allow_bearing': 400.0, 'HB': 650, 'density': 7850.0},
    'Stainless Steel (316)': {
        'E': 193000.0, 'nu': 0.31, 'tau': 50.0, 'sigmaF': 170.0, 'sigmaH': 600.0,
        'sigma_allow_bend': 140.0, 'sigma_allow_bearing': 180.0, 'HB': 200, 'density': 8000.0},
    'Mild Steel (AISI 1020)': {
        'E': 200000.0, 'nu': 0.29, 'tau': 40.0, 'sigmaF': 140.0, 'sigmaH': 450.0,
        'sigma_allow_bend': 110.0, 'sigma_allow_bearing': 140.0, 'HB': 150, 'density': 7870.0},
    'Custom': {
        'E': 200000.0, 'nu': 0.30, 'tau': 180.0, 'sigmaF': 300.0, 'sigmaH': 1200.0,
        'sigma_allow_bend': 250.0, 'sigma_allow_bearing': 300.0, 'HB': 400, 'density': 7850.0},
}

ASME_FACTORS = {
    'Gradually applied / steady load':      (1.5, 1.0),
    'Minor shocks (typical machine drive)': (1.5, 1.2),
    'Heavy shocks / frequent starts':       (2.0, 1.5),
}

PARAM_GLOSSARY = {
    "Operating & Ratio": [
        ("T_in", "Input shaft torque", "N·m", "user slider", "Drives mesh loads; scaled by ratio & eta for output torque."),
        ("n_in", "Input shaft speed", "rpm", "user slider", "Sets omega_in and, with ratio, output/planet spin speeds."),
        ("i (ratio)", "Overall transmission ratio", "-", "target, e.g. 9", "Ring-fixed: i=(S+R)/S. Sun-fixed: i=(S+R)/R. Carrier-fixed: i=R/S."),
        ("eta (efficiency)", "Single-stage mesh efficiency", "-", "0.95-0.97", "T_out = T_in*i*eta."),
        ("Design Torque", "Worst-case output torque, locked band", "N·m", "65-75", "Basis for every strength check."),
        ("Motor Power", "Required input power - always a computed result", "W", "computed", "P=T*omega; never entered directly."),
    ],
    "Tooth Synthesis": [
        ("S / P / R", "Sun / Planet / Ring tooth counts", "-", "manual or auto-search", "R = S+2P keeps sun/planet/ring coaxial."),
        ("Assembly condition", "(S+R) mod N = 0", "-", "must hold", "Lets all N planets phase into mesh simultaneously."),
        ("Clearance condition", "(S+P)*sin(180/N) > P+2", "-", "must hold", "Stops adjacent planet tips overlapping."),
        ("m (module)", "Tooth module", "mm", "auto-selected", "Largest module that still fits the OD envelope."),
    ],
    "Gear Geometry": [
        ("d_pitch/base/tip/root", "Reference/base/outer/root circle diameters", "mm", "computed", "Base diameter drives force-arm calcs; root governs bending."),
        ("b (face width)", "Axial gear width", "mm", "16*m (default rule)", "Wider face spreads the tooth load thinner."),
        ("Centre distance", "Sun-Planet / Ring-Planet spacing", "mm", "computed", "Fixes the physical layout radius."),
    ],
    "Forces & ISO 6336-lite Stress": [
        ("Ft / Fr / Fn", "Tangential / radial / normal mesh force", "N", "computed", "Fn drives Hertzian contact stress; Ft drives root bending."),
        ("Kp", "Planet load-sharing factor", "-", "1.05-1.15", "Inflates the worst-loaded mesh for unequal load sharing."),
        ("KA / Kv", "Application / dynamic factor", "-", "1.0-1.5", "Kv is speed- and accuracy-grade-dependent (computed)."),
        ("KHb / KFb", "Face load-distribution factor", "-", "computed from b/d & accuracy", "Penalises uneven load across the face width."),
        ("YFa / YSa", "Lewis form / stress-correction factor", "-", "interpolated vs tooth count", "Captures tooth-shape effect on root bending moment arm."),
        ("ZH / ZE", "Zone / elasticity factor", "-", "computed", "Curvature & material stiffness terms in Hertzian contact stress."),
        ("ZN / YN", "Life factors (contact / bending)", "-", "from desired duty life", "Raise allowable stress for short target lives, 1.0 beyond reference cycles."),
        ("theta", "Angle between Sun-mesh & Ring-mesh forces on the planet", "deg", "~120", "Vector-combines the two mesh loads into the planet-pin resultant."),
    ],
    "Carrier / Ring / Keys": [
        ("Arm bending", "Carrier arm as a cantilever beam", "MPa", "computed", "sigma=M/Z; flags an undersized arm section."),
        ("Plate bending", "Carrier plate sector as a cantilever", "MPa", "computed", "Tributary width = pin-circle circumference / N planets."),
        ("Y_B", "AGMA-style ring rim-thickness factor", "-", "1.0 if mB>=1.2", "A thin rim behind the teeth is penalised with extra stress."),
        ("F_key / tau_key / sigma_bearing", "Key tangential force / shear / crush stress", "N, MPa", "computed", "F=2T/d; standard flat-key shear + bearing checks."),
    ],
    "Bearings & Deflection": [
        ("C_dyn", "Bearing dynamic load rating (catalogue value)", "N", "user input", "Load a bearing sustains for 1M rev at 90% survival."),
        ("L10", "Basic rating life", "10^6 rev / hours", "L10=(C/P)^p", "p=3 ball, 10/3 roller."),
        ("theta (twist)", "Torsional shaft deflection", "deg", "T*L/(G*J)", "Checked against a practical 0.5 deg limit."),
    ],
    "Component Dimensions": [
        ("bore_d", "Shaft bore through a gear/hub", "mm", "computed", "Sized from shaft dia + running clearance."),
        ("hub_od", "Outer hub diameter", "mm", "computed", "Representative rule from bore/shaft dia."),
        ("rim_thickness", "Radial wall behind the teeth", "mm", "computed", "Governs ring rim strength."),
        ("housing", "Envelope casing around the ring gear", "mm", "computed", "OD, wall, length, flange, base thickness."),
    ],
}


# ================================================================
# 2. TOOTH-COUNT SYNTHESIS
# ================================================================
def ratio_for(fixed_case, S, R):
    if fixed_case == 'Ring Fixed':
        return (S + R) / S
    elif fixed_case == 'Sun Fixed':
        return (S + R) / R
    else:
        return R / S


def find_teeth_combo(fixed_case, target_ratio, n_planets, s_range=(15, 91), p_range=(15, 91)):
    best_err, best_combo = float('inf'), (0, 0, 0, 0.0, False)
    for S in range(*s_range):
        for P in range(*p_range):
            R = S + 2 * P
            if (S + R) % n_planets != 0:
                continue
            if (S + P) * math.sin(math.radians(180 / n_planets)) <= (P + 2):
                continue
            ratio = ratio_for(fixed_case, S, R)
            err = abs(ratio - target_ratio)
            if err < best_err:
                best_err = err
                best_combo = (S, P, R, ratio, True)
                if err < 1e-9:
                    return best_combo
    return best_combo


def evaluate_manual_teeth(fixed_case, S, P, n_planets, target_ratio):
    R = S + 2 * P
    assembly = (S + R) % n_planets == 0
    clearance = (S + P) * math.sin(math.radians(180 / n_planets)) > (P + 2)
    ratio = ratio_for(fixed_case, S, R)
    return R, ratio, assembly, clearance, abs(ratio - target_ratio)


def suggest_tooth_sets(fixed_case, target_ratio, n_planets, max_od, modules=MODULE_LIST):
    rows = []
    for S in range(15, 71):
        for P in range(15, 101):
            R = S + 2 * P
            if (S + R) % n_planets != 0:
                continue
            if (S + P) * math.sin(math.radians(180 / n_planets)) <= (P + 2):
                continue
            ratio = ratio_for(fixed_case, S, R)
            if abs(ratio - target_ratio) > 1e-6:
                continue
            for m in modules:
                od = (R + 2.5) * m + 12.0
                if od <= max_od:
                    rows.append(dict(S=S, P=P, R=R, ratio=ratio, module=m, est_od=od))
    return pd.DataFrame(rows)


# ================================================================
# 3. GEAR GEOMETRY
# ================================================================
def gear_geometry(S, P, R, m, alpha_n_deg, beta_deg=0.0):
    alpha_n = math.radians(alpha_n_deg)
    beta = math.radians(beta_deg)
    alpha_t = math.atan(math.tan(alpha_n) / math.cos(beta))

    def ext(z):
        d = z * m / math.cos(beta)
        return {'z': z, 'd_pitch': d, 'd_base': d * math.cos(alpha_t),
                'd_tip': d + 2 * m, 'd_root': d - 2.5 * m}

    def internal(z):
        d = z * m / math.cos(beta)
        return {'z': z, 'd_pitch': d, 'd_base': d * math.cos(alpha_t),
                'd_tip': d - 2 * m, 'd_root': d + 2.5 * m}

    sun, planet, ring = ext(S), ext(P), internal(R)
    return {
        'alpha_t_deg': math.degrees(alpha_t), 'sun': sun, 'planet': planet, 'ring': ring,
        'center_dist_sun_planet': (S + P) * m / (2 * math.cos(beta)),
        'center_dist_ring_planet': (R - P) * m / (2 * math.cos(beta)),
        'circular_pitch': PI * m,
    }


# ================================================================
# 4. KINEMATICS (supports all 3 fixed-member configurations)
# ================================================================
def compute_kinematics(fixed_case, S, P, n_in_rpm, ratio_actual):
    output_speed = n_in_rpm / ratio_actual
    if fixed_case == 'Ring Fixed':
        n_sun, n_carrier, n_ring = n_in_rpm, output_speed, 0.0
        input_member, output_member = 'Sun', 'Carrier'
    elif fixed_case == 'Sun Fixed':
        n_sun, n_carrier, n_ring = 0.0, output_speed, n_in_rpm
        input_member, output_member = 'Ring', 'Carrier'
    else:
        n_sun, n_carrier, n_ring = n_in_rpm, 0.0, output_speed
        input_member, output_member = 'Sun', 'Ring'
    n_planet_rel = abs(n_sun - n_carrier) * (S / P)
    return {'n_sun': n_sun, 'n_ring': n_ring, 'n_carrier': n_carrier,
            'n_planet_rel_carrier': n_planet_rel, 'output_speed': output_speed,
            'input_member': input_member, 'output_member': output_member}


# ================================================================
# 5. ISO 6336-LITE HELPERS
# ================================================================
def tooth_form_factors(z):
    z_tab = [15, 17, 20, 25, 30, 40, 50, 60, 80, 100]
    yf_tab = [2.95, 2.85, 2.75, 2.65, 2.55, 2.45, 2.35, 2.25, 2.15, 2.05]
    ys_tab = [1.52, 1.54, 1.56, 1.58, 1.60, 1.62, 1.64, 1.66, 1.68, 1.70]
    zc = min(max(z, 15), 100)
    return float(np.interp(zc, z_tab, yf_tab)), float(np.interp(zc, z_tab, ys_tab))


def zone_factor(alpha_t_deg, beta_deg=0.0):
    alpha_t = math.radians(alpha_t_deg)
    beta = math.radians(beta_deg)
    beta_b = math.atan(math.tan(beta) * math.cos(alpha_t))
    return math.sqrt((2 * math.cos(beta_b) * math.cos(alpha_t)) /
                     (math.cos(alpha_t) ** 2 * math.tan(alpha_t)))


def dynamic_factor_Kv(pitch_line_velocity, z1, accuracy_grade):
    grades = [5, 6, 7, 8]
    k1v = [0.04, 0.08, 0.14, 0.20]
    k2v = [0.01, 0.02, 0.04, 0.06]
    idx = max(0, min(int(round(accuracy_grade)) - 5, 3))
    K1, K2 = k1v[idx], k2v[idx]
    vz = max(pitch_line_velocity, 0.0) * z1
    return 1 + (K1 * vz / 100 + K2) * math.sqrt(vz / 100)


def load_distribution_K_Hbeta(b, d1, accuracy_grade):
    if d1 <= 0 or b / d1 <= 0.5:
        return 1.0
    return 1.0 + 0.15 * (b / d1 - 0.5) * max(11 - accuracy_grade, 1) / 3


def load_distribution_K_Halpha(epsilon_alpha):
    if epsilon_alpha >= 2.0:
        return 1.0
    elif epsilon_alpha > 1.0:
        return 1.0 + (2.0 - epsilon_alpha) * 0.2
    return 1.2


def contact_ratio_external(z1, z2, m, alpha_deg):
    alpha = math.radians(alpha_deg)
    r1, r2 = z1 * m / 2, z2 * m / 2
    ra1, ra2 = r1 + m, r2 + m
    rb1, rb2 = r1 * math.cos(alpha), r2 * math.cos(alpha)
    a = r1 + r2
    num = math.sqrt(max(ra1 ** 2 - rb1 ** 2, 0)) + math.sqrt(max(ra2 ** 2 - rb2 ** 2, 0)) - a * math.sin(alpha)
    return num / (PI * m * math.cos(alpha))


def contact_ratio_internal(zr, zp, m, alpha_deg):
    alpha = math.radians(alpha_deg)
    rr, rp = zr * m / 2, zp * m / 2
    ra_r, ra_p = rr - m, rp + m
    rb_r, rb_p = rr * math.cos(alpha), rp * math.cos(alpha)
    a = rr - rp
    num = math.sqrt(max(ra_p ** 2 - rb_p ** 2, 0)) - math.sqrt(max(ra_r ** 2 - rb_r ** 2, 0)) + a * math.sin(alpha)
    return num / (PI * m * math.cos(alpha))


def life_factor(N_cycles, kind='contact'):
    ref = 1e7 if kind == 'contact' else 3e6
    exp = 1 / 6 if kind == 'contact' else 1 / 10
    return 1.0 if N_cycles >= ref else (max(N_cycles, 1.0) / ref) ** exp


# ================================================================
# 6. MESH FORCES + STRESS (ISO 6336-lite)
# ================================================================
def planetary_gear_stress(params):
    S, P, R = params['S'], params['P'], params['R']
    m, alpha_n, beta = params['m'], params['alpha_n'], params['beta']
    b, E1, E2, nu1, nu2 = params['b'], params['E1'], params['E2'], params['nu1'], params['nu2']
    TS, KA, KV, Kp = params['TS'], params['KA'], params['KV'], params['Kp']
    KFbeta, KFalpha, KHbeta, KHalpha = params['KFbeta'], params['KFalpha'], params['KHbeta'], params['KHalpha']
    YFa, YSa, Yeps, Ybeta = params['YFa'], params['YSa'], params['Yeps'], params['Ybeta']
    ZH, Zeps, Zbeta = params['ZH'], params['Zeps'], params['Zbeta']
    theta_deg, n_planets = params['theta_deg'], params['n_planets']

    alpha_t = math.atan(math.tan(math.radians(alpha_n)) / math.cos(math.radians(beta)))
    beta_r = math.radians(beta)
    rS = S * m / (2 * math.cos(beta_r)); rP = P * m / (2 * math.cos(beta_r)); rR = R * m / (2 * math.cos(beta_r))
    rbS, rbP, rbR = rS * math.cos(alpha_t), rP * math.cos(alpha_t), rR * math.cos(alpha_t)
    dS, dP = 2 * rS, 2 * rP

    Ft_sp = (TS / (n_planets * rbS)) * Kp
    Ft_rp = Ft_sp * (rbS / rbR)
    Fn_sp, Fn_rp = Ft_sp / math.cos(alpha_t), Ft_rp / math.cos(alpha_t)
    Fr_sp, Fr_rp = Ft_sp * math.tan(alpha_t), Ft_rp * math.tan(alpha_t)

    common = KA * KV * KFbeta * KFalpha
    sF_sp = (Ft_sp * common / (b * m)) * YFa['S'] * YSa['S'] * Yeps * Ybeta
    sF_rp = (Ft_rp * common / (b * m)) * YFa['P'] * YSa['P'] * Yeps * Ybeta
    theta = math.radians(theta_deg)
    sF_planet = math.sqrt(max(sF_sp ** 2 + sF_rp ** 2 - 2 * sF_sp * sF_rp * math.cos(theta), 0.0))
    F_pin = math.sqrt(max(Fn_sp ** 2 + Fn_rp ** 2 - 2 * Fn_sp * Fn_rp * math.cos(theta), 0.0))

    ZE = math.sqrt(1.0 / (PI * ((1 - nu1 ** 2) / E1 + (1 - nu2 ** 2) / E2)))
    u_sp, u_rp = P / S, R / P
    term_sp = (Ft_sp * KA * KV * KHbeta * KHalpha) / (b * dS) * (u_sp + 1) / u_sp
    term_rp = (Ft_rp * KA * KV * KHbeta * KHalpha) / (b * dP) * max(u_rp - 1, 1e-9) / u_rp
    sH_sp = ZH * ZE * Zeps * Zbeta * math.sqrt(max(term_sp, 0.0))
    sH_rp = ZH * ZE * Zeps * Zbeta * math.sqrt(max(term_rp, 0.0))

    return dict(Ft_SP=Ft_sp, Ft_RP=Ft_rp, Fr_SP=Fr_sp, Fr_RP=Fr_rp, Fn_SP=Fn_sp, Fn_RP=Fn_rp,
                sigmaF_SP=sF_sp, sigmaF_RP=sF_rp, sigmaF_planet=sF_planet,
                sigmaH_SP=sH_sp, sigmaH_RP=sH_rp, F_pin=F_pin, ZE=ZE)


# ================================================================
# 7. SHAFT / PIN / DEFLECTION / BEARINGS
# ================================================================
def shaft_diameter_asme(T_Nmm, M_Nmm, tau_allow, Kb, Kt, Kw=1.0):
    Te = math.sqrt((Kb * M_Nmm) ** 2 + (Kt * Kw * T_Nmm) ** 2)
    return (16.0 * Te / (PI * tau_allow)) ** (1.0 / 3.0), Te


def design_planet_pin(F_N, span_mm, face_width_mm, sigma_bend, tau_shear, allow_pressure):
    Mmax = F_N * span_mm / 4.0
    V = F_N / 2.0
    d_bend = (32.0 * Mmax / (PI * sigma_bend)) ** (1.0 / 3.0)
    d_shear = math.sqrt(4.0 * V / (PI * tau_shear))
    d = max(d_bend, d_shear)
    p = F_N / (d * face_width_mm)
    return dict(M_max=Mmax, V=V, d_bend=d_bend, d_shear=d_shear, d_pin=d,
                bearing_pressure=p, pressure_ok=p <= allow_pressure)


def torsional_deflection_deg(T_Nmm, L_mm, E_mpa, nu, d_mm):
    G = E_mpa / (2.0 * (1.0 + nu))
    J = PI * d_mm ** 4 / 32.0
    return {'G': G, 'J': J, 'theta_deg': math.degrees(T_Nmm * L_mm / (G * J))}


def shaft_diameter_for_twist(T_Nmm, L_mm, E_mpa, nu, theta_max_deg=0.5):
    """Minimum shaft diameter so torsional twist <= theta_max_deg."""
    G = E_mpa / (2.0 * (1.0 + nu))
    theta_rad = math.radians(theta_max_deg)
    return (32.0 * T_Nmm * L_mm / (PI * G * theta_rad)) ** 0.25


def bearing_L10_life(C_dyn, P_eq, n_rpm, bearing_type='Ball'):
    p = 3.0 if bearing_type == 'Ball' else 10.0 / 3.0
    if P_eq <= 0 or n_rpm <= 0:
        return {'L10_Mrev': float('inf'), 'L10_h': float('inf')}
    L10 = (C_dyn / P_eq) ** p
    return {'L10_Mrev': L10, 'L10_h': (L10 * 1e6) / (60.0 * n_rpm)}


# ================================================================
# 8. CARRIER / RING RIM / KEYS
# ================================================================
def carrier_arm_bending(F_pin, L, w, t, sigma_allow):
    M = F_pin * L
    Z = w * t ** 2 / 6.0
    sigma = M / Z if Z > 0 else float('inf')
    sf = sigma_allow / sigma if sigma > 0 else float('inf')
    return {'M': M, 'sigma': sigma, 'sf': sf, 'pass': sf >= 1.0}


def carrier_plate_bending(F_pin, n_planets, r_pin_circle, r_bore, t_plate, sigma_allow):
    tw = (2 * PI * r_pin_circle) / n_planets
    arm = max(r_pin_circle - r_bore, 1e-6)
    M = F_pin * arm
    Z = tw * t_plate ** 2 / 6.0
    sigma = M / Z if Z > 0 else float('inf')
    sf = sigma_allow / sigma if sigma > 0 else float('inf')
    return {'M': M, 'tributary_width': tw, 'sigma': sigma, 'sf': sf, 'pass': sf >= 1.0}


def ring_rim_strength(sigmaF_ring, m, d_root_ring, d_od_ring, sigmaF_lim):
    ht = 2.25 * m
    rim_t = (d_od_ring - d_root_ring) / 2.0
    mB = rim_t / ht if ht > 0 else 0.0
    if mB >= 1.2:
        YB = 1.0
    elif mB > 0:
        YB = 1.6 * math.log(2.242 / mB)
    else:
        YB = float('inf')
    adj = sigmaF_ring * YB
    sf = sigmaF_lim / adj if adj > 0 else float('inf')
    return {'rim_thickness': rim_t, 'ht': ht, 'mB': mB, 'YB': YB, 'sigmaF_adj': adj,
            'sf': sf, 'pass': sf >= 1.0}


def key_sizing(T_Nmm, d_shaft, w, h, l, tau_allow, sigma_allow_bearing):
    F = 2.0 * T_Nmm / d_shaft
    tau_k = F / (w * l)
    sig_b = F / (0.5 * h * l)
    sf_s = tau_allow / tau_k if tau_k > 0 else float('inf')
    sf_b = sigma_allow_bearing / sig_b if sig_b > 0 else float('inf')
    return {'F_key': F, 'tau_key': tau_k, 'sigma_bearing': sig_b, 'sf_shear': sf_s,
            'sf_bearing': sf_b, 'pass': sf_s >= 1.0 and sf_b >= 1.0}


# ================================================================
# 9. COMPONENT DIMENSIONS + HOUSING
# ================================================================
def component_dimensions(m, b, d_sun, d_planet, d_ring, d_in_shaft, d_out_shaft,
                          d_pin, max_od, pin_span, n_planets):
    sun_bore = d_in_shaft + 2.0
    sun_hub_od = max(sun_bore + 6.0, d_in_shaft * 1.6)
    sun_hub_len = max(1.2 * d_in_shaft, b)
    planet_bore = d_pin + 2.0
    planet_hub_od = max(planet_bore + 4.0, d_pin * 1.45)
    planet_hub_len = max(b, pin_span * 0.75)

    ring_root_d = d_ring + 2.5 * m
    ring_radial_rim = max(0.18 * d_ring, 8.0)
    ring_outer_d = ring_root_d + 2 * ring_radial_rim

    carrier_pitch_r = (d_sun + d_planet) / 2.0
    carrier_plate_thk = max(6.0, 0.30 * b, 0.20 * d_pin)
    carrier_od = min(max_od - 4.0, 2 * (carrier_pitch_r + 1.6 * d_pin))
    carrier_bore = d_out_shaft + 2.0
    carrier_hub_od = max(d_out_shaft + 8.0, d_out_shaft * 1.5)
    carrier_hub_len = max(1.5 * d_out_shaft, 1.5 * b)
    pin_boss_od = d_pin + 6.0
    pin_boss_len = max(6.0, 0.5 * d_pin)
    carrier_total_h = carrier_plate_thk + carrier_hub_len + 2 * pin_boss_len

    in_len = 40.0 + b + 20.0
    out_len = 40.0 + b + 20.0 + 30.0
    pin_head_d = d_pin * 1.3
    pin_head_thick = max(3.0, 0.2 * d_pin)
    pin_total_len = pin_span + 2 * pin_head_thick

    env_od = ring_outer_d + 2 * 2.0
    wall = max(3.0, 0.06 * env_od)
    housing_len = in_len + b + out_len
    flange_od = env_od + 2 * wall
    base_thick = max(5.0, 0.15 * wall)

    return {
        'Sun Gear': {'bore_d': sun_bore, 'hub_od': sun_hub_od, 'hub_length': sun_hub_len,
                     'face_width': b, 'total_height': b + sun_hub_len},
        'Planet Gear': {'bore_d': planet_bore, 'hub_od': planet_hub_od, 'hub_length': planet_hub_len,
                        'face_width': b, 'total_height': b + planet_hub_len},
        'Ring Gear': {'tooth_root_d': ring_root_d, 'outer_d': ring_outer_d,
                      'rim_thickness': ring_radial_rim, 'face_width': b + 2.0},
        'Carrier Plate': {'plate_od': carrier_od, 'output_bore': carrier_bore,
                          'hub_od': carrier_hub_od, 'hub_length': carrier_hub_len,
                          'plate_thickness': carrier_plate_thk, 'pin_circle_d': 2 * carrier_pitch_r,
                          'pin_boss_od': pin_boss_od, 'pin_boss_length': pin_boss_len,
                          'total_height': carrier_total_h},
        'Input Shaft': {'diameter': d_in_shaft, 'length': in_len},
        'Output Shaft': {'diameter': d_out_shaft, 'length': out_len},
        'Planet Pin': {'diameter': d_pin, 'span': pin_span, 'head_diameter': pin_head_d,
                       'head_thickness': pin_head_thick, 'total_length': pin_total_len},
        'Housing': {'outer_d': env_od, 'wall_thickness': wall, 'length': housing_len,
                    'flange_od': flange_od, 'base_thickness': base_thick,
                    'inner_d': env_od - 2 * wall},
    }


# ================================================================
# 10. TRUE-INVOLUTE 3D SOLIDS (Plotly Mesh3d) + explode support
# ================================================================
def _resample_closed_polygon(poly, n):
    poly = np.asarray(poly, dtype=float)
    closed = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    if cum[-1] <= 0:
        return np.repeat(poly[:1], n, axis=0)
    cum = cum / cum[-1]
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.column_stack([np.interp(t, cum, closed[:, 0]), np.interp(t, cum, closed[:, 1])])


def _extrude_disk(outer_xy, z0, z1):
    outer = np.asarray(outer_xy, dtype=float)
    n = len(outer)
    cx, cy = float(outer[:, 0].mean()), float(outer[:, 1].mean())
    vx = np.concatenate([outer[:, 0], outer[:, 0], [cx, cx]])
    vy = np.concatenate([outer[:, 1], outer[:, 1], [cy, cy]])
    vz = np.concatenate([np.full(n, z0), np.full(n, z1), [z0, z1]])
    c_bot, c_top = 2 * n, 2 * n + 1
    I, J, K = [], [], []
    for i in range(n):
        ni = (i + 1) % n
        I += [i, i, c_bot, c_top]
        J += [ni, ni + n, ni, ni + n]
        K += [ni + n, i + n, i, i + n]
    return vx, vy, vz, I, J, K


def _extrude_ring(outer_xy, inner_xy, z0, z1):
    n = max(len(outer_xy), len(inner_xy), 48)
    outer = _resample_closed_polygon(outer_xy, n)
    inner = _resample_closed_polygon(inner_xy, n)
    vx = np.concatenate([outer[:, 0], outer[:, 0], inner[:, 0], inner[:, 0]])
    vy = np.concatenate([outer[:, 1], outer[:, 1], inner[:, 1], inner[:, 1]])
    vz = np.concatenate([np.full(n, z0), np.full(n, z1), np.full(n, z0), np.full(n, z1)])
    I, J, K = [], [], []
    for i in range(n):
        ni = (i + 1) % n
        ob, ot, hb, ht = i, i + n, i + 2 * n, i + 3 * n
        nob, notp, nhb, nht = ni, ni + n, ni + 2 * n, ni + 3 * n
        I += [ob, ob, hb, hb, ob, ob, ot, ot]
        J += [nob, notp, nht, ht, hb, nhb, notp, nht]
        K += [notp, ot, nhb, nht, nhb, nob, nht, ht]
    return vx, vy, vz, I, J, K


def _circle_poly(r, n=64, offset=0.0):
    a = np.linspace(0.0, 2.0 * PI, n, endpoint=False) + offset
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def _involute_point(r, r_base):
    if r <= r_base:
        return np.array([r_base, 0.0])
    th = math.sqrt(max((r / r_base) ** 2 - 1.0, 0.0))
    return np.array([r_base * (math.cos(th) + th * math.sin(th)),
                     r_base * (math.sin(th) - th * math.cos(th))])


def _single_tooth_polygon(z, m, alpha_deg=20.0, ppf=10):
    alpha = math.radians(alpha_deg)
    r_pitch = z * m / 2.0
    r_base = r_pitch * math.cos(alpha)
    r_tip, r_root = r_pitch + m, r_pitch - 1.25 * m
    r_start = max(r_base, r_root)
    rs = np.linspace(r_start, r_tip, ppf)
    right = np.array([_involute_point(r, r_base) for r in rs])
    left = np.array([[-x, y] for x, y in right[::-1]])
    half_root = (PI / (2.0 * z)) * 1.35
    root_arc = np.array([[r_root * math.cos(-half_root + 2 * half_root * k / 5.0),
                           r_root * math.sin(-half_root + 2 * half_root * k / 5.0)] for k in range(6)])
    pts = [root_arc[0]] + right.tolist() + left.tolist() + [root_arc[-1]] + root_arc[::-1][1:].tolist()
    return np.array(pts)


def _gear_outline(z, m, alpha_deg=20.0):
    tooth = _single_tooth_polygon(z, m, alpha_deg)
    period = 2.0 * PI / z
    out = []
    for i in range(z):
        a = i * period
        c, s = math.cos(a), math.sin(a)
        out.append(np.column_stack([tooth[:, 0] * c - tooth[:, 1] * s, tooth[:, 0] * s + tooth[:, 1] * c]))
    return np.vstack(out)


def _mesh_trace(mesh, color, name, opacity=1.0):
    vx, vy, vz, I, J, K = mesh
    return go.Mesh3d(x=vx, y=vy, z=vz, i=I, j=J, k=K, color=color, opacity=opacity, name=name,
                      flatshading=False,
                      lighting=dict(ambient=0.40, diffuse=0.85, specular=0.75, roughness=0.30, fresnel=0.15),
                      lightposition=dict(x=300, y=200, z=400))


def _gear_mesh(z, m, alpha_deg, face_width, bore_dia=None):
    outline = _gear_outline(z, m, alpha_deg)
    if bore_dia:
        return _extrude_ring(outline, _circle_poly(bore_dia / 2.0, 64), 0.0, face_width)
    return _extrude_disk(outline, 0.0, face_width)


def _ring_gear_mesh(z, m, alpha_deg, face_width, outer_dia):
    r_root = z * m / 2.0 + 1.25 * m
    return _extrude_ring(_circle_poly(outer_dia / 2.0, 96), _circle_poly(r_root, 96), 0.0, face_width)


def _carrier_mesh(pitch_r, d_pin, plate_thk, hub_od, bore_dia):
    plate_r = pitch_r + d_pin * 1.5
    outer = _circle_poly(plate_r, 96)
    if bore_dia:
        return _extrude_ring(outer, _circle_poly(bore_dia / 2.0, 96), 0.0, plate_thk)
    return _extrude_disk(outer, 0.0, plate_thk)


def _shaft_mesh(diameter, length, offset_z=0.0):
    return _extrude_disk(_circle_poly(diameter / 2.0, 48), offset_z, offset_z + length)


def create_assembly_3d(S, P, R, m, n_planets, d_pin, face_width, d_in_shaft, d_out_shaft,
                        sun_geom, planet_geom, comp, explode=0.0):
    pitch_r = (sun_geom['d_pitch'] + planet_geom['d_pitch']) / 2.0
    traces = []
    gap = face_width * 1.6 * explode

    traces.append(_mesh_trace(_gear_mesh(S, m, 20.0, face_width, comp['Sun Gear']['bore_d']), '#E85D2A', 'Sun Gear'))
    for k in range(n_planets):
        ang = k * 2.0 * PI / n_planets
        c, s = math.cos(ang), math.sin(ang)
        vx0, vy0, vz0, I0, J0, K0 = _gear_mesh(P, m, 20.0, face_width, comp['Planet Gear']['bore_d'])
        vx = vx0 * c - vy0 * s + pitch_r * c
        vy = vx0 * s + vy0 * c + pitch_r * s
        traces.append(_mesh_trace((vx, vy, vz0 + gap, I0, J0, K0), '#F2B01E', f'Planet {k + 1}'))

    rmesh = _ring_gear_mesh(R, m, 20.0, face_width + 2.0, comp['Ring Gear']['outer_d'])
    rvx, rvy, rvz, rI, rJ, rK = rmesh
    traces.append(_mesh_trace((rvx, rvy, rvz + 2.0 * gap, rI, rJ, rK), '#9AA0A6', 'Ring Gear', 0.45))

    cvx, cvy, cvz, cI, cJ, cK = _carrier_mesh(pitch_r, d_pin, comp['Carrier Plate']['plate_thickness'],
                                               comp['Carrier Plate']['hub_od'], comp['Carrier Plate']['output_bore'])
    traces.append(_mesh_trace((cvx, cvy, cvz + face_width + 3.0 * gap, cI, cJ, cK), '#3F6FB5', 'Carrier', 0.60))

    traces.append(_mesh_trace(_shaft_mesh(d_in_shaft, 40.0, offset_z=-45.0 - gap), '#B0B0B0', 'Input Shaft'))
    traces.append(_mesh_trace(_shaft_mesh(d_out_shaft, 40.0 + face_width,
                                           offset_z=face_width + comp['Carrier Plate']['plate_thickness'] + 4.0 * gap),
                               '#B0B0B0', 'Output Shaft'))
    for k in range(n_planets):
        ang = k * 2.0 * PI / n_planets
        c, s = math.cos(ang), math.sin(ang)
        pvx, pvy, pvz, pI, pJ, pK = _shaft_mesh(d_pin, comp['Planet Pin']['total_length'], offset_z=-2.0 + gap)
        traces.append(_mesh_trace((pvx + pitch_r * c, pvy + pitch_r * s, pvz, pI, pJ, pK), '#2E2E2E', f'Pin {k + 1}'))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(xaxis=dict(title='X (mm)', backgroundcolor='#0e1117', gridcolor='#2a2f3a', showbackground=True),
                   yaxis=dict(title='Y (mm)', backgroundcolor='#0e1117', gridcolor='#2a2f3a', showbackground=True),
                   zaxis=dict(title='Z (mm)', backgroundcolor='#0e1117', gridcolor='#2a2f3a', showbackground=True),
                   aspectmode='data', camera=dict(eye=dict(x=1.6, y=1.4, z=1.0))),
        paper_bgcolor='#0e1117', font=dict(color='#e6e6e6'),
        title=dict(text='3D Planetary Gearbox - True-Involute Assembly', font=dict(size=17, color='#e6e6e6')),
        height=720, margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(bgcolor='rgba(20,20,20,0.7)', bordercolor='#444', borderwidth=1),
    )
    return fig


def create_component_3d(kind, p):
    fig = go.Figure()
    if kind == 'Sun Gear':
        fig.add_trace(_mesh_trace(_gear_mesh(p['z'], p['m'], 20.0, p['b'], p['bore']), '#E85D2A', kind))
        title = f"Sun Gear - z={p['z']}, m={p['m']:.2f}mm"
    elif kind == 'Planet Gear':
        fig.add_trace(_mesh_trace(_gear_mesh(p['z'], p['m'], 20.0, p['b'], p['bore']), '#F2B01E', kind))
        title = f"Planet Gear - z={p['z']}, m={p['m']:.2f}mm"
    elif kind == 'Ring Gear':
        fig.add_trace(_mesh_trace(_ring_gear_mesh(p['z'], p['m'], 20.0, p['b'], p['outer_d']), '#9AA0A6', kind, 0.55))
        title = f"Ring Gear - z={p['z']}, m={p['m']:.2f}mm"
    elif kind == 'Carrier':
        fig.add_trace(_mesh_trace(_carrier_mesh(p['pitch_r'], p['d_pin'], p['thk'], p['hub_od'], p['bore']),
                                   '#3F6FB5', kind, 0.70))
        title = "Carrier Plate"
    elif kind in ('Input Shaft', 'Output Shaft', 'Planet Pin'):
        color = '#B0B0B0' if 'Shaft' in kind else '#2E2E2E'
        fig.add_trace(_mesh_trace(_shaft_mesh(p['d'], p['length']), color, kind))
        title = f"{kind} - dia {p['d']:.1f}mm"
    else:
        title = kind
    fig.update_layout(
        scene=dict(xaxis=dict(title='X (mm)', backgroundcolor='#0e1117', gridcolor='#2a2f3a', showbackground=True),
                   yaxis=dict(title='Y (mm)', backgroundcolor='#0e1117', gridcolor='#2a2f3a', showbackground=True),
                   zaxis=dict(title='Z (mm)', backgroundcolor='#0e1117', gridcolor='#2a2f3a', showbackground=True),
                   aspectmode='data', camera=dict(eye=dict(x=1.5, y=1.3, z=0.9))),
        paper_bgcolor='#0e1117', font=dict(color='#e6e6e6'),
        title=dict(text=title, font=dict(size=15, color='#e6e6e6')),
        height=460, margin=dict(l=0, r=0, t=40, b=0))
    return fig


# ================================================================
# 11. OPENSCAD PARAMETRIC EXPORT
# ================================================================
def generate_openscad(S, P, R, m, n_planets, d_pin, b, sun_hub_od, sun_bore,
                       planet_hub_od, planet_bore, ring_outer_d, ring_face_width,
                       pitch_r, plate_thk, carrier_hub_od, carrier_bore,
                       din, dout, pin_span, housing_od, housing_len, show="assembly"):
    sun_tip, sun_root = S * m + 2 * m, S * m - 2.5 * m
    p_tip, p_root = P * m + 2 * m, P * m - 2.5 * m
    r_tip, r_root = R * m - 2 * m, R * m + 2.5 * m
    carrier_od = 2 * (pitch_r + d_pin * 1.5)
    in_len, out_len = 40 + b + 20, 40 + b + 20 + 30

    return f"""// =====================================================================
// PLANETARY GEARBOX - OPENSCAD PARAMETRIC MODEL
// =====================================================================
// Generated by Planetary Gearbox Designer v5
// S={S} P={P} R={R} m={m}mm  ratio ~ {(S + R) / S:.4f}  planets={n_planets}
//
// NOTE: Trapezoidal tooth approximation (fast preview / roughing).
//       For true involute profiles use the 'gears.scad' library.
// =====================================================================

PI = 3.14159265358979;
$fn = 32;

// ---- Primary parameters ----
zs        = {S};
zp        = {P};
zr        = {R};
m         = {m};
n_planets = {n_planets};
b         = {b};
d_pin     = {d_pin};
pin_span  = {pin_span};
pitch_r   = {pitch_r:.4f};

// ---- Derived diameters ----
sun_tip_d    = {sun_tip:.3f};
sun_root_d   = {sun_root:.3f};
planet_tip_d = {p_tip:.3f};
planet_root_d= {p_root:.3f};
ring_tip_d   = {r_tip:.3f};
ring_root_d  = {r_root:.3f};
carrier_od   = {carrier_od:.3f};

// =====================================================================
// MODULES
// =====================================================================
module tooth_wedge(r_in, r_out, w_in, w_out, depth) {{
    linear_extrude(height = depth, center = true)
        polygon(points = [
            [r_in,  -w_in/2],
            [r_in,   w_in/2],
            [r_out,  w_out/2],
            [r_out, -w_out/2]
        ]);
}}

module external_gear(z, m, tip_d, root_d, width) {{
    pitch_r = z * m / 2;
    tw      = (PI * pitch_r / z) * 0.9;
    overlap = m * 0.5;
    union() {{
        cylinder(h = width, d = root_d, center = true);
        for (i = [0 : z - 1])
            rotate([0, 0, i * 360 / z])
                tooth_wedge(root_d/2 - overlap, tip_d/2, tw, tw * 0.55, width);
    }}
}}

module ring_gear(z, m, tip_d, root_d, outer_d, width) {{
    pitch_r = z * m / 2;
    gt      = (PI * pitch_r / z) * 1.05;
    gr      = gt * 1.6;
    overlap = m * 0.5;
    difference() {{
        cylinder(h = width, d = outer_d, center = true);
        cylinder(h = width + 2, d = tip_d, center = true);
        for (i = [0 : z - 1])
            rotate([0, 0, i * 360 / z + (180 / z)])
                tooth_wedge(tip_d/2 - overlap, root_d/2 + overlap, gt, gr, width + 4);
    }}
}}

module sun_gear() {{
    difference() {{
        union() {{
            external_gear(zs, m, sun_tip_d, sun_root_d, b);
            cylinder(h = b, d = {sun_hub_od:.3f}, center = true);
        }}
        cylinder(h = b + 10, d = {sun_bore:.3f}, center = true);
    }}
}}

module planet_gear() {{
    difference() {{
        union() {{
            external_gear(zp, m, planet_tip_d, planet_root_d, b);
            cylinder(h = b, d = {planet_hub_od:.3f}, center = true);
        }}
        cylinder(h = b + 10, d = {planet_bore:.3f}, center = true);
    }}
}}

module ring_gear_part() {{
    ring_gear(zr, m, ring_tip_d, ring_root_d, {ring_outer_d:.3f}, {ring_face_width:.3f});
}}

module carrier() {{
    difference() {{
        union() {{
            cylinder(h = {plate_thk:.3f}, d = carrier_od, center = true);
            cylinder(h = b + 4, d = {carrier_hub_od:.3f}, center = true);
            for (i = [0 : n_planets - 1])
                rotate([0, 0, i * (360 / n_planets)])
                    translate([pitch_r, 0, 0])
                        cylinder(h = {plate_thk:.3f} + 10, d = d_pin + 6, center = true);
        }}
        cylinder(h = b + 10, d = {carrier_bore:.3f}, center = true);
        for (i = [0 : n_planets - 1])
            rotate([0, 0, i * (360 / n_planets)])
                translate([pitch_r, 0, 0])
                    cylinder(h = {plate_thk:.3f} + 20, d = d_pin + 1.0, center = true);
    }}
}}

module input_shaft()  {{ cylinder(h = {in_len:.1f},  d = {din:.3f},  center = true); }}
module output_shaft() {{ cylinder(h = {out_len:.1f}, d = {dout:.3f}, center = true); }}
module planet_pin_part() {{ cylinder(h = pin_span, d = d_pin, center = true); }}

module housing() {{
    difference() {{
        union() {{
            cylinder(h = {housing_len:.1f}, d = {housing_od:.3f}, center = true);
            cylinder(h = 6, d = {housing_od:.3f} + 16, center = true);   // end flange
        }}
        cylinder(h = {housing_len:.1f} + 2, d = {housing_od:.3f} - 6, center = true);
    }}
}}

// =====================================================================
// ASSEMBLY
// =====================================================================
module assembly(exploded = false) {{
    gap = exploded ? b * 1.5 : 0.4;

    color("DarkOrange") sun_gear();

    color("DimGray")
        translate([0, 0, -(({in_len:.1f})/2 + b/2 + gap)])
            input_shaft();

    for (i = [0 : n_planets - 1])
        rotate([0, 0, i * (360 / n_planets)])
            translate([pitch_r, 0, 0]) {{
                color("Gold") planet_gear();
                color("DarkSlateGray")
                    translate([0, 0, exploded ? gap : 0])
                        planet_pin_part();
            }}

    color("SlateGray", 0.55)
        translate([0, 0, exploded ? gap * 2 : gap])
            ring_gear_part();

    color("SteelBlue", 0.85)
        translate([0, 0, b/2 + {plate_thk:.3f}/2 + (exploded ? gap * 1.5 : gap)])
            carrier();

    color("DimGray")
        translate([0, 0, b/2 + {plate_thk:.3f} + ({out_len:.1f})/2 + (exploded ? gap * 2.5 : gap)])
            output_shaft();
}}

// =====================================================================
// VIEW SELECTOR
// =====================================================================
SHOW = "{show}";
if      (SHOW == "assembly")     assembly(false);
else if (SHOW == "exploded")     assembly(true);
else if (SHOW == "sun")          sun_gear();
else if (SHOW == "planet")       planet_gear();
else if (SHOW == "ring")         ring_gear_part();
else if (SHOW == "carrier")      carrier();
else if (SHOW == "input_shaft")  input_shaft();
else if (SHOW == "output_shaft") output_shaft();
else if (SHOW == "planet_pin")   planet_pin_part();
else if (SHOW == "housing")      housing();
"""


# ================================================================
# 12. STREAMLIT APP
# ================================================================
st.set_page_config(page_title="Planetary Gearbox Designer", page_icon="⚙️", layout="wide",
                    initial_sidebar_state="expanded")
st.markdown("""
<style>
.main-header{font-size:2.0rem;font-weight:800;color:#1f4e78;text-align:center;margin-bottom:.2rem;}
.sub-header{font-size:.9rem;color:#666;text-align:center;margin-bottom:1rem;}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="main-header">⚙️ Planetary Gearbox Designer — Ultimate Edition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">3-planet single-stage epicyclic gearbox — full stress, deflection, '
            'bearing-life, 3D CAD & OpenSCAD suite</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("1. Ratio / Envelope / Config")
    fixed_case = st.selectbox("Fixed Member:", ['Ring Fixed', 'Sun Fixed', 'Carrier Fixed'])
    target_ratio = st.number_input("Target Ratio (1:x):", 2.0, 20.0, TARGET_RATIO_DEFAULT, 0.5)
    max_od = st.number_input("Max Ring OD (mm):", 50.0, 800.0, MAX_OD_DEFAULT_MM, 5.0)
    efficiency = st.number_input("Mesh Efficiency:", 0.80, 0.99, EFFICIENCY_DEFAULT, 0.01)
    st.caption(f"Planets: **{N_PLANETS}** (fixed)")

    st.header("2. Teeth & Module")
    manual_teeth = st.checkbox("Manually set Sun/Planet teeth & module", value=False)
    if manual_teeth:
        S_manual = st.number_input("Sun teeth (S):", 15, 120, 20, 1)
        P_manual = st.number_input("Planet teeth (P):", 15, 150, 32, 1)
        m_manual = st.selectbox("Module (mm):", MODULE_LIST, index=2)
    else:
        st.caption("Sun/Planet auto-searched for closest ratio match; Ring = Sun + 2·Planet.")

    st.header("3. Operating Conditions")
    t_in_max = st.number_input("Max Input Torque available (N·m):", RATED_T_IN_NM_MIN, 500.0, 20.0, 0.5)
    t_in_nm = st.slider("Input Torque (N·m):", RATED_T_IN_NM_MIN, t_in_max, 5.0, 0.1)
    n_in_max = st.number_input("Max Input Speed available (rpm):", RATED_N_IN_RPM_MIN, 20000.0, 3000.0, 50.0)
    n_in_rpm = st.slider("Input Speed (rpm):", RATED_N_IN_RPM_MIN, n_in_max, 1500.0, 10.0)
    st.caption("Motor power is always a **computed result**, never a direct input.")

    st.header("4. Design (worst-case) Torque — 65–75 N·m")
    design_out_tq = st.slider("Design Output Torque (N·m):", DESIGN_TQ_MIN, DESIGN_TQ_MAX, 70.0, 0.5)

    st.header("5. Material")
    selected_mat = st.selectbox("Gear / Shaft / Pin / Carrier Material:", list(MATERIALS.keys()))
    mat = dict(MATERIALS[selected_mat])
    if selected_mat == 'Custom':
        with st.expander("Custom properties", expanded=True):
            mat['E'] = st.number_input("E (MPa):", value=mat['E'])
            mat['nu'] = st.number_input("nu:", value=mat['nu'], step=0.01)
            mat['tau'] = st.number_input("tau (MPa):", value=mat['tau'])
            mat['sigmaF'] = st.number_input("sigmaF limit (MPa):", value=mat['sigmaF'])
            mat['sigmaH'] = st.number_input("sigmaH limit (MPa):", value=mat['sigmaH'])
            mat['sigma_allow_bend'] = st.number_input("Allow. bend (MPa):", value=mat['sigma_allow_bend'])
            mat['sigma_allow_bearing'] = st.number_input("Allow. bearing/crush (MPa):", value=mat['sigma_allow_bearing'])

    st.header("6. Design Factors")
    Kb = st.number_input("ASME Kb:", 1.0, 3.0, 1.5, 0.1)
    Kt = st.number_input("ASME Kt:", 1.0, 2.0, 1.0, 0.1)
    Kw = st.number_input("Keyway Kw:", 1.0, 2.0, 1.3, 0.05)
    Kp = st.number_input("Planet load-sharing Kp:", 1.00, 1.30, 1.05, 0.01)
    KA = st.number_input("Application factor KA:", 1.0, 2.0, 1.25, 0.05)
    accuracy_grade = st.slider("Gear accuracy grade (ISO 1328):", 5, 8, 6)
    theta_deg = st.slider("Force angle theta on planet (deg):", 60.0, 180.0, 120.0, 1.0)
    desired_life_hours = st.number_input("Desired duty life (hours):", 100, 100000, 10000, 500)

    st.header("7. Main Bearings (Sun/Ring shaft)")
    sf_bearing = st.number_input("Bearing service factor:", 1.0, 3.0, 1.5, 0.1)
    cdyn = st.number_input("Main bearing Cdyn (N):", 500.0, 100000.0, 12000.0, 500.0)
    main_bearing_type = st.selectbox("Main bearing type:", ['Ball', 'Roller'])

    st.header("8. Planet Pin / Bearing")
    pin_span_mm = st.number_input("Pin support span (mm):", 5.0, 200.0, 30.0, 1.0)
    pin_support = st.selectbox("Pin support type:", ['Needle Roller Bearing', 'Plain Bronze Bush'])
    if pin_support == 'Needle Roller Bearing':
        allow_bearing_pressure = st.number_input("Allow. dynamic pressure (MPa):", value=25.0)
        cdyn_planet = st.number_input("Planet bearing Cdyn (N):", value=6000.0, step=250.0)
    else:
        allow_bearing_pressure = st.number_input("Allow. static bush pressure (MPa):", value=10.0)
        cdyn_planet = None

    st.header("9. Carrier Geometry")
    arm_length_mm = st.number_input("Carrier arm length (mm):", 5.0, 200.0, 25.0, 1.0)
    arm_width_mm = st.number_input("Carrier arm width (mm):", 5.0, 100.0, 18.0, 1.0)
    arm_thickness_mm = st.number_input("Carrier arm thickness (mm):", 3.0, 60.0, 10.0, 1.0)
    plate_thickness_mm = st.number_input("Carrier plate thickness (mm):", 3.0, 60.0, 12.0, 1.0)
    plate_bore_margin_mm = st.number_input("Plate bore margin over shaft dia (mm):", 0.0, 20.0, 2.0, 0.5)

    st.header("10. Keys / Splines")
    key_width_mm = st.number_input("Key width w (mm):", 2.0, 40.0, 6.0, 0.5)
    key_height_mm = st.number_input("Key height h (mm):", 2.0, 40.0, 6.0, 0.5)
    key_length_mm = st.number_input("Key length l (mm):", 5.0, 150.0, 20.0, 1.0)

    st.header("11. Shaft Lengths & Twist Limit")
    shaft_len_in_mm = st.number_input("Input shaft length (mm):", 10.0, 500.0, 60.0, 5.0)
    shaft_len_out_mm = st.number_input("Output shaft length (mm):", 10.0, 500.0, 60.0, 5.0)
    twist_limit_deg = st.number_input("Max allowed shaft twist (deg):", 0.05, 5.0, 0.5, 0.05,
                                      help="0.5 deg suits precise motion control. General power "
                                           "transmission can often allow 1-2 deg.")
    st.header("12. Face Width")
    face_width_factor = st.number_input("Face width factor (b = k·m):", 8.0, 24.0, FACE_WIDTH_FACTOR_DEFAULT, 1.0)

# ================================================================
# CALCULATIONS
# ================================================================
if manual_teeth:
    R, ratio_actual, assembly_pass, clearance_pass, ratio_err = evaluate_manual_teeth(
        fixed_case, S_manual, P_manual, N_PLANETS, target_ratio)
    S, P, m_use, found = S_manual, P_manual, m_manual, True
else:
    S, P, R, ratio_actual, found = find_teeth_combo(fixed_case, target_ratio, N_PLANETS)
    assembly_pass = (S + R) % N_PLANETS == 0
    clearance_pass = (S + P) * math.sin(math.radians(180 / N_PLANETS)) > (P + 2)
    ratio_err = abs(ratio_actual - target_ratio)
    m_use = MODULE_LIST[0]
    for mm in MODULE_LIST:
        if (R + 2.5) * mm + 12.0 <= max_od:
            m_use = mm

if not found or S <= 0:
    st.error("No valid tooth combination found. Adjust ratio / envelope / manual teeth.")
    st.stop()

est_od = (R + 2.5) * m_use + 12.0
od_fits = est_od <= max_od

geom = gear_geometry(S, P, R, m_use, PRESSURE_ANGLE_DEFAULT, HELIX_ANGLE_DEFAULT)
d_sun, d_planet, d_ring = geom['sun']['d_pitch'], geom['planet']['d_pitch'], geom['ring']['d_pitch']
face_width = face_width_factor * m_use

kin = compute_kinematics(fixed_case, S, P, n_in_rpm, ratio_actual)
input_member, output_member = kin['input_member'], kin['output_member']
output_speed = kin['output_speed']

motor_power_operating_w = (2.0 * PI * n_in_rpm / 60.0) * t_in_nm
Tin_design_Nm = design_out_tq / (ratio_actual * efficiency)
motor_power_design_w = (2.0 * PI * n_in_rpm / 60.0) * Tin_design_Nm
T_in_design_Nmm = Tin_design_Nm * 1000.0
T_out_design_Nmm = design_out_tq * 1000.0

# --- shafts ---
# Strength-based diameters (ASME)
d_in_strength, Te_in = shaft_diameter_asme(T_in_design_Nmm, 0.0, mat['tau'], Kb, Kt, Kw)
d_out_strength, Te_out = shaft_diameter_asme(T_out_design_Nmm, 0.0, mat['tau'], Kb, Kt, Kw)

# Stiffness-based diameters (twist <= 0.5 deg)
TWIST_LIMIT_DEG = 0.5
d_in_twist = shaft_diameter_for_twist(T_in_design_Nmm, shaft_len_in_mm, mat['E'], mat['nu'], TWIST_LIMIT_DEG)
d_out_twist = shaft_diameter_for_twist(T_out_design_Nmm, shaft_len_out_mm, mat['E'], mat['nu'], TWIST_LIMIT_DEG)

# Final diameter = larger of the two, rounded UP to the next 0.5 mm
d_shaft_in = math.ceil(max(d_in_strength, d_in_twist) * 2.0) / 2.0
d_shaft_out = math.ceil(max(d_out_strength, d_out_twist) * 2.0) / 2.0
defl_in = torsional_deflection_deg(T_in_design_Nmm, shaft_len_in_mm, mat['E'], mat['nu'], d_shaft_in)
defl_out = torsional_deflection_deg(T_out_design_Nmm, shaft_len_out_mm, mat['E'], mat['nu'], d_shaft_out)
defl_in_ok, defl_out_ok = defl_in['theta_deg'] <= 0.5, defl_out['theta_deg'] <= 0.5

# --- ISO 6336-lite stress ---
alpha_t_deg = geom['alpha_t_deg']
YF_S, YS_S = tooth_form_factors(S)
YF_P, YS_P = tooth_form_factors(P)
ZH = zone_factor(alpha_t_deg, HELIX_ANGLE_DEFAULT)
pitch_line_v = (PI * d_sun / 1000.0 * n_in_rpm) / 60.0
KV = dynamic_factor_Kv(pitch_line_v, S, accuracy_grade)
eps_sp = contact_ratio_external(S, P, m_use, PRESSURE_ANGLE_DEFAULT)
eps_rp = contact_ratio_internal(R, P, m_use, PRESSURE_ANGLE_DEFAULT)
KHbeta = load_distribution_K_Hbeta(face_width, d_sun, accuracy_grade)
KFbeta = KHbeta
KHalpha = load_distribution_K_Halpha(eps_sp)
KFalpha = KHalpha

N_cycles = desired_life_hours * 60 * n_in_rpm
ZN = life_factor(N_cycles, 'contact')
YN = life_factor(N_cycles, 'bending')
sigmaH_allow = mat['sigmaH'] * ZN
sigmaF_allow = mat['sigmaF'] * YN

stress_params = dict(S=S, P=P, R=R, m=m_use, alpha_n=PRESSURE_ANGLE_DEFAULT, beta=HELIX_ANGLE_DEFAULT, b=face_width,
                      E1=mat['E'], E2=mat['E'], nu1=mat['nu'], nu2=mat['nu'], TS=T_in_design_Nmm,
                      KA=KA, KV=KV, KFbeta=KFbeta, KFalpha=KFalpha, KHbeta=KHbeta, KHalpha=KHalpha, Kp=Kp,
                      YFa={'S': YF_S, 'P': YF_P}, YSa={'S': YS_S, 'P': YS_P},
                      Yeps=0.85, Ybeta=1.0, ZH=ZH, Zeps=0.9, Zbeta=1.0, theta_deg=theta_deg, n_planets=N_PLANETS)
stress = planetary_gear_stress(stress_params)

sfF_sp = sigmaF_allow / max(stress['sigmaF_SP'], 1e-9)
sfF_rp = sigmaF_allow / max(stress['sigmaF_RP'], 1e-9)
sfF_planet = sigmaF_allow / max(stress['sigmaF_planet'], 1e-9)
sfH_sp = sigmaH_allow / max(stress['sigmaH_SP'], 1e-9)
sfH_rp = sigmaH_allow / max(stress['sigmaH_RP'], 1e-9)

# --- planet pin ---
pin = design_planet_pin(stress['F_pin'], pin_span_mm, face_width, mat['sigma_allow_bend'], mat['tau'],
                         allow_bearing_pressure)
if pin_support == 'Needle Roller Bearing' and cdyn_planet:
    planet_bearing_life = bearing_L10_life(cdyn_planet, stress['F_pin'], max(kin['n_planet_rel_carrier'], 1e-6), 'Roller')
else:
    planet_bearing_life = None

# --- main bearing ---
d_mesh = d_sun if input_member == 'Sun' else d_ring
main_bearing_radial = math.hypot(stress['Fr_SP'], stress['Ft_SP']) * sf_bearing
main_bearing_speed = n_in_rpm if input_member != 'Carrier' else kin['n_carrier']
main_bearing_pass = main_bearing_radial <= cdyn
main_bearing_life = bearing_L10_life(cdyn, main_bearing_radial / max(sf_bearing, 1e-9),
                                      max(main_bearing_speed, 1e-6), main_bearing_type)

# --- carrier ---
r_pin_circle = d_sun / 2 + d_planet / 2
r_bore_carrier = d_shaft_out / 2.0 + plate_bore_margin_mm
carrier_arm_res = carrier_arm_bending(stress['F_pin'], arm_length_mm, arm_width_mm, arm_thickness_mm,
                                       mat['sigma_allow_bend'])
carrier_plate_res = carrier_plate_bending(stress['F_pin'], N_PLANETS, r_pin_circle, r_bore_carrier,
                                           plate_thickness_mm, mat['sigma_allow_bend'])

# --- ring rim ---
ring_rim_res = ring_rim_strength(stress['sigmaF_RP'], m_use, geom['ring']['d_root'], est_od, mat['sigmaF'])

# --- keys ---
key_in_res = key_sizing(T_in_design_Nmm, d_shaft_in, key_width_mm, key_height_mm, key_length_mm, mat['tau'],
                         mat['sigma_allow_bearing'])
key_out_res = key_sizing(T_out_design_Nmm, d_shaft_out, key_width_mm, key_height_mm, key_length_mm, mat['tau'],
                          mat['sigma_allow_bearing'])

# --- component dims ---
comp = component_dimensions(m_use, face_width, d_sun, d_planet, d_ring, d_shaft_in, d_shaft_out,
                             pin['d_pin'], max_od, pin_span_mm, N_PLANETS)
comp['Carrier Plate']['plate_thickness'] = plate_thickness_mm

# --- overall checks ---
overall_checks = {
    'Ratio matches target': ratio_err < 0.05,
    'Assembly condition': assembly_pass,
    'Clearance condition': clearance_pass,
    'Ring OD fits envelope': od_fits,
    'Sun/Planet bending SF>=1': sfF_sp >= 1,
    'Ring/Planet bending SF>=1': sfF_rp >= 1,
    'Combined planet bending SF>=1': sfF_planet >= 1,
    'Sun/Planet contact SF>=1': sfH_sp >= 1,
    'Ring/Planet contact SF>=1': sfH_rp >= 1,
    'Planet pin pressure OK': pin['pressure_ok'],
    'Main bearing capacity OK': main_bearing_pass,
    'Carrier arm bending OK': carrier_arm_res['pass'],
    'Carrier plate bending OK': carrier_plate_res['pass'],
    'Ring rim strength OK': ring_rim_res['pass'],
    'Input key OK': key_in_res['pass'],
    'Output key OK': key_out_res['pass'],
    'Input shaft twist <=0.5deg': defl_in_ok,
    'Output shaft twist <=0.5deg': defl_out_ok,
}
overall_ok = all(overall_checks.values())
fails = [k for k, v in overall_checks.items() if not v]

# ================================================================
# TOP METRICS
# ================================================================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Achieved Ratio", f"1:{ratio_actual:.3f}")
c2.metric("Motor Power (design pt)", f"{motor_power_design_w:.1f} W")
c3.metric("Motor Power (operating)", f"{motor_power_operating_w:.1f} W")
c4.metric("Ring OD (est.)", f"{est_od:.1f} mm")
c5.metric("Overall Status", "PASS ✅" if overall_ok else "CHECK REQUIRED ⚠️")
if fails:
    st.warning("Failing checks: " + ", ".join(fails))
st.info(f"**Teeth:** S={S} · P={P} · R={R}  |  **Module:** {m_use:.2f} mm  |  "
        f"**Face width:** {face_width:.1f} mm  |  **{input_member} = input, {output_member} = output**")

# ================================================================
# TABS
# ================================================================
tabs = st.tabs([
    "1 · Gear Geometry", "2 · Tooth Synthesis", "3 · Kinematics", "4 · Shafts & Pins", "5 · Bearings",
    "6 · Planet Load Sharing", "7 · Planet Pin", "8 · Carrier", "9 · Ring Rim", "10 · Keys/Splines",
    "11 · Component Dimensions", "12 · Overall Design Results", "13 · Parameter Glossary",
    "14 · 3D Visualization", "15 · OpenSCAD",
])

with tabs[0]:
    st.subheader("Gear Geometry")
    geo_df = pd.DataFrame([
        ['Sun', S, geom['sun']['d_pitch'], geom['sun']['d_base'], geom['sun']['d_tip'], geom['sun']['d_root'], face_width],
        ['Planet', P, geom['planet']['d_pitch'], geom['planet']['d_base'], geom['planet']['d_tip'], geom['planet']['d_root'], face_width],
        ['Ring', R, geom['ring']['d_pitch'], geom['ring']['d_base'], geom['ring']['d_tip'], geom['ring']['d_root'], face_width + 2.0],
    ], columns=['Gear', 'Teeth', 'Pitch (mm)', 'Base (mm)', 'Tip (mm)', 'Root (mm)', 'Face Width (mm)'])
    st.dataframe(geo_df, hide_index=True, use_container_width=True)
    st.write(f"Working transverse pressure angle: {alpha_t_deg:.2f}°  |  Circular pitch: {geom['circular_pitch']:.2f} mm")
    st.write(f"Centre distance (Sun–Planet): {geom['center_dist_sun_planet']:.2f} mm  |  "
             f"Centre distance (Ring–Planet): {geom['center_dist_ring_planet']:.2f} mm")
    st.write(f"Contact ratio: Sun-Planet e={eps_sp:.2f}  |  Ring-Planet e={eps_rp:.2f}")

with tabs[1]:
    st.subheader("Tooth Synthesis")
    st.dataframe(pd.DataFrame([
        ['Sun teeth', S, 'Search range 15–90 / manual'],
        ['Planet teeth', P, 'Search range 15–90 / manual'],
        ['Ring teeth', R, 'R = S + 2P (coaxiality)'],
        ['Achieved ratio', f"{ratio_actual:.4f}", f"target {target_ratio:.3f} (err {ratio_err:.4f})"],
        ['Assembly condition', 'PASS' if assembly_pass else 'FAIL', f'(S+R) mod {N_PLANETS} = {(S + R) % N_PLANETS}'],
        ['Clearance condition', 'PASS' if clearance_pass else 'FAIL', 'planet tip spacing'],
        ['Module', f"{m_use:.2f} mm", f'OD est. {est_od:.1f} <= {max_od:.0f} mm -> {"PASS" if od_fits else "FAIL"}'],
        ['Selection mode', 'Manual' if manual_teeth else 'Auto-search', '-'],
    ], columns=['Item', 'Value', 'Note']), hide_index=True, use_container_width=True)
    cand = suggest_tooth_sets(fixed_case, target_ratio, N_PLANETS, max_od)
    if not cand.empty:
        st.markdown("**Other exact-ratio candidates within the OD limit**")
        st.dataframe(cand.sort_values(['module', 'S']).head(25), hide_index=True, use_container_width=True)
    else:
        st.caption("No alternative exact-ratio candidates found within this envelope.")

with tabs[2]:
    st.subheader("Kinematics")
    st.dataframe(pd.DataFrame({
        'Parameter': ['Configuration', 'Input member', 'Output member', 'Achieved ratio', 'Target ratio',
                      'Input speed (rpm)', 'Output speed (rpm)', 'Sun speed (rpm)', 'Ring speed (rpm)',
                      'Carrier speed (rpm)', 'Planet spin rel. carrier (rpm)'],
        'Value': [fixed_case, input_member, output_member, f"{ratio_actual:.3f}", f"{target_ratio:.2f}",
                  f"{n_in_rpm:.1f}", f"{output_speed:.2f}", f"{kin['n_sun']:.2f}", f"{kin['n_ring']:.2f}",
                  f"{kin['n_carrier']:.2f}", f"{kin['n_planet_rel_carrier']:.2f}"]
    }), hide_index=True, use_container_width=True)
    st.caption(f"Motor power: operating {motor_power_operating_w:.1f} W · design {motor_power_design_w:.1f} W  |  "
               f"Nominal output torque {t_in_nm * ratio_actual * efficiency:.2f} N·m  |  "
               f"Design output torque {design_out_tq:.2f} N·m")

with tabs[3]:
    st.subheader("Shaft Sizing (ASME combined torsion + bending) & Deflection")
    st.caption(f"Input: strength {d_in_strength:.2f} mm vs twist {d_in_twist:.2f} mm | "
           f"Output: strength {d_out_strength:.2f} mm vs twist {d_out_twist:.2f} mm "
           f"-> larger value used (rounded up to 0.5 mm).")
    st.dataframe(pd.DataFrame([
        [f'Input ({input_member})', T_in_design_Nmm / 1000, Te_in / 1000, d_shaft_in, defl_in['theta_deg'],
         'PASS' if defl_in_ok else 'FAIL'],
        [f'Output ({output_member})', T_out_design_Nmm / 1000, Te_out / 1000, d_shaft_out, defl_out['theta_deg'],
         'PASS' if defl_out_ok else 'FAIL'],
    ], columns=['Shaft', 'Design Torque (N·m)', 'Equiv. Torque Te (N·m)', 'Required Dia (mm)', 'Twist (°)', 'Twist <=0.5° Status']),
        hide_index=True, use_container_width=True)
    st.caption(f"Loading: Kb={Kb}, Kt={Kt}, keyway Kw={Kw}. External bending assumed negligible (close-coupled bearings).")
    st.markdown("**Planet pin — sizing summary**")
    st.dataframe(pd.DataFrame({
        'Quantity': ['Resultant mesh load', 'Support span', 'Max bending moment', 'Support shear (each)',
                     'Dia. required (bending)', 'Dia. required (shear)', 'Design pin diameter'],
        'Value': [f"{stress['F_pin']:.1f} N", f"{pin_span_mm:.1f} mm", f"{pin['M_max']:.1f} N·mm",
                  f"{pin['V']:.1f} N", f"{pin['d_bend']:.2f} mm", f"{pin['d_shear']:.2f} mm", f"{pin['d_pin']:.2f} mm"]
    }), hide_index=True, use_container_width=True)

with tabs[4]:
    st.subheader("Bearings — Main + Planet")
    st.markdown(f"**Main shaft bearing** (mesh diameter used: {d_mesh:.2f} mm, on {input_member} shaft)")
    st.dataframe(pd.DataFrame({
        'Quantity': ['Resultant radial load (×SF)', 'Dynamic capacity Cdyn', 'Check', 'L10 life (Mrev)', 'L10 life (hours)'],
        'Value': [f"{main_bearing_radial:.1f} N", f"{cdyn:.1f} N", 'PASS' if main_bearing_pass else 'FAIL',
                  f"{main_bearing_life['L10_Mrev']:.2f}", f"{main_bearing_life['L10_h']:.0f} h"]
    }), hide_index=True, use_container_width=True)
    st.caption(f"Evaluated at {main_bearing_speed:.1f} rpm, {main_bearing_type} bearing.")
    st.markdown("**Planet pin bearing / bush**")
    if planet_bearing_life is not None:
        st.dataframe(pd.DataFrame({
            'Quantity': ['Resultant planet load', 'Cdyn', 'Relative spin speed', 'L10 life (Mrev)', 'L10 life (hours)'],
            'Value': [f"{stress['F_pin']:.1f} N", f"{cdyn_planet:.1f} N", f"{kin['n_planet_rel_carrier']:.1f} rpm",
                      f"{planet_bearing_life['L10_Mrev']:.2f}", f"{planet_bearing_life['L10_h']:.0f} h"]
        }), hide_index=True, use_container_width=True)
    else:
        st.write(f"Plain bronze bush — bearing pressure {pin['bearing_pressure']:.2f} MPa vs allowable "
                 f"{allow_bearing_pressure:.2f} MPa -> {'PASS' if pin['pressure_ok'] else 'FAIL'}")

with tabs[5]:
    st.subheader("Planet Load Sharing")
    st.write(f"**Load-sharing factor Kp:** {Kp:.2f}  |  **Application factor KA:** {KA:.2f}  |  "
             f"**Dynamic factor Kv (computed):** {KV:.3f}  (accuracy grade {accuracy_grade}, "
             f"pitch-line velocity {pitch_line_v:.2f} m/s)")
    st.write(f"**Face load factor KHb/KFb:** {KHbeta:.3f}  |  **Transverse load factor KHa/KFa:** {KHalpha:.3f}")
    st.write(f"**Worst-case Sun-Planet tangential force (incl. Kp):** {stress['Ft_SP']:.1f} N")
    st.write(f"**Life factors from {desired_life_hours:,} h duty:** ZN={ZN:.3f}, YN={YN:.3f} "
             f"(N_cycles ~ {N_cycles:,.0f})")
    st.caption("Real planetary trains rarely share load perfectly between the 3 planets — Kp conservatively "
               "loads the worst mesh above the theoretical 1/N share.")

with tabs[6]:
    st.subheader("Planet Pin — Full Design Check")
    st.caption("Pin modelled as a simply-supported beam spanning the two carrier plates, loaded at mid-span.")
    st.dataframe(pd.DataFrame({
        'Quantity': ['Resultant mesh load on pin', 'Support span', 'Max bending moment', 'Support shear (each)',
                     'Dia required (bending)', 'Dia required (shear)', 'Design pin diameter',
                     'Bearing/bush pressure', 'Allowable pressure', 'Pressure check'],
        'Value': [f"{stress['F_pin']:.1f} N", f"{pin_span_mm:.1f} mm", f"{pin['M_max']:.1f} N·mm",
                  f"{pin['V']:.1f} N", f"{pin['d_bend']:.2f} mm", f"{pin['d_shear']:.2f} mm",
                  f"{pin['d_pin']:.2f} mm", f"{pin['bearing_pressure']:.2f} MPa", f"{allow_bearing_pressure:.2f} MPa",
                  'PASS' if pin['pressure_ok'] else 'FAIL']
    }), hide_index=True, use_container_width=True)

with tabs[7]:
    st.subheader("Carrier — Arm & Plate Bending")
    st.markdown("**Carrier arm** (cantilever beam, hub to pin centre)")
    st.dataframe(pd.DataFrame({
        'Quantity': ['Resultant pin load', 'Arm length', 'Arm width', 'Arm thickness',
                     'Bending moment', 'Bending stress', 'Allowable stress', 'Safety factor', 'Status'],
        'Value': [f"{stress['F_pin']:.1f} N", f"{arm_length_mm:.1f} mm", f"{arm_width_mm:.1f} mm",
                  f"{arm_thickness_mm:.1f} mm", f"{carrier_arm_res['M']:.1f} N·mm",
                  f"{carrier_arm_res['sigma']:.1f} MPa", f"{mat['sigma_allow_bend']:.0f} MPa",
                  f"{carrier_arm_res['sf']:.2f}", 'PASS' if carrier_arm_res['pass'] else 'FAIL']
    }), hide_index=True, use_container_width=True)
    st.markdown("**Carrier plate** (sector-cantilever, bore to pin circle)")
    st.dataframe(pd.DataFrame({
        'Quantity': ['Pin circle radius', 'Bore radius (+margin)', 'Plate thickness', 'Tributary width/planet',
                     'Bending moment', 'Bending stress', 'Allowable stress', 'Safety factor', 'Status'],
        'Value': [f"{r_pin_circle:.2f} mm", f"{r_bore_carrier:.2f} mm", f"{plate_thickness_mm:.1f} mm",
                  f"{carrier_plate_res['tributary_width']:.2f} mm", f"{carrier_plate_res['M']:.1f} N·mm",
                  f"{carrier_plate_res['sigma']:.1f} MPa", f"{mat['sigma_allow_bend']:.0f} MPa",
                  f"{carrier_plate_res['sf']:.2f}", 'PASS' if carrier_plate_res['pass'] else 'FAIL']
    }), hide_index=True, use_container_width=True)

with tabs[8]:
    st.subheader("Ring Gear Rim Strength (AGMA-style YB)")
    st.dataframe(pd.DataFrame({
        'Quantity': ['Whole tooth depth ht', 'Rim thickness', 'Rim ratio mB', 'Rim-thickness factor YB',
                     'Ring sF (unadjusted)', 'Ring sF (rim-adjusted)', 'Allowable sF', 'Safety factor', 'Status'],
        'Value': [f"{ring_rim_res['ht']:.2f} mm", f"{ring_rim_res['rim_thickness']:.2f} mm",
                  f"{ring_rim_res['mB']:.2f}", f"{ring_rim_res['YB']:.2f}", f"{stress['sigmaF_RP']:.1f} MPa",
                  f"{ring_rim_res['sigmaF_adj']:.1f} MPa", f"{mat['sigmaF']:.0f} MPa",
                  f"{ring_rim_res['sf']:.2f}", 'PASS' if ring_rim_res['pass'] else 'FAIL']
    }), hide_index=True, use_container_width=True)
    st.caption("AGMA guidance recommends mB >= 1.2 so the ring doesn't flex/crack behind the teeth.")

with tabs[9]:
    st.subheader("Keys / Splines")
    st.dataframe(pd.DataFrame([
        {'Location': f'Input shaft ({d_shaft_in:.1f}mm)', 'Key force (N)': f"{key_in_res['F_key']:.1f}",
         'Shear (MPa)': f"{key_in_res['tau_key']:.1f}", 'SF shear': f"{key_in_res['sf_shear']:.2f}",
         'Bearing (MPa)': f"{key_in_res['sigma_bearing']:.1f}", 'SF bearing': f"{key_in_res['sf_bearing']:.2f}",
         'Status': 'PASS' if key_in_res['pass'] else 'FAIL'},
        {'Location': f'Output shaft ({d_shaft_out:.1f}mm)', 'Key force (N)': f"{key_out_res['F_key']:.1f}",
         'Shear (MPa)': f"{key_out_res['tau_key']:.1f}", 'SF shear': f"{key_out_res['sf_shear']:.2f}",
         'Bearing (MPa)': f"{key_out_res['sigma_bearing']:.1f}", 'SF bearing': f"{key_out_res['sf_bearing']:.2f}",
         'Status': 'PASS' if key_out_res['pass'] else 'FAIL'},
    ]), hide_index=True, use_container_width=True)
    st.caption(f"Key section {key_width_mm:.1f}×{key_height_mm:.1f}×{key_length_mm:.1f} mm — shear + crush checked.")

with tabs[10]:
    st.subheader("Component Dimensions Summary")
    rows = []
    for name, dims in comp.items():
        for k, v in dims.items():
            rows.append([name, k.replace('_', ' ').title(), f"{v:.2f}", "mm"])
    st.dataframe(pd.DataFrame(rows, columns=['Component', 'Dimension', 'Value', 'Unit']),
                 hide_index=True, use_container_width=True)
    st.caption("Bore/hub/rim/housing allowances use representative sizing rules — confirm against the actual "
               "bearing/bush/key parts chosen.")

with tabs[11]:
    st.subheader("Overall Design Results")
    kpi = pd.DataFrame([
        ['Achieved ratio', f"1:{ratio_actual:.3f}", f"target 1:{target_ratio:.2f}"],
        ['Output speed', f"{output_speed:.2f} rpm", "—"],
        ['Motor power (design)', f"{motor_power_design_w:.1f} W", "computed"],
        ['Motor power (operating)', f"{motor_power_operating_w:.1f} W", "computed"],
        ['Design output torque', f"{design_out_tq:.2f} N·m", "65–75 N·m band"],
        ['Face width', f"{face_width:.1f} mm", f"{face_width_factor:.0f}×m"],
        ['Ring OD', f"{est_od:.1f} mm", f"<= {max_od:.0f} mm"],
        ['Input shaft dia', f"{d_shaft_in:.2f} mm", "ASME"],
        ['Output shaft dia', f"{d_shaft_out:.2f} mm", "ASME"],
        ['Planet pin dia', f"{pin['d_pin']:.2f} mm", "bend/shear governed"],
        ['Gear bending SF (min)', f"{min(sfF_sp, sfF_rp, sfF_planet):.2f}", ">=1 required"],
        ['Gear contact SF (min)', f"{min(sfH_sp, sfH_rp):.2f}", ">=1 required"],
        ['Main bearing L10', f"{main_bearing_life['L10_h']:.0f} h", "—"],
        ['Planet bearing L10', f"{planet_bearing_life['L10_h']:.0f} h" if planet_bearing_life else "n/a", "—"],
    ], columns=['Metric', 'Value', 'Notes'])
    st.dataframe(kpi, hide_index=True, use_container_width=True)

    st.markdown("### Pass / Fail Checks")
    st.dataframe(pd.DataFrame({'Check': list(overall_checks.keys()),
                                'Status': ['PASS ✅' if v else 'FAIL ❌' for v in overall_checks.values()]}),
                 hide_index=True, use_container_width=True)
    st.metric("Overall Design Status", "PASS ✅" if overall_ok else "CHECK REQUIRED ⚠️")
    if overall_ok:
        st.balloons()

    summary_text = f"""PLANETARY GEARBOX — DESIGN SUMMARY
Configuration: {fixed_case}  ({input_member} -> {output_member})
Achieved Ratio: 1:{ratio_actual:.3f}  (target 1:{target_ratio:.2f})
Teeth: S={S} P={P} R={R}  Module={m_use:.2f}mm  Face width={face_width:.1f}mm
Ring OD: {est_od:.1f} mm  (limit {max_od:.0f} mm)

Motor power (operating): {motor_power_operating_w:.1f} W
Motor power (design):    {motor_power_design_w:.1f} W
Design output torque:    {design_out_tq:.2f} N.m

Gear bending SF: SP={sfF_sp:.2f}  RP={sfF_rp:.2f}  combined={sfF_planet:.2f}
Gear contact SF: SP={sfH_sp:.2f}  RP={sfH_rp:.2f}
Input shaft dia:  {d_shaft_in:.2f} mm  (twist {defl_in['theta_deg']:.3f} deg)
Output shaft dia: {d_shaft_out:.2f} mm  (twist {defl_out['theta_deg']:.3f} deg)
Planet pin dia:   {pin['d_pin']:.2f} mm
Carrier arm SF:   {carrier_arm_res['sf']:.2f}
Carrier plate SF: {carrier_plate_res['sf']:.2f}
Ring rim SF:      {ring_rim_res['sf']:.2f}
Main bearing L10: {main_bearing_life['L10_h']:.0f} h

OVERALL STATUS: {'PASS' if overall_ok else 'CHECK REQUIRED'}
"""
    st.download_button("⬇️ Download Design Summary (TXT)", summary_text,
                        "planetary_gearbox_summary.txt", "text/plain")

with tabs[12]:
    st.subheader("📖 Parameter Glossary")
    for category, entries in PARAM_GLOSSARY.items():
        with st.expander(category, expanded=False):
            st.dataframe(pd.DataFrame(entries, columns=['Symbol', 'Meaning', 'Unit', 'Typical Value', 'Role']),
                         hide_index=True, use_container_width=True)

with tabs[13]:
    st.subheader("🧊 3D Visualization — True-Involute CAD-style Solids")
    st.caption("Drag to orbit, scroll to zoom, double-click to reset. Slider explodes the assembly along Z.")
    explode = st.slider("Explode assembly", 0.0, 1.0, 0.0, 0.05)
    fig_asm = create_assembly_3d(S, P, R, m_use, N_PLANETS, pin['d_pin'], face_width, d_shaft_in, d_shaft_out,
                                  geom['sun'], geom['planet'], comp, explode=explode)
    st.plotly_chart(fig_asm, use_container_width=True)

    st.markdown("### Individual Components")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(create_component_3d('Sun Gear', {'z': S, 'm': m_use, 'b': face_width,
                                                           'bore': comp['Sun Gear']['bore_d']}),
                         use_container_width=True)
    with col2:
        st.plotly_chart(create_component_3d('Planet Gear', {'z': P, 'm': m_use, 'b': face_width,
                                                              'bore': comp['Planet Gear']['bore_d']}),
                         use_container_width=True)
    with col3:
        st.plotly_chart(create_component_3d('Ring Gear', {'z': R, 'm': m_use, 'b': face_width + 2.0,
                                                            'outer_d': comp['Ring Gear']['outer_d']}),
                         use_container_width=True)
    col4, col5, col6 = st.columns(3)
    with col4:
        st.plotly_chart(create_component_3d('Carrier', {'pitch_r': r_pin_circle, 'd_pin': pin['d_pin'],
                                                          'thk': plate_thickness_mm,
                                                          'hub_od': comp['Carrier Plate']['hub_od'],
                                                          'bore': comp['Carrier Plate']['output_bore']}),
                         use_container_width=True)
    with col5:
        st.plotly_chart(create_component_3d('Input Shaft', {'d': d_shaft_in, 'length': comp['Input Shaft']['length']}),
                         use_container_width=True)
    with col6:
        st.plotly_chart(create_component_3d('Output Shaft', {'d': d_shaft_out, 'length': comp['Output Shaft']['length']}),
                         use_container_width=True)
    st.plotly_chart(create_component_3d('Planet Pin', {'d': pin['d_pin'], 'length': comp['Planet Pin']['total_length']}),
                     use_container_width=True)

with tabs[14]:
    st.subheader("📐 OpenSCAD Parametric Export")
    st.caption("Trapezoidal-tooth approximation suitable for 3D printing/CNC roughing. "
               "For true involute profiles, use OpenSCAD's `gears.scad` library.")
    scad_view = st.selectbox("View to generate:", [
        "assembly", "exploded", "sun", "planet", "ring", "carrier", "input_shaft", "output_shaft", "planet_pin", "housing"
    ], index=0)
    scad_code = generate_openscad(
        S, P, R, m_use, N_PLANETS, pin['d_pin'], face_width,
        comp['Sun Gear']['hub_od'], comp['Sun Gear']['bore_d'],
        comp['Planet Gear']['hub_od'], comp['Planet Gear']['bore_d'],
        comp['Ring Gear']['outer_d'], comp['Ring Gear']['face_width'],
        r_pin_circle, plate_thickness_mm, comp['Carrier Plate']['hub_od'], comp['Carrier Plate']['output_bore'],
        d_shaft_in, d_shaft_out, pin_span_mm,
        comp['Housing']['outer_d'], comp['Housing']['length'], show=scad_view)
    st.code(scad_code, language="openscad")
    st.download_button("⬇️ Download OpenSCAD file (.scad)", scad_code,
                        f"planetary_gearbox_{scad_view}.scad", "text/plain", key="scad_dl")

st.divider()
with st.expander("📋 Pastable Input Configuration (JSON)"):
    config_json = json.dumps({
        "fixed_case": fixed_case, "target_ratio": target_ratio, "max_od_mm": max_od,
        "efficiency": efficiency, "sun_teeth": S, "planet_teeth": P, "ring_teeth": R,
        "module_mm": m_use, "input_torque_nm": t_in_nm, "input_speed_rpm": n_in_rpm,
        "design_output_torque_nm": design_out_tq, "material": selected_mat,
        "Kb": Kb, "Kt": Kt, "Kw": Kw, "Kp": Kp, "KA": KA, "accuracy_grade": accuracy_grade,
        "theta_deg": theta_deg, "desired_life_hours": desired_life_hours,
        "face_width_factor": face_width_factor,
    }, indent=2)
    st.code(config_json, language="json")

st.caption("⚠️ Simplified sizing tool (ISO 6336-lite / ASME shaft code / AGMA-style rim factor / "
           "Lundberg-Palmgren bearing life). Verify against full standards before production release.")
"""
================================================================================
 gear_profile.py  -  TRUE TOOTH PROFILE ENGINE for the Planetary Gearbox Designer
================================================================================
 One class (GearProfile) generates the tooth outline for ALL three members, so
 sun, planet and ring automatically inherit exactly the same parameters.

 What it computes (spur gears, beta = 0):
   [A] Involute flank      - exact involute of the base circle rb = r*cos(alpha)
   [B] Root fillet         - TROCHOIDAL fillet generated by a rack cutter whose
                             tip is rounded with radius rf (this is the curve a
                             hob / rack cutter really leaves), tangent to both
                             the involute flank and the root circle
   [C] Addendum / dedendum - ha = ha* m,  hf = (ha* + c*) m   (c* = clearance)
   [D] Backlash            - circumferential backlash j_t at the pitch circle,
                             split equally between the two gears of each mesh
                             (each tooth is thinned by j_t/2 on the pitch circle)
   [E] Profile relief      - tip relief and root relief: start point + max depth
                             + exponent (removed normal to the flank)
   [F] Crowning (drum)     - lead crowning depth dl on external gears (OpenSCAD
                             export builds it as stacked slices)
   [G] Assembly phasing    - exact rotation of each planet and of the ring so the
                             teeth really mesh (needed for a correct CAD/print)
   [H] Checks              - undercut, pointed tip, fillet limit, print-readiness
   [I] Exports             - Plotly meshes and a parametric OpenSCAD file that
                             uses the REAL profile points (no trapezoids)

 Not included (be aware): profile shift (x = 0 assumed, standard centre distance),
 helical teeth, keyways/splines, bearing seats, fasteners.
================================================================================
"""

import math
from dataclasses import dataclass, asdict
import numpy as np

PI = math.pi


def inv(a):
    """Involute function inv(a) = tan(a) - a."""
    return math.tan(a) - a


# ================================================================
# [1] PARAMETER SET  (shared by sun, planet and ring)
# ================================================================
@dataclass
class ProfileParams:
    alpha_n_deg: float = 20.0        # normal pressure angle
    ha_star: float = 1.00            # addendum coefficient      ha = ha* . m
    c_star: float = 0.25             # root clearance coefficient c  = c*  . m  -> hf = (ha*+c*) m
    rf_star: float = 0.30            # root fillet radius / module   rf = rf* . m   (auto-clamped)
    backlash_mm: float = 0.15        # circumferential backlash j_t at pitch circle, PER MESH
    # --- profile modifications (measured normal to the flank, mm) ---
    tip_relief_start: float = 0.50   # tip relief starts this many modules ABOVE the pitch circle
    tip_relief_depth_mm: float = 0.03
    root_relief_start: float = 0.50  # root relief starts this many modules BELOW the pitch circle
    root_relief_depth_mm: float = 0.02
    relief_exponent: float = 2.0     # 1 = linear ramp, 2 = parabolic ramp
    crowning_mm: float = 0.0         # lead crowning depth dl at both face ends (external gears)

    def key(self):
        return tuple(asdict(self).values())


# ================================================================
# [2] GEAR PROFILE
# ================================================================
class GearProfile:
    """
    Tooth outline of one gear. Tooth centre-line sits on polar angle 0.
    internal=False -> external gear (sun, planet); True -> internal ring gear.
    Coordinates: outline_xy() is a closed CCW polyline (z teeth).
    """

    def __init__(self, z, m, pp: ProfileParams, internal=False, quality=1.0):
        self.z, self.m, self.pp, self.internal = int(z), float(m), pp, internal
        self.warnings = []
        a = math.radians(pp.alpha_n_deg)
        self.alpha = a
        self.sgn = -1.0 if internal else 1.0          # +1: tip is outside pitch circle
        self.r = z * m / 2.0                           # pitch radius
        self.rb = self.r * math.cos(a)                 # BASE CIRCLE radius  (rb)
        self.ha = pp.ha_star * m                       # ADDENDUM (ha)
        self.hf = (pp.ha_star + pp.c_star) * m         # DEDENDUM (hf)
        self.r_tip = self.r + self.sgn * self.ha
        self.r_root = self.r - self.sgn * self.hf
        p = PI * m
        self.s_pitch = p / 2.0 - pp.backlash_mm / 2.0  # tooth thickness on pitch circle
        self.e_pitch = p - self.s_pitch                # space width on pitch circle

        # --- root fillet radius (rf), limited by the cutter geometry ---
        rho_max = pp.c_star * m / (1.0 - math.sin(a))  # tip round must stay tangent below the active flank
        rho = pp.rf_star * m
        if rho > rho_max:
            self.warnings.append(f"Root fillet rf={rho:.3f} mm > max {rho_max:.3f} mm for c*={pp.c_star}; clamped.")
            rho = rho_max
        self.rho = max(rho, 1e-3 * m)

        self._q = max(quality, 0.25)
        self._build()

    # ---------- involute flank (closed form) ----------
    def _half_angle(self, rr):
        """Half tooth-thickness angle at radius rr (exact involute)."""
        ar = math.acos(min(self.rb / rr, 1.0))
        if not self.internal:
            return self.s_pitch / (2 * self.r) + inv(self.alpha) - inv(ar)
        return self.s_pitch / (2 * self.r) - inv(self.alpha) + inv(ar)

    # ---------- profile relief (tip + root) ----------
    def relief_mm(self, rr):
        """Material removed normal to the flank at radius rr (tip + root relief)."""
        pp, m = self.pp, self.m
        ht = (rr - self.r) * self.sgn                  # >0 above pitch circle, <0 below
        d = 0.0
        ts = pp.tip_relief_start * m
        if pp.tip_relief_depth_mm > 0 and ht > ts:
            t = min((ht - ts) / max(self.ha - ts, 1e-9), 1.0)
            d += pp.tip_relief_depth_mm * t ** pp.relief_exponent
        rs = pp.root_relief_start * m
        if pp.root_relief_depth_mm > 0 and -ht > rs:
            t = min((-ht - rs) / max(self.ha - rs, 1e-9), 1.0)
            d += pp.root_relief_depth_mm * t ** pp.relief_exponent
        return d

    def _relief_angle(self, rr):
        ar = math.acos(min(self.rb / rr, 1.0))
        return self.relief_mm(rr) / (rr * math.cos(ar))   # normal -> tangential -> angular

    # ---------- trochoidal root fillet (rack-cutter generation) ----------
    def _fillet(self, npts):
        """
        Rack cutter with tip radius rho rolls on the pitch circle. The cutter tip-round
        centre is at rack coords (a_c, b_c); the fillet is the offset (by rho) of the
        trochoid traced by that centre.  Returns polar arrays (r, theta) in a frame
        whose tooth-SPACE centre is at theta = 0, running root -> flank.
        """
        a, rho, hf = self.alpha, self.rho, self.hf
        rs = self.sgn * self.r                          # signed pitch radius (internal -> negative)
        ta = math.tan(a)
        b_c = -hf + rho
        a_c = self.e_pitch / 2.0 + b_c * ta - rho / math.cos(a)
        a_t = a_c + rho * math.cos(a)                   # tangent point on straight flank
        b_t = b_c - rho * math.sin(a)
        phi = np.linspace(-a_c / rs, -(a_t + b_t / ta) / rs, npts)   # root-circle tangent -> flank tangent
        xg = (rs + b_c) * np.cos(phi) + (a_c + rs * phi) * np.sin(phi)
        yg = -(rs + b_c) * np.sin(phi) + (a_c + rs * phi) * np.cos(phi)
        dx = -b_c * np.sin(phi) + (a_c + rs * phi) * np.cos(phi)
        dy = -b_c * np.cos(phi) - (a_c + rs * phi) * np.sin(phi)
        n = np.hypot(dx, dy)
        nx, ny = -dy / n, dx / n
        cand = []
        for sg in (+1.0, -1.0):
            px, py = xg + sg * rho * nx, yg + sg * rho * ny
            cand.append((px, py, math.hypot(px[0], py[0])))
        pick = min(cand, key=lambda c: c[2]) if not self.internal else max(cand, key=lambda c: c[2])
        px, py = pick[0], pick[1]
        if self.internal:                                # negative pitch radius -> curve lands at ~180 deg: rotate back
            px, py = -px, -py
        if math.atan2(py[0], px[0]) < 0:                # keep the '+' side at positive angle
            py = -py
        return np.hypot(px, py), np.arctan2(py, px)

    # ---------- assemble one full pitch (one tooth + one space) ----------
    def _build(self):
        z, q = self.z, self._q
        pz = PI / z
        r_f, th_f = self._fillet(max(int(14 * q), 6))
        # numerical guard: the sampled fillet must progress monotonically root -> tip; drop any
        # trailing samples that fold back past the tangent point (sub-micron effect, avoids
        # self-touching polygons in CAD)
        prog = (r_f - self.r_root) * self.sgn
        bad = np.where(np.diff(prog) <= 0)[0]
        if len(bad):
            cut = bad[0] + 1
            r_f, th_f = r_f[:cut], th_f[:cut]
        r_j = r_f[-1]
        # exact-involute angle at the fillet/flank junction (should equal th_f[-1])
        th_inv_j = pz - self._half_angle(r_j)
        self.join_error_mm = abs(th_inv_j - th_f[-1]) * r_j
        th_f[-1] = th_inv_j                              # snap junction exactly onto the involute

        # undercut: involute cannot exist below the base circle (external gears)
        r_start = r_j
        self.undercut = (not self.internal) and (r_j < self.rb * (1 + 1e-9))
        if self.undercut:
            r_start = self.rb * 1.0001
            self.warnings.append(f"z={z}: UNDERCUT (z < 17 for 20 deg, x=0). Use z >= 17 or profile shift.")
            keep = r_f >= r_start
            r_f, th_f = r_f[keep] if keep.any() else r_f[-1:], th_f[keep] if keep.any() else th_f[-1:]

        # relief continuity: fade the join's relief into the fillet
        dj = self._relief_angle(r_start)
        w = np.linspace(0, 1, len(r_f)) ** 2 * (3 - 2 * np.linspace(0, 1, len(r_f)))
        th_f = th_f + dj * w

        # involute flank  r_start -> r_tip
        nfl = max(int(26 * q), 8)
        r_fl = np.linspace(r_start, self.r_tip, nfl)[1:]
        th_fl = np.array([pz - self._half_angle(rr) + self._relief_angle(rr) for rr in r_fl])

        th1 = th_fl[-1]
        if th1 >= pz:
            self.warnings.append(f"z={z}: POINTED teeth (tip thickness <= 0). Reduce backlash/relief or ha*.")
            th1 = pz * 0.999
            th_fl[-1] = th1
        self.tip_width_mm = 2 * self.r_tip * (pz - th1)
        if self.tip_width_mm < 0.2 * self.m:
            self.warnings.append(f"z={z}: tip width {self.tip_width_mm:.2f} mm < 0.2 m (weak/sharp tip).")

        nroot = max(int(3 * q), 2)
        root_r = np.full(nroot, self.r_root)
        root_th = np.linspace(0.0, th_f[0], nroot, endpoint=False)

        half_r = np.concatenate([root_r, r_f, r_fl])
        half_t = np.concatenate([root_th, th_f, th_fl])
        ntip = max(int(3 * q), 2)
        tip_t = np.linspace(th1, 2 * pz - th1, ntip + 2)[1:-1]
        tip_r = np.full(len(tip_t), self.r_tip)

        r_all = np.concatenate([half_r, tip_r, half_r[::-1]])
        t_all = np.concatenate([half_t, tip_t, (2 * pz - half_t)[::-1]])
        r_all, t_all = r_all[:-1], t_all[:-1]           # drop duplicate = next period's first point
        t_all = t_all - pz                              # tooth centre-line -> angle 0
        self.period_xy = np.column_stack([r_all * np.cos(t_all), r_all * np.sin(t_all)])

    # ---------- public geometry ----------
    def outline_xy(self, rotation=0.0):
        """Full closed outline of the gear, rotated by `rotation` rad."""
        out = []
        per = 2 * PI / self.z
        for i in range(self.z):
            ang = rotation + i * per
            c, s = math.cos(ang), math.sin(ang)
            out.append(np.column_stack([self.period_xy[:, 0] * c - self.period_xy[:, 1] * s,
                                        self.period_xy[:, 0] * s + self.period_xy[:, 1] * c]))
        return np.vstack(out)

    def summary(self):
        return {'z': self.z, 'type': 'internal' if self.internal else 'external',
                'r_pitch': self.r, 'r_base (rb)': self.rb, 'r_tip': self.r_tip, 'r_root': self.r_root,
                'ha': self.ha, 'hf': self.hf, 'fillet rf': self.rho,
                'tooth thk @pitch': self.s_pitch, 'space @pitch': self.e_pitch,
                'tip width': self.tip_width_mm, 'undercut': self.undercut,
                'join error (mm)': self.join_error_mm}


# ================================================================
# [3] SHARED PROFILE SET + ASSEMBLY PHASING
# ================================================================
_CACHE = {}


def build_profiles(S, P, R, m, pp: ProfileParams, quality=1.0):
    """Sun, planet, ring - all inherit the SAME ProfileParams."""
    k = (S, P, R, round(m, 6), pp.key(), quality)
    if k not in _CACHE:
        if len(_CACHE) > 40:
            _CACHE.clear()
        _CACHE[k] = {'sun': GearProfile(S, m, pp, False, quality),
                     'planet': GearProfile(P, m, pp, False, quality),
                     'ring': GearProfile(R, m, pp, True, quality)}
    return _CACHE[k]


def planetary_phases(S, P, R, N):
    """
    Rotation (rad) for each planet and for the ring so that teeth mesh exactly
    with the sun (sun tooth centred on +X, angle 0). Returns (planet_rot[], ring_rot, max_phase_error).
    Requires (S+R) % N == 0.
    """
    tp = 2 * PI
    planet_rot = []
    for k in range(N):
        phi = k * tp / N
        u_s = (phi * S / tp) % 1.0                      # sun tooth phase seen from planet k
        # external mesh: the planet sits on the opposite side of the contact, so its phase runs
        # the other way round (u_p = 0.5 - u_s)
        planet_rot.append(phi + PI - tp * (0.5 - u_s) / P)   # planet tooth <-> sun space
    phi0 = 0.0
    u_p0 = ((phi0 - planet_rot[0]) * P / tp) % 1.0
    ring_rot = phi0 - tp * (u_p0 + 0.5) / R             # planet tooth <-> ring space
    err = 0.0
    for k in range(N):
        phi = k * tp / N
        u_p = ((phi - planet_rot[k]) * P / tp) % 1.0
        v = ((phi - ring_rot) * R / tp) % 1.0
        d = abs(((v - (u_p + 0.5)) + 0.5) % 1.0 - 0.5)
        err = max(err, d)
    return planet_rot, ring_rot, err


# ================================================================
# [4] CHECKS
# ================================================================
def print_readiness(profiles, sun_bore, m, nozzle_mm=0.4, process="FDM"):
    """Guideline checks for a 3D printed gearset (heuristics, not standards)."""
    sun, planet, ring = profiles['sun'], profiles['planet'], profiles['ring']
    pp = sun.pp
    rows = []

    def add(name, ok, detail):
        rows.append({'Check': name, 'Status': 'PASS' if ok else 'FAIL', 'Detail': detail})

    add("No undercut (sun/planet)", not (sun.undercut or planet.undercut),
        f"sun z={sun.z}, planet z={planet.z}; need z>=17 for 20 deg with x=0")
    add("Tooth tips not pointed", min(sun.tip_width_mm, planet.tip_width_mm, ring.tip_width_mm) >= 0.2 * m,
        f"min tip width {min(sun.tip_width_mm, planet.tip_width_mm, ring.tip_width_mm):.2f} mm (>= {0.2 * m:.2f})")
    if process == "FDM":
        add("Module printable on FDM", m >= 1.5 or nozzle_mm <= 0.3,
            f"m={m:.2f} mm with {nozzle_mm} mm nozzle (m >= 1.5 recommended; teeth ~ 3 perimeters thick)")
        add("Backlash >= FDM tolerance", pp.backlash_mm >= 0.15,
            f"j_t={pp.backlash_mm:.2f} mm (typ. 0.15-0.30 mm for FDM, 0.10-0.15 mm SLA/SLS)")
    add("Root fillet >= half nozzle", sun.rho >= nozzle_mm / 2 or process != "FDM",
        f"rf={sun.rho:.2f} mm")
    rim = (sun.r_root * 2 - sun_bore) / 2
    add("Sun rim behind teeth >= 1.5 mm", rim >= 1.5,
        f"(root dia - bore)/2 = {rim:.2f} mm. Keyway NOT modelled: a 6x6 key will not fit if rim is small - use D-flat / set screw")
    add("Fillet within cutter limit", not any('clamped' in w for w in sun.warnings), f"rf={sun.rho:.2f} mm")
    return rows


def verify_meshing_shapely(S, P, R, m, N, pp: ProfileParams, steps=12):
    """
    Numerical proof of correct meshing: rotates the train through one tooth pitch of the sun
    (ring fixed) and measures overlap area and minimum gap. Needs `shapely` (pip install shapely).
    Returns dict(max_overlap_mm2, min_gap_sp_mm, min_gap_pr_mm, phase_err).
    """
    from shapely.geometry import Polygon
    from shapely import affinity
    prof = build_profiles(S, P, R, m, pp, quality=1.5)
    planet_rot, ring_rot, perr = planetary_phases(S, P, R, N)
    rc = (S + P) * m / 2.0
    sun_poly0 = Polygon(prof['sun'].outline_xy(0.0))
    ring_void0 = Polygon(prof['ring'].outline_xy(ring_rot))
    ring_mat = Polygon(_circle(R * m / 2 + 4 * m, 720)).difference(ring_void0)
    pl0 = prof['planet'].outline_xy(0.0)
    worst, gsp, gpr = 0.0, 1e9, 1e9
    i_ratio = 1.0 + R / S
    for st in range(steps):
        c = (2 * PI / S) / i_ratio * st / steps        # carrier angle sweeping one sun pitch
        sun = affinity.rotate(sun_poly0, c * i_ratio, origin=(0, 0), use_radians=True)
        for k in range(N):
            phi = k * 2 * PI / N + c
            th = planet_rot[k] + c * (1.0 - R / P)
            cs, sn = math.cos(th), math.sin(th)
            xy = np.column_stack([pl0[:, 0] * cs - pl0[:, 1] * sn + rc * math.cos(phi),
                                  pl0[:, 0] * sn + pl0[:, 1] * cs + rc * math.sin(phi)])
            pp_ = Polygon(xy)
            worst = max(worst, pp_.intersection(sun).area, pp_.intersection(ring_mat).area)
            gsp = min(gsp, pp_.distance(sun))
            gpr = min(gpr, pp_.distance(ring_mat))
    return {'max_overlap_mm2': worst, 'min_gap_sun_planet_mm': gsp,
            'min_gap_planet_ring_mm': gpr, 'phase_err': perr}


def _circle(r, n=96, off=0.0):
    a = np.linspace(0, 2 * PI, n, endpoint=False) + off
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


# ================================================================
# [5] PLOTLY 3D  (uses the real profile; replaces the old approximated teeth)
# ================================================================
def _ring_between(outer, inner, z0, z1):
    """Solid between two equally-sampled closed polylines (matched by polar angle)."""
    n = len(outer)
    vx = np.concatenate([outer[:, 0], outer[:, 0], inner[:, 0], inner[:, 0]])
    vy = np.concatenate([outer[:, 1], outer[:, 1], inner[:, 1], inner[:, 1]])
    vz = np.concatenate([np.full(n, z0), np.full(n, z1), np.full(n, z0), np.full(n, z1)])
    I, J, K = [], [], []
    for i in range(n):
        ni = (i + 1) % n
        ob, ot, hb, ht = i, i + n, i + 2 * n, i + 3 * n
        nob, nt, nhb, nht = ni, ni + n, ni + 2 * n, ni + 3 * n
        I += [ob, ob, hb, hb, ob, ob, ot, ot]
        J += [nob, nt, nht, ht, hb, nhb, nt, nht]
        K += [nt, ot, nhb, nht, nhb, nob, nht, ht]
    return vx, vy, vz, I, J, K


def _disk(poly, z0, z1):
    n = len(poly)
    vx = np.concatenate([poly[:, 0], poly[:, 0], [0, 0]])
    vy = np.concatenate([poly[:, 1], poly[:, 1], [0, 0]])
    vz = np.concatenate([np.full(n, z0), np.full(n, z1), [z0, z1]])
    cb, ct = 2 * n, 2 * n + 1
    I, J, K = [], [], []
    for i in range(n):
        ni = (i + 1) % n
        I += [i, i, cb, ct]
        J += [ni, ni + n, ni, ni + n]
        K += [ni + n, i + n, i, i + n]
    return vx, vy, vz, I, J, K


def _match_circle(poly, radius):
    ang = np.arctan2(poly[:, 1], poly[:, 0])
    return np.column_stack([radius * np.cos(ang), radius * np.sin(ang)])


def mesh_external(profile, rot, cx, cy, z0, z1, bore_d=None):
    o = profile.outline_xy(rot)
    m = _ring_between(o, _match_circle(o, bore_d / 2), z0, z1) if bore_d else _disk(o, z0, z1)
    vx, vy, vz, I, J, K = m
    return vx + cx, vy + cy, vz, I, J, K


def mesh_ring(profile, rot, outer_d, z0, z1):
    inner = profile.outline_xy(rot)
    return _ring_between(_match_circle(inner, outer_d / 2), inner, z0, z1)


def _trace(mesh, color, name, opacity=1.0):
    import plotly.graph_objects as go
    vx, vy, vz, I, J, K = mesh
    return go.Mesh3d(x=vx, y=vy, z=vz, i=I, j=J, k=K, color=color, opacity=opacity, name=name,
                     flatshading=False, showlegend=True,
                     lighting=dict(ambient=0.40, diffuse=0.85, specular=0.75, roughness=0.30, fresnel=0.15),
                     lightposition=dict(x=300, y=200, z=400))


def _scene(title, height):
    import plotly.graph_objects as go
    ax = dict(backgroundcolor='#0e1117', gridcolor='#2a2f3a', showbackground=True)
    fig = go.Figure()
    fig.update_layout(scene=dict(xaxis=dict(title='X (mm)', **ax), yaxis=dict(title='Y (mm)', **ax),
                                 zaxis=dict(title='Z (mm)', **ax), aspectmode='data',
                                 camera=dict(eye=dict(x=1.6, y=1.4, z=1.0))),
                      paper_bgcolor='#0e1117', font=dict(color='#e6e6e6'),
                      title=dict(text=title, font=dict(size=16, color='#e6e6e6')),
                      height=height, margin=dict(l=0, r=0, t=45, b=0),
                      legend=dict(bgcolor='rgba(20,20,20,0.7)', bordercolor='#444', borderwidth=1))
    return fig


def create_assembly_3d(S, P, R, m, N, pp, b, d_pin, d_in, d_out, comp, plate_thk, explode=0.0, quality=0.5):
    """Correctly PHASED assembly built from the real tooth profile."""
    prof = build_profiles(S, P, R, m, pp, quality)
    planet_rot, ring_rot, _ = planetary_phases(S, P, R, N)
    rc = (S + P) * m / 2.0
    gap = b * 1.6 * explode
    fig = _scene("3D Planetary Gearbox - true involute + trochoidal fillet profile", 720)
    fig.add_trace(_trace(mesh_external(prof['sun'], 0.0, 0, 0, 0, b, comp['Sun Gear']['bore_d']), '#E85D2A', 'Sun'))
    for k in range(N):
        phi = k * 2 * PI / N
        fig.add_trace(_trace(mesh_external(prof['planet'], planet_rot[k], rc * math.cos(phi), rc * math.sin(phi),
                                            gap, b + gap, comp['Planet Gear']['bore_d']), '#F2B01E', f'Planet {k + 1}'))
    rv = mesh_ring(prof['ring'], ring_rot, comp['Ring Gear']['outer_d'], -1 + 2 * gap, b + 1 + 2 * gap)
    fig.add_trace(_trace(rv, '#9AA0A6', 'Ring', 0.45))
    zc = b + 0.6 + 3 * gap
    cr = rc + 1.5 * d_pin
    cv = _ring_between(_circle(cr, 96), _circle(comp['Carrier Plate']['output_bore'] / 2, 96), zc, zc + plate_thk)
    fig.add_trace(_trace(cv, '#3F6FB5', 'Carrier', 0.6))
    fig.add_trace(_trace(_disk(_circle(d_in / 2, 48), -45 - gap, b), '#B0B0B0', 'Input shaft'))
    fig.add_trace(_trace(_disk(_circle(d_out / 2, 48), zc, zc + plate_thk + 40 + 4 * gap), '#B0B0B0', 'Output shaft'))
    for k in range(N):
        phi = k * 2 * PI / N
        pc = _disk(_circle(d_pin / 2, 32), gap, zc + plate_thk)
        fig.add_trace(_trace((pc[0] + rc * math.cos(phi), pc[1] + rc * math.sin(phi), pc[2], pc[3], pc[4], pc[5]),
                             '#2E2E2E', f'Pin {k + 1}'))
    return fig


def create_gear_3d(kind, S, P, R, m, pp, b, bore_d=None, outer_d=None, quality=1.0):
    """Single-gear preview: kind in {'Sun Gear','Planet Gear','Ring Gear'}."""
    prof = build_profiles(S, P, R, m, pp, quality)
    fig = _scene(kind, 460)
    if kind == 'Sun Gear':
        fig.add_trace(_trace(mesh_external(prof['sun'], 0, 0, 0, 0, b, bore_d), '#E85D2A', kind))
    elif kind == 'Planet Gear':
        fig.add_trace(_trace(mesh_external(prof['planet'], 0, 0, 0, 0, b, bore_d), '#F2B01E', kind))
    else:
        fig.add_trace(_trace(mesh_ring(prof['ring'], 0, outer_d, 0, b), '#9AA0A6', kind, 0.6))
    return fig


# ================================================================
# [6] OPENSCAD EXPORT  (real profile points, correct phasing, crowning)
# ================================================================
def _scad_pts(arr):
    return "[" + ",".join(f"[{x:.4f},{y:.4f}]" for x, y in arr) + "]"


def generate_openscad(S, P, R, m, N, pp: ProfileParams, d_pin, b, sun_bore, planet_bore, ring_outer_d,
                      pitch_r, plate_thk, carrier_hub_od, carrier_bore, din, dout, in_len, out_len,
                      housing_od, housing_len, axial_gap=0.6, crown_slices=8, show="assembly"):
    prof = build_profiles(S, P, R, m, pp, quality=1.0)
    planet_rot, ring_rot, perr = planetary_phases(S, P, R, N)
    pr_deg = ",".join(f"{math.degrees(a):.6f}" for a in planet_rot)
    hub_len = max(1.5 * dout, 10.0)
    return f"""// =====================================================================
// PLANETARY GEARBOX - OPENSCAD MODEL (true involute + trochoidal root fillet)
// Generated by Planetary Gearbox Designer v6  |  units: mm
// S={S} P={P} R={R} m={m}  ratio(ring fixed) = {(S + R) / S:.4f}  planets={N}
// alpha={pp.alpha_n_deg} deg  ha*={pp.ha_star}  c*={pp.c_star}  rf={prof['sun'].rho:.3f} mm
// backlash j_t={pp.backlash_mm} mm  tip relief={pp.tip_relief_depth_mm} mm  root relief={pp.root_relief_depth_mm} mm
// crowning={pp.crowning_mm} mm   phase error (should be ~0) = {perr:.2e}
//
// SCOPE: gear set + carrier + shafts + pins. NOT modelled: keyways, bearing seats,
// retaining rings, ring-gear fastening, real housing. Set SHOW to a single part
// and export STL for printing (one part per file, print gears flat on the bed).
// =====================================================================
SHOW = "{show}";     // assembly | exploded | sun | planet | ring | carrier | input_shaft | output_shaft | planet_pin | housing

n_planets = {N};
zs = {S}; zp = {P}; zr = {R};
b = {b};                    // face width
pitch_r = {pitch_r:.4f};   // planet centre radius = (zs+zp)*m/2
d_pin = {d_pin};
axial_gap = {axial_gap};
crown = {pp.crowning_mm};  crown_n = {crown_slices};

// ---- one pitch of tooth outline per gear (CCW, tooth centre on +X) ----
sun_period    = {_scad_pts(prof['sun'].period_xy)};
planet_period = {_scad_pts(prof['planet'].period_xy)};
ring_period   = {_scad_pts(prof['ring'].period_xy)};

// ---- assembly phasing (deg) so the teeth actually mesh ----
planet_rot = [{pr_deg}];
ring_rot   = {math.degrees(ring_rot):.6f};

function rot(p, a) = [p[0]*cos(a) - p[1]*sin(a), p[0]*sin(a) + p[1]*cos(a)];
function gear_pts(period, z) = [for (i = [0 : z-1]) for (p = period) rot(p, i*360/z)];

module gear2d(period, z) {{ polygon(gear_pts(period, z)); }}

// External gear (sun / planet); lead crowning as stacked slices when crown > 0
module ext_gear(period, z, bore) {{
    difference() {{
        if (crown <= 0) linear_extrude(height = b) gear2d(period, z);
        else for (i = [0 : crown_n-1]) {{
            t = 2*(i + 0.5)/crown_n - 1;
            translate([0, 0, i*b/crown_n])
                linear_extrude(height = b/crown_n + 0.001)
                    offset(delta = -crown*t*t) gear2d(period, z);
        }}
        translate([0, 0, -1]) cylinder(h = b + 2, d = bore, $fn = 128);
    }}
}}

module sun_gear()    {{ ext_gear(sun_period, zs, {sun_bore:.3f}); }}
module planet_gear() {{ ext_gear(planet_period, zp, {planet_bore:.3f}); }}

module ring_gear_part() {{
    rf = b + 2;
    difference() {{
        translate([0, 0, -1]) cylinder(h = rf, d = {ring_outer_d:.3f}, $fn = 256);
        translate([0, 0, -2]) linear_extrude(height = rf + 2) rotate(ring_rot) gear2d(ring_period, zr);
    }}
}}

module carrier() {{
    difference() {{
        union() {{
            cylinder(h = {plate_thk:.3f}, r = pitch_r + 1.5*d_pin, $fn = 128);
            cylinder(h = {plate_thk:.3f} + {hub_len:.2f}, d = {carrier_hub_od:.3f}, $fn = 96);
        }}
        translate([0, 0, -1]) cylinder(h = {plate_thk:.3f} + {hub_len:.2f} + 2, d = {carrier_bore:.3f}, $fn = 96);
        for (i = [0 : n_planets-1]) rotate([0, 0, i*360/n_planets]) translate([pitch_r, 0, -1])
            cylinder(h = {plate_thk:.3f} + 2, d = d_pin + 0.1, $fn = 48);   // press/slip fit for the pin
    }}
}}

module input_shaft()  {{ cylinder(h = {in_len:.2f},  d = {din:.3f},  $fn = 64); }}
module output_shaft() {{ cylinder(h = {out_len:.2f}, d = {dout:.3f}, $fn = 64); }}
module planet_pin_part() {{ cylinder(h = b + axial_gap + {plate_thk:.3f}, d = d_pin, $fn = 48); }}

module housing() {{   // ENVELOPE ONLY - a plain tube, no bearing seats
    difference() {{
        cylinder(h = {housing_len:.2f}, d = {housing_od:.3f}, $fn = 256);
        translate([0, 0, -1]) cylinder(h = {housing_len:.2f} + 2, d = {housing_od:.3f} - 6, $fn = 256);
    }}
}}

// =====================================================================
// ASSEMBLY  (gears occupy z = 0 .. b)
// =====================================================================
module assembly(ex = 0) {{
    g = ex * b * 1.6;
    color("DarkOrange") sun_gear();
    color("DimGray") translate([0, 0, -{in_len:.2f} + b - g]) input_shaft();
    for (i = [0 : n_planets-1]) rotate([0, 0, i*360/n_planets]) translate([pitch_r, 0, 0]) {{
        color("Gold") translate([0, 0, g]) rotate(planet_rot[i] - i*360/n_planets) planet_gear();
        color("DarkSlateGray") translate([0, 0, g]) planet_pin_part();
    }}
    color("SlateGray", 0.5) translate([0, 0, 2*g]) ring_gear_part();
    color("SteelBlue", 0.85) translate([0, 0, b + axial_gap + 3*g]) carrier();
    color("DimGray") translate([0, 0, b + axial_gap + 3*g]) output_shaft();
}}

if      (SHOW == "assembly")     assembly(0);
else if (SHOW == "exploded")     assembly(0.6);
else if (SHOW == "sun")          sun_gear();
else if (SHOW == "planet")       planet_gear();
else if (SHOW == "ring")         ring_gear_part();
else if (SHOW == "carrier")      carrier();
else if (SHOW == "input_shaft")  input_shaft();
else if (SHOW == "output_shaft") output_shaft();
else if (SHOW == "planet_pin")   planet_pin_part();
else if (SHOW == "housing")      housing();
"""
