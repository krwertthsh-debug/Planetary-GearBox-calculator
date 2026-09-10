"""
================================================================================
 PLANETARY GEARBOX DESIGNER — COMBINED v4
================================================================================
- Fixed 3-planet, single-stage epicyclic gearbox (target ratio 1:9)
- Motor power is COMPUTED from torque/speed/ratio/η (never a direct input)
- Design (worst-case) torque locked to 65-75 N·m band
- 15 output tabs with full calculations + 3D + OpenSCAD
--------------------------------------------------------------------------------
Uses simplified ISO 6336-lite / ASME shaft code / AGMA-style rim factor /
Lundberg-Palmgren bearing life / lumped thermal balance. First-pass sizing aid.
Verify against full standards before production release.
================================================================================
"""

import math
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

PI = math.pi

# ================================================================
# 1. FIXED CONSTANTS
# ================================================================
N_PLANETS_FIXED       = 3               # ← fixed per request
TARGET_RATIO_DEFAULT  = 9.0
MAX_OD_DEFAULT_MM     = 200.0
EFFICIENCY_DEFAULT    = 0.97
PRESSURE_ANGLE        = 20.0
HELIX_ANGLE           = 0.0
MODULE_LIST           = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
DESIGN_OUT_TQ_MIN     = 65.0
DESIGN_OUT_TQ_MAX     = 75.0
RATED_T_IN_NM_MIN     = 5.0
RATED_N_IN_RPM_MIN    = 1500.0
FACE_WIDTH_FACTOR     = 16.0            # b = 16·m  (auto-computed, no UI slider)

MATERIAL_PROPS = {
    '17CrNiMo6 / 18CrNiMo7-6 (Case Carburized)': {
        'tau': 240.0, 'E': 210000.0, 'nu': 0.30,
        'sigmaF_lim': 430.0, 'sigmaH_lim': 1500.0, 'sigma_allow_bend': 380.0,
        'sigma_allow_bearing': 480.0, 'density': 7850.0},
    'Alloy Steel (EN24 / 4340 Hardened)': {
        'tau': 150.0, 'E': 206000.0, 'nu': 0.30,
        'sigmaF_lim': 310.0, 'sigmaH_lim': 1150.0, 'sigma_allow_bend': 260.0,
        'sigma_allow_bearing': 320.0, 'density': 7850.0},
    'Case Carburized Steel (20MnCr5 / 16MnCr5)': {
        'tau': 140.0, 'E': 210000.0, 'nu': 0.30,
        'sigmaF_lim': 380.0, 'sigmaH_lim': 1350.0, 'sigma_allow_bend': 320.0,
        'sigma_allow_bearing': 400.0, 'density': 7850.0},
    'SAE 6150 / 51CrV4 (Spring Steel)': {
        'tau': 170.0, 'E': 207000.0, 'nu': 0.30,
        'sigmaF_lim': 350.0, 'sigmaH_lim': 1250.0, 'sigma_allow_bend': 290.0,
        'sigma_allow_bearing': 360.0, 'density': 7850.0},
    'Stainless Steel (316)': {
        'tau': 50.0, 'E': 193000.0, 'nu': 0.31,
        'sigmaF_lim': 170.0, 'sigmaH_lim': 600.0, 'sigma_allow_bend': 140.0,
        'sigma_allow_bearing': 180.0, 'density': 8000.0},
    'Mild Steel (AISI 1020)': {
        'tau': 40.0, 'E': 200000.0, 'nu': 0.29,
        'sigmaF_lim': 140.0, 'sigmaH_lim': 450.0, 'sigma_allow_bend': 110.0,
        'sigma_allow_bearing': 140.0, 'density': 7870.0},
    'Custom': {
        'tau': 240.0, 'E': 200000.0, 'nu': 0.30,
        'sigmaF_lim': 300.0, 'sigmaH_lim': 1200.0, 'sigma_allow_bend': 250.0,
        'sigma_allow_bearing': 300.0, 'density': 7850.0},
}

ASME_FACTORS = {
    'Gradually applied / steady load':      (1.5, 1.0),
    'Minor shocks (typical machine drive)': (1.5, 1.2),
    'Heavy shocks / frequent starts':       (2.0, 1.5),
}

# ================================================================
# 2. TOOTH-COUNT SYNTHESIS
# ================================================================
def find_teeth_combo(fixed_case, target_ratio, n_planets,
                     s_range=(15, 91), p_range=(15, 91)):
    best_err, best_combo = float('inf'), (0, 0, 0, 0.0, False)
    for S in range(s_range[0], s_range[1]):
        for P in range(p_range[0], p_range[1]):
            R = S + 2 * P
            assembly_ok  = (S + R) % n_planets == 0
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
    assembly_ok  = (S + R) % n_planets == 0
    clearance_ok = (S + P) * math.sin(math.radians(180 / n_planets)) > (P + 2)
    if fixed_case == 'Ring Fixed':
        ratio = (S + R) / S
    elif fixed_case == 'Sun Fixed':
        ratio = (S + R) / R
    else:
        ratio = R / S
    return R, ratio, assembly_ok, clearance_ok, abs(ratio - target_ratio)


def suggest_tooth_sets(target_ratio, n_planets, max_od, modules=MODULE_LIST):
    rows = []
    for zs in range(15, 91):
        for zp in range(15, 91):
            zr = zs + 2 * zp
            if (zs + zr) % n_planets != 0:
                continue
            if (zs + zp) * math.sin(math.radians(180 / n_planets)) <= (zp + 2):
                continue
            ratio = (zs + zr) / zs
            if abs(ratio - target_ratio) > 1e-6:
                continue
            for m in modules:
                est_od = (zr + 2.5) * m
                if est_od <= max_od:
                    rows.append(dict(zs=zs, zp=zp, zr=zr, ratio=ratio,
                                     module=m, od=est_od))
    return pd.DataFrame(rows)


# ================================================================
# 3. GEAR GEOMETRY  (2D / analytical)
# ================================================================
def gear_geometry(S, P, R, m, alpha_n_deg, beta_deg=0.0):
    alpha_n = math.radians(alpha_n_deg)
    beta    = math.radians(beta_deg)
    alpha_t = math.atan(math.tan(alpha_n) / math.cos(beta))

    def ext(z):
        d = z * m / math.cos(beta)
        return {'z': z, 'd_pitch': d, 'd_base': d * math.cos(alpha_t),
                'd_tip': d + 2 * m, 'd_root': d - 2.5 * m}

    def internal(z):
        d = z * m / math.cos(beta)
        return {'z': z, 'd_pitch': d, 'd_base': d * math.cos(alpha_t),
                'd_tip': d - 2 * m, 'd_root': d + 2.5 * m}

    return {
        'alpha_t_deg': math.degrees(alpha_t),
        'sun': ext(S), 'planet': ext(P), 'ring': internal(R),
        'center_dist_sun_planet': (S + P) * m / (2 * math.cos(beta)),
        'center_dist_ring_planet': (R - P) * m / (2 * math.cos(beta)),
        'circular_pitch': PI * m,
    }


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
            'n_planet_spin_rel_carrier': n_planet_spin_rel,
            'output_speed': output_speed}


# ================================================================
# 5. GEAR TOOTH FORCE & STRESS
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
    YFa, YSa = params['YFa'], params['YSa']
    Yeps, Ybeta = params['Yeps'], params['Ybeta']
    ZH, Zeps, Zbeta = params['ZH'], params['Zeps'], params['Zbeta']
    ZR, YR = params['ZR'], params['YR']
    theta_deg = params['theta_deg']
    n_planets = params.get('n_planets', 3)

    alpha_n_rad = math.radians(alpha_n)
    beta_rad    = math.radians(beta)
    alpha_t     = math.atan(math.tan(alpha_n_rad) / math.cos(beta_rad))

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

    num_S = Ft_SP * KA * KV * KFbeta * KFalpha
    num_R = Ft_RP * KA * KV * KFbeta * KFalpha
    den   = b * mn
    sigmaF_SP = (num_S / den) * YFa['S'] * YSa['S'] * Yeps * Ybeta
    sigmaF_RP = (num_R / den) * YFa['P'] * YSa['P'] * Yeps * Ybeta

    theta = math.radians(theta_deg)
    sigmaF_planet = math.sqrt(max(sigmaF_SP**2 + sigmaF_RP**2
                                  - 2 * sigmaF_SP * sigmaF_RP * math.cos(theta), 0))
    F_pin = math.sqrt(max(Fn_SP**2 + Fn_RP**2
                          - 2 * Fn_SP * Fn_RP * math.cos(theta), 0))

    ZE = math.sqrt(1.0 / (PI * (((1 - nu1**2) / E1) + ((1 - nu2**2) / E2))))
    u_SP = zP / zS
    u_RP = zR / zP
    term_SP = (Ft_SP * KA * KV * KHbeta * KHalpha) / (b * dS) * (u_SP + 1) / u_SP
    sigmaH_SP = ZH * ZE * Zeps * Zbeta * math.sqrt(max(term_SP, 0.0))
    term_RP = (Ft_RP * KA * KV * KHbeta * KHalpha) / (b * dP) * (u_RP - 1) / u_RP
    sigmaH_RP = ZH * ZE * Zeps * Zbeta * math.sqrt(max(term_RP, 0.0))
    sigmaH_ring = sigmaH_RP * ZR
    sigmaF_ring = sigmaF_RP * YR

    return {'Ft_SP': Ft_SP, 'Ft_RP': Ft_RP, 'Fr_SP': Fr_SP, 'Fr_RP': Fr_RP,
            'Fn_SP': Fn_SP, 'Fn_RP': Fn_RP, 'F_pin': F_pin,
            'sigmaF_SP': sigmaF_SP, 'sigmaF_RP': sigmaF_RP, 'sigmaF_planet': sigmaF_planet,
            'sigmaH_SP': sigmaH_SP, 'sigmaH_RP': sigmaH_RP,
            'sigmaH_ring': sigmaH_ring, 'sigmaF_ring': sigmaF_ring,
            'rS': rS, 'rP': rP, 'rR': rR, 'ZE': ZE}


# ================================================================
# 6. SHAFT / PIN / CARRIER / RIM / KEY / BEARING / THERMAL
# ================================================================
def shaft_diameter_asme(T_nmm, M_nmm, tau_allow_mpa, Kb, Kt, Kw=1.0):
    Te = math.sqrt((Kb * M_nmm) ** 2 + (Kt * Kw * T_nmm) ** 2)
    d = (16.0 * Te / (PI * tau_allow_mpa)) ** (1.0 / 3.0)
    return d, Te


def design_planet_pin(F_pin_N, span_mm, face_width_mm, sigma_bend, tau_shear,
                       allow_pressure):
    M_max = F_pin_N * span_mm / 4.0
    V_sup = F_pin_N / 2.0
    d_bend  = (32.0 * M_max / (PI * sigma_bend)) ** (1.0 / 3.0)
    d_shear = math.sqrt(4.0 * V_sup / (PI * tau_shear))
    d_pin = max(d_bend, d_shear)
    bp = F_pin_N / (d_pin * face_width_mm)
    return {'M_max_Nmm': M_max, 'V_support_N': V_sup, 'd_bend_mm': d_bend,
            'd_shear_mm': d_shear, 'd_pin_mm': d_pin,
            'bearing_pressure_mpa': bp, 'pressure_ok': bp <= allow_pressure}


def carrier_arm_bending(F_pin_N, L_mm, w_mm, t_mm, sigma_allow):
    M = F_pin_N * L_mm
    Z = w_mm * t_mm ** 2 / 6.0
    s = M / Z if Z > 0 else float('inf')
    sf = sigma_allow / s if s > 0 else float('inf')
    return {'M_Nmm': M, 'sigma_mpa': s, 'sf': sf, 'pass': sf >= 1.0}


def carrier_plate_bending(F_pin_N, n_planets, r_pin_circle_mm, r_bore_mm,
                           t_plate_mm, sigma_allow):
    tw = (2 * PI * r_pin_circle_mm) / n_planets
    arm = max(r_pin_circle_mm - r_bore_mm, 1e-6)
    M = F_pin_N * arm
    Z = tw * t_plate_mm ** 2 / 6.0
    s = M / Z if Z > 0 else float('inf')
    sf = sigma_allow / s if s > 0 else float('inf')
    return {'M_Nmm': M, 'tributary_width_mm': tw, 'sigma_mpa': s,
            'sf': sf, 'pass': sf >= 1.0}


def ring_rim_strength(sigmaF_ring, m_mm, d_root_ring_mm, d_od_ring_mm, sigmaF_lim):
    ht = 2.25 * m_mm
    rim_t = (d_od_ring_mm - d_root_ring_mm) / 2.0
    mB = rim_t / ht if ht > 0 else 0.0
    if mB >= 1.2:
        YB = 1.0
    elif mB > 0:
        YB = 1.6 * math.log(2.242 / mB)
    else:
        YB = float('inf')
    adj = sigmaF_ring * YB
    sf = sigmaF_lim / adj if adj > 0 else float('inf')
    return {'rim_thickness_mm': rim_t, 'ht_mm': ht, 'mB': mB, 'YB': YB,
            'sigmaF_ring_adj_mpa': adj, 'sf': sf, 'pass': sf >= 1.0}


def key_sizing(T_nmm, d_shaft_mm, w, h, l, tau_allow, sigma_allow_bearing):
    F = 2.0 * T_nmm / d_shaft_mm
    tau_k = F / (w * l)
    sig_b = F / (0.5 * h * l)
    return {'F_key_N': F, 'tau_key_mpa': tau_k, 'sigma_bearing_mpa': sig_b,
            'sf_shear': tau_allow / tau_k if tau_k > 0 else float('inf'),
            'sf_bearing': sigma_allow_bearing / sig_b if sig_b > 0 else float('inf'),
            'pass': (tau_allow / tau_k >= 1.0) and (sigma_allow_bearing / sig_b >= 1.0)}


def torsional_deflection_deg(T_nmm, L_mm, E_mpa, nu, d_mm):
    G = E_mpa / (2.0 * (1.0 + nu))
    J = PI * d_mm ** 4 / 32.0
    return {'G_mpa': G, 'J_mm4': J,
            'theta_deg': math.degrees(T_nmm * L_mm / (G * J))}


def mesh_tooth_deflection_um(Fn_N, b_mm, E_mpa=210000.0):
    c_prime = 0.04 * E_mpa / 210000.0 * 20.0
    k = c_prime * b_mm * 1000.0
    return {'stiffness_N_per_mm': k, 'delta_um': (Fn_N / k) * 1000.0 if k > 0 else float('inf')}


def bearing_L10_life(C_dyn_N, P_eq_N, n_rpm, bearing_type='Ball'):
    p = 3.0 if bearing_type == 'Ball' else 10.0 / 3.0
    if P_eq_N <= 0 or n_rpm <= 0:
        return {'L10_Mrev': float('inf'), 'L10_h': float('inf'), 'p': p}
    L10 = (C_dyn_N / P_eq_N) ** p
    return {'L10_Mrev': L10, 'L10_h': (L10 * 1.0e6) / (60.0 * n_rpm), 'p': p}


def thermal_lubrication_check(power_loss_w, ring_od_mm, face_width_mm,
                               ambient_c=25.0, h_conv=15.0,
                               d_sun_mm=None, n_in_rpm=None):
    h_od_m = (ring_od_mm + 20.0) / 1000.0
    h_len_m = (face_width_mm + 40.0) / 1000.0
    area = PI * h_od_m * h_len_m + 2 * (PI * (h_od_m / 2) ** 2)
    dT = power_loss_w / (h_conv * area) if area > 0 else float('inf')
    Tss = ambient_c + dT
    plv = None
    if d_sun_mm and n_in_rpm:
        plv = (PI * d_sun_mm / 1000.0 * n_in_rpm) / 60.0
    splash_ok = (plv is not None) and (plv <= 15.0)
    rec = ("Splash / bath lubrication acceptable" if splash_ok
           else "Pitch-line velocity high — consider forced/spray lubrication")
    return {'housing_area_m2': area, 'delta_T_c': dT, 'steady_state_temp_c': Tss,
            'pitch_line_velocity_mps': plv, 'splash_ok': splash_ok,
            'lube_recommendation': rec, 'temp_ok': Tss <= 90.0}


# ================================================================
# 7. 3D GEOMETRY (visualisation-only simplified trapezoidal teeth)
# ================================================================
def make_gear_outline(z, m, internal=False):
    """Simple trapezoidal tooth outline for 3D visualisation only."""
    r_pitch = z * m / 2.0
    if internal:
        r_tip  = r_pitch - m
        r_root = r_pitch + 1.25 * m
    else:
        r_tip  = r_pitch + m
        r_root = r_pitch - 1.25 * m
    half_tip  = PI / (2 * z) * 0.55
    half_root = PI / (2 * z) * 1.05
    period = 2 * PI / z
    pts = []
    for i in range(z):
        t0 = i * period
        pts.append((r_root * math.cos(t0 - half_root), r_root * math.sin(t0 - half_root)))
        pts.append((r_tip  * math.cos(t0 - half_tip),  r_tip  * math.sin(t0 - half_tip)))
        pts.append((r_tip  * math.cos(t0 + half_tip),  r_tip  * math.sin(t0 + half_tip)))
        pts.append((r_root * math.cos(t0 + half_root), r_root * math.sin(t0 + half_root)))
    return np.array(pts)


def make_circle_outline(r, n):
    a = np.linspace(0, 2 * PI, n, endpoint=False)
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def extrude_solid(outline_xy, z0, z1):
    n = len(outline_xy)
    x, y = outline_xy[:, 0], outline_xy[:, 1]
    vx = np.concatenate([x, x, [0.0, 0.0]])
    vy = np.concatenate([y, y, [0.0, 0.0]])
    vz = np.concatenate([np.full(n, z0), np.full(n, z1), [z0, z1]])
    c_bot, c_top = 2 * n, 2 * n + 1
    I, J, K = [], [], []
    for i in range(n):
        ni = (i + 1) % n
        I.append(i); J.append(ni); K.append(ni + n)
        I.append(i); J.append(ni + n); K.append(i + n)
        I.append(c_bot); J.append(ni); K.append(i)
        I.append(c_top); J.append(i + n); K.append(ni + n)
    return vx, vy, vz, I, J, K


def extrude_annulus(outer_xy, inner_xy, z0, z1):
    n = len(outer_xy)
    ox, oy = outer_xy[:, 0], outer_xy[:, 1]
    ix, iy = inner_xy[:, 0], inner_xy[:, 1]
    vx = np.concatenate([ox, ox, ix, ix])
    vy = np.concatenate([oy, oy, iy, iy])
    vz = np.concatenate([np.full(n, z0), np.full(n, z1),
                         np.full(n, z0), np.full(n, z1)])
    ob, ot, ib, it = 0, n, 2 * n, 3 * n
    I, J, K = [], [], []
    for i in range(n):
        ni = (i + 1) % n
        I.append(ob + i); J.append(ob + ni); K.append(ot + ni)
        I.append(ob + i); J.append(ot + ni); K.append(ot + i)
        I.append(ib + i); J.append(it + ni); K.append(ib + ni)
        I.append(ib + i); J.append(it + i); K.append(it + ni)
        I.append(ob + i); J.append(ib + i); K.append(ib + ni)
        I.append(ob + i); J.append(ib + ni); K.append(ob + ni)
        I.append(ot + i); J.append(ot + ni); K.append(it + ni)
        I.append(ot + i); J.append(it + ni); K.append(it + i)
    return vx, vy, vz, I, J, K


def cylinder_mesh(r, z0, z1, n=32):
    a = np.linspace(0, 2 * PI, n, endpoint=False)
    x = r * np.cos(a)
    y = r * np.sin(a)
    vx = np.concatenate([x, x, [0.0, 0.0]])
    vy = np.concatenate([y, y, [0.0, 0.0]])
    vz = np.concatenate([np.full(n, z0), np.full(n, z1), [z0, z1]])
    c_bot, c_top = 2 * n, 2 * n + 1
    I, J, K = [], [], []
    for i in range(n):
        ni = (i + 1) % n
        I.append(i); J.append(ni); K.append(ni + n)
        I.append(i); J.append(ni + n); K.append(i + n)
        I.append(c_bot); J.append(ni); K.append(i)
        I.append(c_top); J.append(i + n); K.append(ni + n)
    return vx, vy, vz, I, J, K


def create_assembly_3d(zs, zp, zr, m, n_planets, d_pin, face_width,
                       d_in_shaft, d_out_shaft, comp_dims, pin_d,
                       carrier_pitch_r):
    traces = []
    # Sun
    vx, vy, vz, I, J, K = extrude_solid(make_gear_outline(zs, m, False),
                                         -face_width/2, face_width/2)
    traces.append(go.Mesh3d(x=vx, y=vy, z=vz, i=I, j=J, k=K,
                            color='#D9531E', opacity=1.0, name='Sun',
                            flatshading=True, lighting=dict(ambient=0.5, diffuse=0.9)))
    # Planets
    for k in range(n_planets):
        ang = k * 2 * PI / n_planets
        vx, vy, vz, I, J, K = extrude_solid(make_gear_outline(zp, m, False),
                                             -face_width/2, face_width/2)
        px, py = carrier_pitch_r * math.cos(ang), carrier_pitch_r * math.sin(ang)
        traces.append(go.Mesh3d(x=vx + px, y=vy + py, z=vz, i=I, j=J, k=K,
                                color='#EDB120', opacity=1.0, name=f'Planet {k+1}',
                                flatshading=True))
    # Ring
    inner = make_gear_outline(zr, m, True)
    n = len(inner)
    outer = make_circle_outline(comp_dims['ring']['outer_d'] / 2.0, n)
    vx, vy, vz, I, J, K = extrude_annulus(outer, inner,
                                           -face_width/2 - 1, face_width/2 + 1)
    traces.append(go.Mesh3d(x=vx, y=vy, z=vz, i=I, j=J, k=K,
                            color='#8a8a8a', opacity=0.65, name='Ring'))
    # Carrier
    vx, vy, vz, I, J, K = cylinder_mesh(comp_dims['carrier']['plate_od'] / 2.0,
                                         -face_width/2 - 1.2,
                                         -face_width/2 - 1.2 + comp_dims['carrier']['plate_thickness'])
    traces.append(go.Mesh3d(x=vx, y=vy, z=vz, i=I, j=J, k=K,
                            color='#4C72B0', opacity=0.5, name='Carrier'))
    # Input shaft
    vx, vy, vz, I, J, K = cylinder_mesh(d_in_shaft / 2.0,
                                         -face_width/2 - 40, -face_width/2)
    traces.append(go.Mesh3d(x=vx, y=vy, z=vz, i=I, j=J, k=K,
                            color='#7C7C7C', opacity=1.0, name='Input Shaft'))
    # Output shaft
    vx, vy, vz, I, J, K = cylinder_mesh(d_out_shaft / 2.0,
                                         face_width/2,
                                         face_width/2 + 40)
    traces.append(go.Mesh3d(x=vx, y=vy, z=vz, i=I, j=J, k=K,
                            color='#7C7C7C', opacity=1.0, name='Output Shaft'))
    # Pins
    for k in range(n_planets):
        ang = k * 2 * PI / n_planets
        px, py = carrier_pitch_r * math.cos(ang), carrier_pitch_r * math.sin(ang)
        vx, vy, vz, I, J, K = cylinder_mesh(pin_d / 2.0,
                                             -face_width/2 - 3,
                                             face_width/2 + 3, n=20)
        traces.append(go.Mesh3d(x=vx + px, y=vy + py, z=vz, i=I, j=J, k=K,
                                color='#2B2B2B', opacity=0.9, name=f'Pin {k+1}'))
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
            bgcolor='rgb(20,24,33)',
        ),
        title=dict(text='3D Planetary Gearbox — Interactive Assembly',
                   font=dict(size=16)),
        height=750,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgb(20,24,33)',
        font=dict(color='white'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
    )
    return fig


def create_component_3d(kind, params):
    fig = go.Figure()
    if kind == 'Sun Gear':
        vx, vy, vz, I, J, K = extrude_solid(make_gear_outline(params['z'], params['m']),
                                            0, params['face_width'])
        fig.add_trace(go.Mesh3d(x=vx, y=vy, z=vz, i=I, j=J, k=K,
                                color='#D9531E', name='Sun', flatshading=True))
        title = f"Sun — z={params['z']}, m={params['m']:.2f}"
    elif kind == 'Planet Gear':
        vx, vy, vz, I, J, K = extrude_solid(make_gear_outline(params['z'], params['m']),
                                            0, params['face_width'])
        fig.add_trace(go.Mesh3d(x=vx, y=vy, z=vz, i=I, j=J, k=K,
                                color='#EDB120', name='Planet', flatshading=True))
        title = f"Planet — z={params['z']}, m={params['m']:.2f}"
    elif kind == 'Ring Gear':
        inner = make_gear_outline(params['z'], params['m'], True)
        outer = make_circle_outline(params['outer_d'] / 2.0, len(inner))
        vx, vy, vz, I, J, K = extrude_annulus(outer, inner, 0, params['face_width'])
        fig.add_trace(go.Mesh3d(x=vx, y=vy, z=vz, i=I, j=J, k=K,
                                color='#8a8a8a', name='Ring', flatshading=True))
        title = f"Ring — z={params['z']}, m={params['m']:.2f}"
    elif kind == 'Planet Pin':
        vx, vy, vz, I, J, K = cylinder_mesh(params['diameter'] / 2.0,
                                             0, params['length'])
        fig.add_trace(go.Mesh3d(x=vx, y=vy, z=vz, i=I, j=J, k=K,
                                color='#2B2B2B', name='Pin'))
        title = f"Planet Pin ⌀{params['diameter']:.1f}"
    elif kind == 'Input Shaft':
        vx, vy, vz, I, J, K = cylinder_mesh(params['diameter'] / 2.0,
                                             0, params['length'])
        fig.add_trace(go.Mesh3d(x=vx, y=vy, z=vz, i=I, j=J, k=K,
                                color='#7C7C7C', name='Input Shaft'))
        title = f"Input Shaft ⌀{params['diameter']:.1f}"
    elif kind == 'Output Shaft':
        vx, vy, vz, I, J, K = cylinder_mesh(params['diameter'] / 2.0,
                                             0, params['length'])
        fig.add_trace(go.Mesh3d(x=vx, y=vy, z=vz, i=I, j=J, k=K,
                                color='#7C7C7C', name='Output Shaft'))
        title = f"Output Shaft ⌀{params['diameter']:.1f}"
    else:
        return None
    fig.update_layout(
        scene=dict(aspectmode='data', bgcolor='rgb(20,24,33)'),
        title=dict(text=title, font=dict(color='white')),
        height=450, margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgb(20,24,33)',
    )
    return fig


# ================================================================
# 8. OPENSCAD GENERATOR
# ================================================================
def generate_openscad_assembly(zs, zp, zr, m, n_planets, d_pin, b,
                                sun_hub_od, sun_bore,
                                planet_hub_od, planet_bore,
                                ring_outer_d, ring_face_width,
                                carrier_pitch_r, carrier_plate_thk,
                                carrier_hub_od, carrier_bore,
                                din, dout, pin_span):
    sun_tip  = zs * m + 2 * m
    sun_root = zs * m - 2.5 * m
    p_tip    = zp * m + 2 * m
    p_root   = zp * m - 2.5 * m
    r_root   = zr * m + 2.5 * m
    carrier_od = 2 * (carrier_pitch_r + d_pin * 1.5)
    in_len  = 40 + b + 20
    out_len = 40 + b + 20 + 30

    return f"""// =====================================================
// PLANETARY GEARBOX — OpenSCAD GENERATED CODE (Designer v4)
// =====================================================
// NOTE: Simplified trapezoidal teeth. For true involute
// profiles, use the 'gears.scad' library.

zs = {zs};   zp = {zp};   zr = {zr};   m = {m};
n_planets = {n_planets};
b = {b};     d_pin = {d_pin};   pin_span = {pin_span};

module gear_approx(z, m, width, tip_d, root_d) {{
    pitch_r = z * m / 2;
    tooth_angle = 360 / z;
    tooth_width = m * PI / 2;
    tooth_width_top = tooth_width * 0.6;
    difference() {{
        cylinder(h=width, d=tip_d, center=true);
        cylinder(h=width+2, d=root_d, center=true);
    }}
    for (i = [0 : z-1]) {{
        rotate([0, 0, i * tooth_angle])
            translate([pitch_r, 0, 0])
                polyhedron(
                    points = [
                        [-tooth_width/2, 0, -width/2],
                        [ tooth_width/2, 0, -width/2],
                        [ tooth_width_top/2, 0,  width/2],
                        [-tooth_width_top/2, 0,  width/2]
                    ],
                    faces = [[0,1,2,3]]
                );
    }}
}}

module sun_gear() {{
    difference() {{
        union() {{
            gear_approx(z={zs}, m={m}, width={b},
                        tip_d={sun_tip:.2f}, root_d={sun_root:.2f});
            cylinder(h={b}, d={sun_hub_od:.2f}, center=true);
        }}
        cylinder(h={b + 10}, d={sun_bore:.2f}, center=true);
    }}
}}

module planet_gear() {{
    difference() {{
        union() {{
            gear_approx(z={zp}, m={m}, width={b},
                        tip_d={p_tip:.2f}, root_d={p_root:.2f});
            cylinder(h={b}, d={planet_hub_od:.2f}, center=true);
        }}
        cylinder(h={b + 10}, d={planet_bore:.2f}, center=true);
    }}
}}

module ring_gear() {{
    difference() {{
        cylinder(h={ring_face_width}, d={ring_outer_d:.2f}, center=true);
        cylinder(h={ring_face_width + 2}, d={r_root:.2f}, center=true);
        for (i = [0 : {zr-1}])
            rotate([0, 0, i * 360 / {zr}])
                translate([0, {r_root/2 - m:.2f}, 0])
                    cube([{m}, {m*1.5}, {ring_face_width}], center=true);
    }}
}}

module carrier() {{
    difference() {{
        union() {{
            cylinder(h={carrier_plate_thk}, d={carrier_od:.2f}, center=true);
            cylinder(h={carrier_plate_thk*2}, d={carrier_hub_od:.2f}, center=true);
            for (i = [0 : {n_planets-1}])
                rotate([0, 0, i * 360 / {n_planets}])
                    translate([{carrier_pitch_r:.2f}, 0, 0])
                        cylinder(h={carrier_plate_thk + 10}, d={d_pin+6:.2f}, center=true);
        }}
        cylinder(h={carrier_plate_thk*3}, d={carrier_bore:.2f}, center=true);
        for (i = [0 : {n_planets-1}])
            rotate([0, 0, i * 360 / {n_planets}])
                translate([{carrier_pitch_r:.2f}, 0, 0])
                    cylinder(h={carrier_plate_thk + 20}, d={d_pin:.2f}, center=true);
    }}
}}

module input_shaft()  {{ cylinder(h={in_len},  d={din:.2f},  center=true); }}
module output_shaft() {{ cylinder(h={out_len}, d={dout:.2f}, center=true); }}
module planet_pin()   {{ cylinder(h={pin_span + 10}, d={d_pin:.2f}, center=true); }}

module assembly() {{
    sun_gear();
    for (i = [0 : {n_planets-1}])
        rotate([0, 0, i * 360 / {n_planets}])
            translate([{carrier_pitch_r:.2f}, 0, 0])
                planet_gear();
    ring_gear();
    carrier();
    translate([0, 0, -{in_len/2 + b/2:.2f}])  input_shaft();
    translate([0, 0,  {carrier_plate_thk/2 + b/2:.2f}]) output_shaft();
    for (i = [0 : {n_planets-1}])
        rotate([0, 0, i * 360 / {n_planets}])
            translate([{carrier_pitch_r:.2f}, 0, 0])
                planet_pin();
}}

assembly();

// Uncomment to render individual parts:
// sun_gear(); planet_gear(); ring_gear();
// carrier(); input_shaft(); output_shaft(); planet_pin();
"""


# ================================================================
# 9. STREAMLIT APP
# ================================================================
st.set_page_config(page_title="Planetary Gearbox Designer v4",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header { font-size: 2.0rem; font-weight: 800; color: #1f4e78;
              text-align: center; margin-bottom: 0.3rem;}
.sub-header  { font-size: 0.95rem; color: #666; text-align: center;
              margin-bottom: 1.2rem;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="main-header">⚙️ PLANETARY GEARBOX DESIGNER v4</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Single-Stage 1:9 Reduction · 3-Planet Fixed · '
            'Full Stress/Geometry/Bearing/3D/OpenSCAD Suite</div>',
            unsafe_allow_html=True)


# ================================================================
# SIDEBAR — INPUTS  (per user specification)
# ================================================================
with st.sidebar:
    st.header("1. Ratio / Envelope / Planets")
    target_ratio = st.number_input("Target Ratio (1:x):", 2.0, 20.0,
                                    TARGET_RATIO_DEFAULT, 0.5)
    max_od_mm = st.number_input("Max Ring OD (mm):", 50.0, 1000.0,
                                 MAX_OD_DEFAULT_MM, 5.0)
    efficiency = st.number_input("Mesh Efficiency:", 0.80, 0.99,
                                  EFFICIENCY_DEFAULT, 0.01)
    fixed_case = st.selectbox("Fixed Member:",
                               ['Ring Fixed', 'Sun Fixed', 'Carrier Fixed'])
    st.caption("**Number of planets: 3 (fixed)**")

    st.header("2. Teeth & Module")
    manual_teeth = st.checkbox("Manually set Sun / Planet teeth & module",
                                value=False)
    if manual_teeth:
        S_manual = st.number_input("Sun teeth (Zs):", 10, 150, 20, 1)
        P_manual = st.number_input("Planet teeth (Zp):", 10, 150, 32, 1)
        m_manual = st.selectbox("Module (mm):", MODULE_LIST,
                                 index=MODULE_LIST.index(1.5))
    else:
        st.caption("Auto-search finds the closest ratio match. Ring = Sun + 2·Planet.")

    st.header("3. Operating Conditions")
    t_in_max = st.number_input("Max available Input Torque (N·m):",
                                RATED_T_IN_NM_MIN, 500.0, 20.0, 0.5)
    t_in_nm = st.slider("Input Torque (N·m):", RATED_T_IN_NM_MIN,
                         t_in_max, RATED_T_IN_NM_MIN, 0.1)
    n_in_max = st.number_input("Max available Input Speed (rpm):",
                                RATED_N_IN_RPM_MIN, 20000.0, 3000.0, 50.0)
    n_in_rpm = st.slider("Input Speed (rpm):", RATED_N_IN_RPM_MIN,
                          n_in_max, RATED_N_IN_RPM_MIN, 10.0)

    st.header("4. Design (worst-case) Torque — 65-75 N·m")
    design_out_tq = st.slider("Design Output Torque (N·m):",
                               DESIGN_OUT_TQ_MIN, DESIGN_OUT_TQ_MAX,
                               70.0, 0.5)

    st.header("5. Material")
    selected_mat = st.selectbox("Material:",
                                 list(MATERIAL_PROPS.keys()))
    mat_data = dict(MATERIAL_PROPS[selected_mat])
    if selected_mat == 'Custom':
        with st.expander("Custom material properties", expanded=True):
            mat_data['E'] = st.number_input("E (MPa):", value=mat_data['E'])
            mat_data['nu'] = st.number_input("ν:", value=mat_data['nu'], step=0.01)
            mat_data['sigmaF_lim'] = st.number_input("σF limit (MPa):", value=mat_data['sigmaF_lim'])
            mat_data['sigmaH_lim'] = st.number_input("σH limit (MPa):", value=mat_data['sigmaH_lim'])
            mat_data['sigma_allow_bend'] = st.number_input("Allow. bend (MPa):", value=mat_data['sigma_allow_bend'])
            mat_data['sigma_allow_bearing'] = st.number_input("Allow. bearing (MPa):", value=mat_data['sigma_allow_bearing'])

    tau = st.number_input("Allowable shear τ (MPa):",
                          1.0, 1000.0, mat_data['tau'])
    kw = st.number_input("Keyway stress-conc. factor Kw:",
                          1.0, 2.0, 1.3, 0.05)
    shock_label = st.selectbox("Shaft loading condition (ASME Kb/Kt):",
                                list(ASME_FACTORS.keys()))
    Kb, Kt = ASME_FACTORS[shock_label]

    st.header("6. Main Bearings (Sun / Ring shaft)")
    sf_bearing = st.number_input("Bearing Service Factor:",
                                  1.0, 3.0, 1.5, 0.1)
    cdyn = st.number_input("Main bearing Cdyn (N):",
                            1.0, 100000.0, 12000.0, 500.0)
    main_bearing_type = st.selectbox("Main Bearing Type:", ['Ball', 'Roller'])

    st.header("7. Planet Pin / Bearing")
    pin_span_mm = st.number_input("Pin support span (mm):",
                                   5.0, 200.0, 30.0, 1.0)
    pin_support = st.selectbox("Pin support type:",
                                ['Needle Roller Bearing', 'Plain Bronze Bush'])
    if pin_support == 'Needle Roller Bearing':
        allow_bearing_pressure = st.number_input("Allow. dyn. pressure (MPa):", value=25.0)
        cdyn_planet = st.number_input("Planet Bearing Cdyn (N):", value=6000.0, step=250.0)
    else:
        allow_bearing_pressure = st.number_input("Allow. static bush pressure (MPa):", value=10.0)
        cdyn_planet = None

    st.header("8. Carrier Geometry")
    arm_length_mm = st.number_input("Carrier arm length (mm):",
                                     5.0, 200.0, 25.0, 1.0)
    arm_width_mm = st.number_input("Carrier arm width (mm):",
                                    5.0, 100.0, 18.0, 1.0)
    arm_thickness_mm = st.number_input("Carrier arm thickness (mm):",
                                        3.0, 60.0, 10.0, 1.0)
    plate_thickness_mm = st.number_input("Carrier plate thickness (mm):",
                                          3.0, 60.0, 12.0, 1.0)
    plate_bore_margin_mm = st.number_input("Carrier plate bore margin (mm):",
                                            0.0, 20.0, 2.0, 0.5)

    st.header("9. Keys / Splines")
    key_width_mm = st.number_input("Key width w (mm):",
                                    2.0, 40.0, 6.0, 0.5)
    key_height_mm = st.number_input("Key height h (mm):",
                                     2.0, 40.0, 6.0, 0.5)
    key_length_mm = st.number_input("Key length l (mm):",
                                     5.0, 150.0, 20.0, 1.0)

    st.header("10. Shaft Lengths (for deflection)")
    shaft_len_in_mm = st.number_input("Input shaft length (mm):",
                                       10.0, 500.0, 60.0, 5.0)
    shaft_len_out_mm = st.number_input("Output shaft length (mm):",
                                        10.0, 500.0, 60.0, 5.0)

    st.header("11. Layout")
    theta_deg = st.slider("Angle between mesh force lines on planet (°):",
                           60.0, 180.0, 120.0, 1.0)


# ================================================================
# CALCULATIONS
# ================================================================
n_planets = N_PLANETS_FIXED

if manual_teeth:
    R, ratio_actual, assembly_ok, clearance_ok, ratio_err = evaluate_manual_teeth(
        fixed_case, S_manual, P_manual, n_planets, target_ratio)
    S, P, m_use, found = S_manual, P_manual, m_manual, True
else:
    S, P, R, ratio_actual, found = find_teeth_combo(fixed_case, target_ratio, n_planets)
    assembly_ok = (S + R) % n_planets == 0
    clearance_ok = (S + P) * math.sin(math.radians(180 / n_planets)) > (P + 2)

if not found or S <= 0:
    st.error("No valid tooth combination found. Adjust inputs.")
    st.stop()

if fixed_case == 'Ring Fixed':
    input_member, output_member = 'Sun', 'Carrier'
elif fixed_case == 'Sun Fixed':
    input_member, output_member = 'Ring', 'Carrier'
else:
    input_member, output_member = 'Sun', 'Ring'

output_speed = n_in_rpm / ratio_actual
output_torque_nominal = t_in_nm * ratio_actual * efficiency

# --- Module selection (auto mode) ---
if not manual_teeth:
    m_use = MODULE_LIST[0]
    for m in reversed(MODULE_LIST):
        if (R + 2.5) * m <= max_od_mm:
            m_use = m
            break
est_od = (R + 2.5) * m_use
od_fits = est_od <= max_od_mm

# --- Motor power (computed, never user input) ---
motor_power_operating_w = (2.0 * PI * n_in_rpm / 60.0) * t_in_nm
motor_power_design_w = (2.0 * PI * n_in_rpm / 60.0) * (design_out_tq / (ratio_actual * efficiency))

# --- Geometry ---
geom = gear_geometry(S, P, R, m_use, PRESSURE_ANGLE, HELIX_ANGLE)
d_sun, d_ring = geom['sun']['d_pitch'], geom['ring']['d_pitch']
face_width = FACE_WIDTH_FACTOR * m_use       # auto-computed (no user face-width-ratio)

kin = compute_kinematics_speeds(fixed_case, S, P, n_in_rpm, ratio_actual)

# --- Shaft sizing (ASME) ---
T_in_design_Nmm  = (design_out_tq / (ratio_actual * efficiency)) * 1000.0
T_out_design_Nmm = design_out_tq * 1000.0
d_shaft_in, Te_in   = shaft_diameter_asme(T_in_design_Nmm,  0.0, tau, Kb, Kt, kw)
d_shaft_out, Te_out = shaft_diameter_asme(T_out_design_Nmm, 0.0, tau, Kb, Kt, kw)

# --- Main bearing ---
d_mesh = d_sun if input_member == 'Sun' else d_ring
ft_design = 2 * ((design_out_tq / ratio_actual) * 1000) / d_mesh
f_res = ft_design / math.cos(math.radians(PRESSURE_ANGLE))
f_design = f_res * sf_bearing
bearing_pass = f_design <= cdyn
main_bearing_speed = n_in_rpm if input_member != 'Carrier' else kin['n_carrier']
main_bearing_life = bearing_L10_life(cdyn, f_res,
                                     max(main_bearing_speed, 1e-6), main_bearing_type)

# --- Gear stress ---
stress_params = {
    'zS': S, 'zP': P, 'zR': R, 'mn': m_use, 'alpha_n': PRESSURE_ANGLE,
    'beta': HELIX_ANGLE, 'b': face_width,
    'E1': mat_data['E'], 'E2': mat_data['E'],
    'nu1': mat_data['nu'], 'nu2': mat_data['nu'],
    'TS': T_in_design_Nmm, 'KA': 1.25, 'KV': 1.15,
    'KFbeta': 1.2, 'KFalpha': 1.0, 'KHbeta': 1.25, 'KHalpha': 1.0, 'Kp': 1.05,
    'YFa': {'S': 2.8, 'P': 2.5, 'R': 2.2}, 'YSa': {'S': 1.5, 'P': 1.6, 'R': 1.7},
    'Yeps': 0.85, 'Ybeta': 1.0, 'ZH': 2.5, 'Zeps': 0.9, 'Zbeta': 1.0,
    'ZR': 1.0, 'YR': 1.0, 'theta_deg': theta_deg, 'n_planets': n_planets,
}
stress_res = planetary_gear_stress_3planets(stress_params)
sfF_SP = mat_data['sigmaF_lim'] / stress_res['sigmaF_SP']
sfF_RP = mat_data['sigmaF_lim'] / stress_res['sigmaF_RP']
sfH_SP = mat_data['sigmaH_lim'] / stress_res['sigmaH_SP']
sfH_RP = mat_data['sigmaH_lim'] / stress_res['sigmaH_RP']

# --- Planet pin ---
pin_res = design_planet_pin(stress_res['F_pin'], pin_span_mm, face_width,
                             mat_data['sigma_allow_bend'], tau, allow_bearing_pressure)
if pin_support == 'Needle Roller Bearing' and cdyn_planet:
    planet_bearing_life = bearing_L10_life(cdyn_planet, stress_res['F_pin'],
                                            max(kin['n_planet_spin_rel_carrier'], 1e-6),
                                            'Roller')
else:
    planet_bearing_life = None

# --- Carrier ---
r_pin_circle = geom['sun']['d_pitch'] / 2 + geom['planet']['d_pitch'] / 2
r_bore_carrier = (d_shaft_out / 2.0) + plate_bore_margin_mm
carrier_arm_res = carrier_arm_bending(stress_res['F_pin'], arm_length_mm,
                                       arm_width_mm, arm_thickness_mm,
                                       mat_data['sigma_allow_bend'])
carrier_plate_res = carrier_plate_bending(stress_res['F_pin'], n_planets,
                                           r_pin_circle, r_bore_carrier,
                                           plate_thickness_mm,
                                           mat_data['sigma_allow_bend'])
carrier_plate_od = 2 * (r_pin_circle + pin_res['d_pin_mm'] * 1.5)

# --- Ring rim ---
ring_rim_res = ring_rim_strength(stress_res['sigmaF_ring'], m_use,
                                  geom['ring']['d_root'], est_od,
                                  mat_data['sigmaF_lim'])

# --- Keys ---
key_sun_res = key_sizing(T_in_design_Nmm,  d_shaft_in,  key_width_mm,
                          key_height_mm, key_length_mm, tau,
                          mat_data['sigma_allow_bearing'])
key_out_res = key_sizing(T_out_design_Nmm, d_shaft_out, key_width_mm,
                          key_height_mm, key_length_mm, tau,
                          mat_data['sigma_allow_bearing'])

# --- Deflection ---
defl_in  = torsional_deflection_deg(T_in_design_Nmm,  shaft_len_in_mm,
                                     mat_data['E'], mat_data['nu'], d_shaft_in)
defl_out = torsional_deflection_deg(T_out_design_Nmm, shaft_len_out_mm,
                                     mat_data['E'], mat_data['nu'], d_shaft_out)
mesh_defl_sp = mesh_tooth_deflection_um(stress_res['Fn_SP'], face_width, mat_data['E'])
mesh_defl_rp = mesh_tooth_deflection_um(stress_res['Fn_RP'], face_width, mat_data['E'])
defl_in_ok  = defl_in['theta_deg']  <= 0.5
defl_out_ok = defl_out['theta_deg'] <= 0.5

# --- Thermal (computed, still shown inside overall tab) ---
power_loss_w = motor_power_design_w * (1.0 - efficiency)
thermal_res = thermal_lubrication_check(power_loss_w, est_od, face_width,
                                         ambient_c=25.0, h_conv=15.0,
                                         d_sun_mm=d_sun, n_in_rpm=n_in_rpm)

# --- Component dimensions for display/OpenSCAD ---
sun_bore_mm     = d_shaft_in
planet_bore_mm  = pin_res['d_pin_mm'] + 6.0
carrier_id_mm   = 2 * r_bore_carrier
comp_dims = {
    'sun':     {'hub_od': max(sun_bore_mm + 6.0, d_shaft_in * 1.6),
                'bore_d': sun_bore_mm},
    'planet':  {'hub_od': max(planet_bore_mm + 4.0, pin_res['d_pin_mm'] * 1.45),
                'bore_d': planet_bore_mm},
    'ring':    {'outer_d': est_od},
    'carrier': {'plate_od': carrier_plate_od,
                'plate_thickness': plate_thickness_mm,
                'hub_od': max(d_shaft_out + 8.0, d_shaft_out * 1.5),
                'output_bore': d_shaft_out + 2.0},
}

# --- Overall checks ---
overall_checks = {
    'Ring OD fits envelope':       od_fits,
    'Assembly condition':          assembly_ok,
    'Clearance condition':         clearance_ok,
    'Sun/Planet bending SF ≥ 1':   sfF_SP >= 1,
    'Ring/Planet bending SF ≥ 1':  sfF_RP >= 1,
    'Sun/Planet contact SF ≥ 1':   sfH_SP >= 1,
    'Ring/Planet contact SF ≥ 1':  sfH_RP >= 1,
    'Planet pin pressure OK':      pin_res['pressure_ok'],
    'Main bearing OK':             bearing_pass,
    'Carrier arm bending OK':      carrier_arm_res['pass'],
    'Carrier plate bending OK':    carrier_plate_res['pass'],
    'Ring rim strength OK':        ring_rim_res['pass'],
    'Sun key OK':                  key_sun_res['pass'],
    'Output key OK':               key_out_res['pass'],
    'Input shaft twist ≤ 0.5°':    defl_in_ok,
    'Output shaft twist ≤ 0.5°':   defl_out_ok,
    'Thermal steady-state ≤ 90 °C': thermal_res['temp_ok'],
}
overall_ok = all(overall_checks.values())
fails = [k for k, v in overall_checks.items() if not v]


# ================================================================
# TOP METRICS
# ================================================================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Achieved Ratio",  f"1:{ratio_actual:.3f}")
c2.metric("Motor Power (design)",  f"{motor_power_design_w:.1f} W")
c3.metric("Motor Power (operating)", f"{motor_power_operating_w:.1f} W")
c4.metric("Ring OD (est.)",  f"{est_od:.1f} mm")
c5.metric("Overall Status", "PASS ✅" if overall_ok else "CHECK ⚠️")
if fails:
    st.warning("Failing checks: " + ", ".join(fails))
st.info(f"**Teeth:** Zs={S} · Zp={P} · Zr={R}  |  "
        f"**Module:** {m_use:.2f} mm  |  "
        f"**Face width:** {face_width:.1f} mm  (auto b = {FACE_WIDTH_FACTOR:.0f}·m)  |  "
        f"**Planets:** {n_planets} (fixed)")


# ================================================================
# OUTPUT TABS
# ================================================================
tabs = st.tabs([
    "1 · Gear Geometry",
    "2 · Tooth Synthesis",
    "3 · Kinematics",
    "4 · Shafts & Pins",
    "5 · Bearings",
    "6 · Planet Load Sharing",
    "7 · Planet Pin",
    "8 · Carrier",
    "9 · Ring Rim",
    "10 · Keys / Splines",
    "11 · Component Dimensions",
    "12 · Overall Design Results",
    "13 · Parameters Glossary",
    "14 · 3D Visualisation",
    "15 · OpenSCAD",
])

# ---------------- 1 · GEAR GEOMETRY ----------------
with tabs[0]:
    st.subheader("Gear Geometry")
    geo_rows = []
    for label, g in [('Sun', geom['sun']), ('Planet', geom['planet']),
                     ('Ring', geom['ring'])]:
        geo_rows.append({
            'Gear': label, 'Teeth z': g['z'],
            'Pitch Dia (mm)':  f"{g['d_pitch']:.3f}",
            'Base Dia (mm)':   f"{g['d_base']:.3f}",
            'Tip Dia (mm)':    f"{g['d_tip']:.3f}",
            'Root Dia (mm)':   f"{g['d_root']:.3f}",
        })
    st.dataframe(pd.DataFrame(geo_rows), hide_index=True, use_container_width=True)
    st.write(f"**Module:** m = {m_use:.2f} mm  |  "
             f"**Working transverse pressure angle:** {geom['alpha_t_deg']:.2f}°  |  "
             f"**Circular pitch:** {geom['circular_pitch']:.3f} mm")
    st.write(f"**Centre distance (Sun–Planet):** {geom['center_dist_sun_planet']:.3f} mm")
    st.write(f"**Centre distance (Ring–Planet):** {geom['center_dist_ring_planet']:.3f} mm")
    st.write(f"**Face width b:** {face_width:.1f} mm  (auto = {FACE_WIDTH_FACTOR:.0f}·m)")
    st.write(f"**Ring OD (est.):** {est_od:.2f} mm ≤ {max_od_mm:.0f} mm "
             f"→ {'PASS' if od_fits else 'FAIL'}")

# ---------------- 2 · TOOTH SYNTHESIS ----------------
with tabs[1]:
    st.subheader("Tooth Synthesis")
    ts_df = pd.DataFrame([
        {'Item': 'Sun teeth',     'Value': S,   'Note': 'Input member (fixed-case dependent)'},
        {'Item': 'Planet teeth',  'Value': P,   'Note': 'Coaxial mesh'},
        {'Item': 'Ring teeth',    'Value': R,   'Note': 'R = S + 2·P (coaxiality)'},
        {'Item': 'Assembly',      'Value': 'PASS' if assembly_ok else 'FAIL',
         'Note': f'(S+R) mod N = {(S+R) % n_planets}'},
        {'Item': 'Clearance',     'Value': 'PASS' if clearance_ok else 'FAIL',
         'Note': 'Pitch-circle spacing'},
        {'Item': 'Achieved ratio', 'Value': f"{ratio_actual:.4f}",
         'Note': f"Target {target_ratio:.3f}  (err {ratio_err:.4f})"},
        {'Item': 'Mode', 'Value': 'Manual' if manual_teeth else 'Auto-search',
         'Note': '—'},
    ])
    st.dataframe(ts_df, hide_index=True, use_container_width=True)

    cand = suggest_tooth_sets(target_ratio, n_planets, max_od_mm)
    if not cand.empty:
        st.markdown("**Other valid 1:9 candidates**")
        st.dataframe(cand.head(25), hide_index=True, use_container_width=True)
    else:
        st.caption("No alternative exact-ratio candidates within the given envelope.")

# ---------------- 3 · KINEMATICS ----------------
with tabs[2]:
    st.subheader("Kinematics")
    kin_df = pd.DataFrame({
        'Parameter': [
            'Configuration', 'Input member', 'Output member',
            'Achieved ratio', 'Target ratio',
            'Input speed (rpm)', 'Output speed (rpm)',
            'Sun speed (rpm)', 'Ring speed (rpm)', 'Carrier speed (rpm)',
            'Planet spin rel. carrier (rpm)'],
        'Value': [fixed_case, input_member, output_member,
                  f"{ratio_actual:.3f}", f"{target_ratio:.2f}",
                  f"{n_in_rpm:.1f}", f"{output_speed:.2f}",
                  f"{kin['n_sun']:.2f}", f"{kin['n_ring']:.2f}",
                  f"{kin['n_carrier']:.2f}",
                  f"{kin['n_planet_spin_rel_carrier']:.2f}"]
    })
    st.dataframe(kin_df, hide_index=True, use_container_width=True)
    st.caption(f"Motor power (operating) = {motor_power_operating_w:.1f} W  |  "
               f"Motor power (design) = {motor_power_design_w:.1f} W  |  "
               f"Nominal output torque = {output_torque_nominal:.2f} N·m  |  "
               f"Design output torque = {design_out_tq:.2f} N·m")

# ---------------- 4 · SHAFTS & PINS ----------------
with tabs[3]:
    st.subheader("Shaft Sizing — ASME combined torsion + bending")
    st.caption(f"Loading condition: {shock_label} (Kb = {Kb}, Kt = {Kt}); "
               f"keyway factor Kw = {kw}. External bending assumed negligible.")
    shaft_df = pd.DataFrame([
        {'Shaft': f'Input ({input_member})',
         'Design Torque (N·m)': f"{T_in_design_Nmm/1000:.2f}",
         'Equivalent Torque Te (N·m)': f"{Te_in/1000:.2f}",
         'Required Dia (mm)': f"{d_shaft_in:.2f}"},
        {'Shaft': f'Output ({output_member})',
         'Design Torque (N·m)': f"{T_out_design_Nmm/1000:.2f}",
         'Equivalent Torque Te (N·m)': f"{Te_out/1000:.2f}",
         'Required Dia (mm)': f"{d_shaft_out:.2f}"},
    ])
    st.dataframe(shaft_df, hide_index=True, use_container_width=True)

    st.markdown("**Torsional deflection**")
    defl_df = pd.DataFrame([
        {'Shaft': 'Input',
         'Length (mm)': f"{shaft_len_in_mm:.1f}",
         'Dia (mm)': f"{d_shaft_in:.2f}",
         'G (MPa)': f"{defl_in['G_mpa']:.0f}",
         'Twist (°)': f"{defl_in['theta_deg']:.3f}",
         'Status (≤0.5°)': 'PASS' if defl_in_ok else 'FAIL'},
        {'Shaft': 'Output',
         'Length (mm)': f"{shaft_len_out_mm:.1f}",
         'Dia (mm)': f"{d_shaft_out:.2f}",
         'G (MPa)': f"{defl_out['G_mpa']:.0f}",
         'Twist (°)': f"{defl_out['theta_deg']:.3f}",
         'Status (≤0.5°)': 'PASS' if defl_out_ok else 'FAIL'},
    ])
    st.dataframe(defl_df, hide_index=True, use_container_width=True)

    st.markdown("**Approximate mesh (tooth-pair) deflection**")
    mesh_df = pd.DataFrame({
        'Mesh': ['Sun–Planet', 'Ring–Planet'],
        'Normal force (N)': [f"{stress_res['Fn_SP']:.1f}",
                              f"{stress_res['Fn_RP']:.1f}"],
        'Est. deflection (µm)': [f"{mesh_defl_sp['delta_um']:.2f}",
                                  f"{mesh_defl_rp['delta_um']:.2f}"],
    })
    st.dataframe(mesh_df, hide_index=True, use_container_width=True)

# ---------------- 5 · BEARINGS ----------------
with tabs[4]:
    st.subheader("Bearings — Main + Planet")
    st.markdown("**Main shaft bearing**")
    st.write(f"Mesh point diameter used: **{d_mesh:.2f} mm** (on {input_member})")
    mb_df = pd.DataFrame({
        'Quantity': ['Design tangential force Ft', 'Resultant radial Fr',
                     'Design load (×SF)', 'Cdyn', 'Check',
                     'L10 life (Mrev)', 'L10 life (hours)'],
        'Value': [f"{ft_design:.1f} N", f"{f_res:.1f} N",
                  f"{f_design:.1f} N", f"{cdyn:.1f} N",
                  'PASS' if bearing_pass else 'FAIL',
                  f"{main_bearing_life['L10_Mrev']:.2f}",
                  f"{main_bearing_life['L10_h']:.0f} h"]
    })
    st.dataframe(mb_df, hide_index=True, use_container_width=True)
    st.caption(f"Evaluated at {main_bearing_speed:.1f} rpm, {main_bearing_type} bearing")

    if planet_bearing_life is not None:
        st.markdown("**Planet bearing (needle roller)**")
        pb_df = pd.DataFrame({
            'Quantity': ['Resultant planet load', 'Cdyn', 'L10 life (Mrev)',
                         'L10 life (hours)', 'Spin speed rel. carrier (rpm)'],
            'Value': [f"{stress_res['F_pin']:.1f} N",
                      f"{cdyn_planet:.1f} N",
                      f"{planet_bearing_life['L10_Mrev']:.2f}",
                      f"{planet_bearing_life['L10_h']:.0f} h",
                      f"{kin['n_planet_spin_rel_carrier']:.1f}"]
        })
        st.dataframe(pb_df, hide_index=True, use_container_width=True)
    elif pin_support == 'Plain Bronze Bush':
        st.info("Plain bronze bush selected — see Planet Pin tab for pressure check.")

# ---------------- 6 · PLANET LOAD SHARING ----------------
with tabs[5]:
    st.subheader("Planet Load Sharing")
    st.write(f"**Load-sharing / imbalance factor Kp = {stress_params['Kp']:.2f}** — "
             f"applied to the worst-loaded mesh to account for unequal load "
             f"sharing between the {n_planets} planets.")
    st.write(f"Nominal equal-share torque per planet: "
             f"{T_in_design_Nmm / n_planets:.1f} N·mm")
    st.write(f"Worst-case mesh tangential force (incl. Kp): "
             f"**{stress_res['Ft_SP']:.1f} N**")
    st.caption("Real planetary trains rarely share load perfectly. Kp > 1 "
               "conservatively overloads one mesh. Increase Kp for looser "
               "manufacturing tolerances or a non-floating sun.")

# ---------------- 7 · PLANET PIN ----------------
with tabs[6]:
    st.subheader("Planet Pin Design")
    st.caption("Modelled as a simply-supported beam spanning the two carrier "
               "plates, loaded at mid-span by the resultant mesh force.")
    pin_df = pd.DataFrame({
        'Quantity': ['Resultant mesh load', 'Support span', 'Max bending moment',
                     'Support shear (each)', 'Dia. req. (bending)',
                     'Dia. req. (shear)', 'Design pin diameter',
                     'Bearing/bush pressure', 'Allowable pressure', 'Pressure check'],
        'Value': [f"{stress_res['F_pin']:.1f} N", f"{pin_span_mm:.1f} mm",
                  f"{pin_res['M_max_Nmm']:.1f} N·mm",
                  f"{pin_res['V_support_N']:.1f} N",
                  f"{pin_res['d_bend_mm']:.2f} mm",
                  f"{pin_res['d_shear_mm']:.2f} mm",
                  f"{pin_res['d_pin_mm']:.2f} mm",
                  f"{pin_res['bearing_pressure_mpa']:.2f} MPa",
                  f"{allow_bearing_pressure:.2f} MPa",
                  'PASS' if pin_res['pressure_ok'] else 'FAIL']
    })
    st.dataframe(pin_df, hide_index=True, use_container_width=True)

# ---------------- 8 · CARRIER ----------------
with tabs[7]:
    st.subheader("Carrier — Arm & Plate Bending")
    st.markdown("**Carrier arm** (cantilever beam, hub to pin centre)")
    arm_df = pd.DataFrame({
        'Quantity': ['Pin load', 'Arm length', 'Arm width', 'Arm thickness',
                     'Bending moment', 'Bending stress', 'Allowable stress',
                     'Safety factor', 'Status'],
        'Value': [f"{stress_res['F_pin']:.1f} N", f"{arm_length_mm:.1f} mm",
                  f"{arm_width_mm:.1f} mm", f"{arm_thickness_mm:.1f} mm",
                  f"{carrier_arm_res['M_Nmm']:.1f} N·mm",
                  f"{carrier_arm_res['sigma_mpa']:.1f} MPa",
                  f"{mat_data['sigma_allow_bend']:.0f} MPa",
                  f"{carrier_arm_res['sf']:.2f}",
                  'PASS' if carrier_arm_res['pass'] else 'FAIL']
    })
    st.dataframe(arm_df, hide_index=True, use_container_width=True)

    st.markdown("**Carrier plate** (sector-cantilever from bore to pin circle)")
    plate_df = pd.DataFrame({
        'Quantity': ['Pin-circle radius', 'Bore radius (incl. margin)',
                     'Plate thickness', 'Tributary width per planet',
                     'Bending moment', 'Bending stress', 'Allowable stress',
                     'Safety factor', 'Status'],
        'Value': [f"{r_pin_circle:.2f} mm", f"{r_bore_carrier:.2f} mm",
                  f"{plate_thickness_mm:.1f} mm",
                  f"{carrier_plate_res['tributary_width_mm']:.2f} mm",
                  f"{carrier_plate_res['M_Nmm']:.1f} N·mm",
                  f"{carrier_plate_res['sigma_mpa']:.1f} MPa",
                  f"{mat_data['sigma_allow_bend']:.0f} MPa",
                  f"{carrier_plate_res['sf']:.2f}",
                  'PASS' if carrier_plate_res['pass'] else 'FAIL']
    })
    st.dataframe(plate_df, hide_index=True, use_container_width=True)
    st.caption("Simplified beam-sector approximation. Full carriers should also "
               "be FEA-checked for plate torsion and pin-boss stress concentration.")

# ---------------- 9 · RING RIM ----------------
with tabs[8]:
    st.subheader("Ring Gear Rim Strength (AGMA-style YB)")
    rim_df = pd.DataFrame({
        'Quantity': ['Whole tooth depth ht', 'Rim thickness (OD − root radius)',
                     'Rim ratio mB', 'Rim-thickness factor YB',
                     'Ring σF (unadjusted)', 'Ring σF (rim-adjusted)',
                     'Allowable σF', 'Safety factor', 'Status'],
        'Value': [f"{ring_rim_res['ht_mm']:.2f} mm",
                  f"{ring_rim_res['rim_thickness_mm']:.2f} mm",
                  f"{ring_rim_res['mB']:.2f}",
                  f"{ring_rim_res['YB']:.2f}",
                  f"{stress_res['sigmaF_ring']:.1f} MPa",
                  f"{ring_rim_res['sigmaF_ring_adj_mpa']:.1f} MPa",
                  f"{mat_data['sigmaF_lim']:.0f} MPa",
                  f"{ring_rim_res['sf']:.2f}",
                  'PASS' if ring_rim_res['pass'] else 'FAIL']
    })
    st.dataframe(rim_df, hide_index=True, use_container_width=True)
    st.caption("AGMA guidance recommends mB ≥ 1.2 (rim thickness ≥ 1.2 × whole "
               "tooth depth) to avoid the ring flexing/cracking behind the teeth.")

# ---------------- 10 · KEYS / SPLINES ----------------
with tabs[9]:
    st.subheader("Keys / Splines")
    key_df = pd.DataFrame([
        {'Location': f'Sun/Input ⌀{d_shaft_in:.1f} mm',
         'Key force (N)': f"{key_sun_res['F_key_N']:.1f}",
         'Shear (MPa)': f"{key_sun_res['tau_key_mpa']:.1f}",
         'SF shear': f"{key_sun_res['sf_shear']:.2f}",
         'Bearing (MPa)': f"{key_sun_res['sigma_bearing_mpa']:.1f}",
         'SF bearing': f"{key_sun_res['sf_bearing']:.2f}",
         'Status': 'PASS' if key_sun_res['pass'] else 'FAIL'},
        {'Location': f'Output ⌀{d_shaft_out:.1f} mm',
         'Key force (N)': f"{key_out_res['F_key_N']:.1f}",
         'Shear (MPa)': f"{key_out_res['tau_key_mpa']:.1f}",
         'SF shear': f"{key_out_res['sf_shear']:.2f}",
         'Bearing (MPa)': f"{key_out_res['sigma_bearing_mpa']:.1f}",
         'SF bearing': f"{key_out_res['sf_bearing']:.2f}",
         'Status': 'PASS' if key_out_res['pass'] else 'FAIL'},
    ])
    st.dataframe(key_df, hide_index=True, use_container_width=True)
    st.caption(f"Key section: {key_width_mm:.1f} × {key_height_mm:.1f} × "
               f"{key_length_mm:.1f} mm. Shear + bearing checked.")

# ---------------- 11 · COMPONENT DIMENSIONS ----------------
with tabs[10]:
    st.subheader("All Components — Dimensions Summary")
    comp_df = pd.DataFrame([
        {'Component': 'Sun Gear',
         'Outer/Tip Dia (mm)': f"{geom['sun']['d_tip']:.2f}",
         'Root Dia (mm)': f"{geom['sun']['d_root']:.2f}",
         'Bore/Inner Dia (mm)': f"{sun_bore_mm:.2f}",
         'Face width / Height (mm)': f"{face_width:.1f}"},
        {'Component': 'Planet Gear',
         'Outer/Tip Dia (mm)': f"{geom['planet']['d_tip']:.2f}",
         'Root Dia (mm)': f"{geom['planet']['d_root']:.2f}",
         'Bore/Inner Dia (mm)': f"{planet_bore_mm:.2f}",
         'Face width / Height (mm)': f"{face_width:.1f}"},
        {'Component': 'Ring Gear',
         'Outer/Tip Dia (mm)': f"{est_od:.2f}",
         'Root Dia (mm)': f"{geom['ring']['d_root']:.2f}",
         'Bore/Inner Dia (mm)': f"{geom['ring']['d_tip']:.2f}",
         'Face width / Height (mm)': f"{face_width:.1f}"},
        {'Component': 'Carrier Plate',
         'Outer/Tip Dia (mm)': f"{carrier_plate_od:.2f}",
         'Root Dia (mm)': "—",
         'Bore/Inner Dia (mm)': f"{carrier_id_mm:.2f}",
         'Face width / Height (mm)': f"{plate_thickness_mm:.1f}"},
        {'Component': 'Carrier Arm',
         'Outer/Tip Dia (mm)': f"w={arm_width_mm:.1f}",
         'Root Dia (mm)': "—",
         'Bore/Inner Dia (mm)': "—",
         'Face width / Height (mm)': f"t={arm_thickness_mm:.1f}, L={arm_length_mm:.1f}"},
        {'Component': 'Planet Pin',
         'Outer/Tip Dia (mm)': f"{pin_res['d_pin_mm']:.2f}",
         'Root Dia (mm)': "—",
         'Bore/Inner Dia (mm)': "—",
         'Face width / Height (mm)': f"span={pin_span_mm:.1f}"},
        {'Component': 'Input Shaft',
         'Outer/Tip Dia (mm)': f"{d_shaft_in:.2f}",
         'Root Dia (mm)': "—",
         'Bore/Inner Dia (mm)': "—",
         'Face width / Height (mm)': f"L={shaft_len_in_mm:.1f}"},
        {'Component': 'Output Shaft',
         'Outer/Tip Dia (mm)': f"{d_shaft_out:.2f}",
         'Root Dia (mm)': "—",
         'Bore/Inner Dia (mm)': "—",
         'Face width / Height (mm)': f"L={shaft_len_out_mm:.1f}"},
    ])
    st.dataframe(comp_df, hide_index=True, use_container_width=True)
    st.caption("Planet bore assumes +6 mm over pin diameter for a needle bearing "
               "or bronze bush — verify against the actual bearing chosen.")

# ---------------- 12 · OVERALL DESIGN RESULTS ----------------
with tabs[11]:
    st.subheader("Overall Design Results — PASS / FAIL Dashboard")
    kpi = pd.DataFrame([
        ['Achieved ratio',      f"1:{ratio_actual:.3f}",  f"target 1:{target_ratio:.2f}"],
        ['Output speed',        f"{output_speed:.2f} rpm", "—"],
        ['Motor power (design)',    f"{motor_power_design_w:.1f} W",   "computed"],
        ['Motor power (operating)', f"{motor_power_operating_w:.1f} W","computed"],
        ['Nominal output torque',   f"{output_torque_nominal:.2f} N·m","= Tin · ratio · η"],
        ['Design output torque',    f"{design_out_tq:.2f} N·m",        "65-75 N·m band"],
        ['Face width',          f"{face_width:.1f} mm",   f"auto = {FACE_WIDTH_FACTOR:.0f}·m"],
        ['Ring OD',             f"{est_od:.1f} mm",       f"≤ {max_od_mm:.0f} mm"],
        ['Input shaft ⌀',       f"{d_shaft_in:.2f} mm",   "ASME"],
        ['Output shaft ⌀',      f"{d_shaft_out:.2f} mm",  "ASME"],
        ['Planet pin ⌀',        f"{pin_res['d_pin_mm']:.2f} mm", "bend/shear governed"],
        ['Gear bending SF',     f"{min(sfF_SP, sfF_RP):.2f}", "≥ 1 required"],
        ['Gear contact SF',     f"{min(sfH_SP, sfH_RP):.2f}", "≥ 1 required"],
        ['Main bearing L10',    f"{main_bearing_life['L10_h']:.0f} h", "—"],
        ['Planet bearing L10',
         f"{planet_bearing_life['L10_h']:.0f} h" if planet_bearing_life else "n/a", "—"],
        ['Thermal steady-state', f"{thermal_res['steady_state_temp_c']:.1f} °C", "≤ 90 °C"],
    ], columns=['Metric', 'Value', 'Notes'])
    st.dataframe(kpi, hide_index=True, use_container_width=True)

    st.markdown("### Pass / Fail Checks")
    ck_df = pd.DataFrame({'Check': list(overall_checks.keys()),
                          'Status': ['PASS ✅' if v else 'FAIL ❌'
                                     for v in overall_checks.values()]})
    st.dataframe(ck_df, hide_index=True, use_container_width=True)
    st.metric("Overall Status", "PASS ✅" if overall_ok else "CHECK REQUIRED ⚠️")
    st.info(thermal_res['lube_recommendation'])

# ---------------- 13 · PARAMETERS GLOSSARY ----------------
with tabs[12]:
    st.subheader("Parameters Glossary")
    glossary = pd.DataFrame([
        ['Zs, Zp, Zr', 'Sun / Planet / Ring tooth counts', '—', 'User or auto-searched'],
        ['m', 'Module', 'mm', 'User or auto-selected'],
        ['m_use', 'Selected module used for the design', 'mm', 'Result'],
        ['b', 'Face width', 'mm', f'Auto = {FACE_WIDTH_FACTOR:.0f}·m (no UI slider)'],
        ['α', 'Normal pressure angle', '°', f'{PRESSURE_ANGLE} (fixed)'],
        ['β', 'Helix angle', '°', f'{HELIX_ANGLE} (fixed, spur)'],
        ['η', 'Stage mesh efficiency', '—', 'User input'],
        ['Tin', 'Input torque', 'N·m', 'User slider'],
        ['nin', 'Input speed', 'rpm', 'User slider'],
        ['Tdesign', 'Design (worst-case) output torque', 'N·m', 'Locked 65–75 N·m slider'],
        ['Pmotor', 'Motor power', 'W', 'Computed from torque · speed · ratio · η'],
        ['Kp', 'Planet load-sharing factor', '—', 'Conservative mesh-overload factor'],
        ['KA', 'Application factor', '—', '1.25 (moderate shocks)'],
        ['KV', 'Dynamic factor', '—', '1.15'],
        ['KFβ, KFα', 'Face & transverse load factors (bending)', '—', '1.20 / 1.00'],
        ['KHβ, KHα', 'Face & transverse load factors (contact)', '—', '1.25 / 1.00'],
        ['YFa, YSa', 'Form & stress-correction factors (bending)', '—', 'Gear-dependent'],
        ['ZH, ZE, Zε', 'Zone / elasticity / contact-ratio factors', '—', 'ISO 6336 style'],
        ['σF', 'Root bending stress', 'MPa', 'Computed'],
        ['σH', 'Contact (pitting) stress', 'MPa', 'Computed'],
        ['Kb, Kt', 'ASME bending & torsion shock factors', '—', 'User-selected'],
        ['Kw', 'Keyway stress-concentration factor', '—', 'User input'],
        ['τ', 'Allowable shear stress', 'MPa', 'Material / user'],
        ['C_dyn', 'Bearing dynamic capacity', 'N', 'User input'],
        ['L10', 'Rated bearing life (90 % reliability)', 'Mrev / h', 'Lundberg–Palmgren'],
        ['YB, mB', 'Rim-thickness factor & rim ratio', '—', 'AGMA 2001-style'],
        ['θ', 'Angle between mesh force lines on planet', '°', 'User slider'],
        ['b/m', 'Face-width-to-module ratio', '—', f'Auto = {FACE_WIDTH_FACTOR:.0f}'],
    ], columns=['Symbol / Term', 'Meaning', 'Unit', 'Source / Value'])
    st.dataframe(glossary, hide_index=True, use_container_width=True)

# ---------------- 14 · 3D VISUALISATION ----------------
with tabs[13]:
    st.subheader("3D Visualisation — Interactive Assembly")
    st.caption("Drag to rotate · scroll to zoom · right-click to pan. "
               "Teeth shown with simplified trapezoidal profiles for visualisation.")
    fig_asm = create_assembly_3d(S, P, R, m_use, n_planets, pin_res['d_pin_mm'],
                                  face_width, d_shaft_in, d_shaft_out,
                                  comp_dims, pin_res['d_pin_mm'],
                                  geom['sun']['d_pitch'] / 2 + geom['planet']['d_pitch'] / 2)
    st.plotly_chart(fig_asm, use_container_width=True)

    st.markdown("### Individual Components")
    cols = st.columns(3)
    with cols[0]:
        st.plotly_chart(create_component_3d('Sun Gear',
            {'z': S, 'm': m_use, 'face_width': face_width}),
            use_container_width=True)
    with cols[1]:
        st.plotly_chart(create_component_3d('Planet Gear',
            {'z': P, 'm': m_use, 'face_width': face_width}),
            use_container_width=True)
    with cols[2]:
        st.plotly_chart(create_component_3d('Ring Gear',
            {'z': R, 'm': m_use, 'face_width': face_width, 'outer_d': est_od}),
            use_container_width=True)
    cols2 = st.columns(3)
    with cols2[0]:
        st.plotly_chart(create_component_3d('Planet Pin',
            {'diameter': pin_res['d_pin_mm'], 'length': pin_span_mm + 10}),
            use_container_width=True)
    with cols2[1]:
        st.plotly_chart(create_component_3d('Input Shaft',
            {'diameter': d_shaft_in, 'length': shaft_len_in_mm}),
            use_container_width=True)
    with cols2[2]:
        st.plotly_chart(create_component_3d('Output Shaft',
            {'diameter': d_shaft_out, 'length': shaft_len_out_mm}),
            use_container_width=True)

# ---------------- 15 · OPENSCAD ----------------
with tabs[14]:
    st.subheader("OpenSCAD CAD Model Generation")
    st.caption("Copy into OpenSCAD or download the .scad file. "
               "Teeth use a simplified trapezoidal approximation — for true "
               "involute profiles use the `gears.scad` library.")
    scad_code = generate_openscad_assembly(
        S, P, R, m_use, n_planets, pin_res['d_pin_mm'], face_width,
        comp_dims['sun']['hub_od'], comp_dims['sun']['bore_d'],
        comp_dims['planet']['hub_od'], comp_dims['planet']['bore_d'],
        comp_dims['ring']['outer_d'], face_width + 2.0,
        geom['sun']['d_pitch'] / 2 + geom['planet']['d_pitch'] / 2,
        plate_thickness_mm,
        comp_dims['carrier']['hub_od'], comp_dims['carrier']['output_bore'],
        d_shaft_in, d_shaft_out, pin_span_mm
    )
    st.code(scad_code, language='openscad')
    st.download_button("⬇️ Download OpenSCAD file (.scad)", scad_code,
                        "planetary_gearbox.scad", "text/plain",
                        key="scad_dl")


# ================================================================
# EXPORT — Full design summary as TXT
# ================================================================
st.divider()
with st.expander("📋 Plain-text Design Summary (copy / export)"):
    summary = f"""===== PLANETARY GEARBOX — DESIGN SUMMARY (v4) =====
Configuration       : {fixed_case}
Input / Output      : {input_member} -> {output_member}
Achieved Ratio      : {ratio_actual:.3f}   (target 1:{target_ratio:.2f})
Input torque/speed  : {t_in_nm:.3f} N·m @ {n_in_rpm:.1f} rpm
Output speed        : {output_speed:.2f} rpm
Nominal out torque  : {output_torque_nominal:.2f} N·m
Design out torque   : {design_out_tq:.2f} N·m   (65–75 N·m band)
Motor power (oper.) : {motor_power_operating_w:.1f} W   (computed)
Motor power (design): {motor_power_design_w:.1f} W   (computed)

Teeth               : S={S}  P={P}  R={R}  (mode: {'manual' if manual_teeth else 'auto'})
Module / Ring OD    : m={m_use:.2f} mm  |  OD={est_od:.1f} mm <= {max_od_mm:.0f} mm
Assembly / clearance: {'OK' if assembly_ok else 'FAIL'} / {'OK' if clearance_ok else 'FAIL'}
Face width          : {face_width:.1f} mm (auto)
Planets             : {n_planets} (fixed)

--- Gear stresses (design load) ---
Bending SP / RP     : {stress_res['sigmaF_SP']:.1f} / {stress_res['sigmaF_RP']:.1f} MPa
  SF                : {sfF_SP:.2f} / {sfF_RP:.2f}
Contact SP / RP     : {stress_res['sigmaH_SP']:.1f} / {stress_res['sigmaH_RP']:.1f} MPa
  SF                : {sfH_SP:.2f} / {sfH_RP:.2f}

--- Shafts (ASME) ---
Input shaft ⌀       : {d_shaft_in:.2f} mm   twist {defl_in['theta_deg']:.3f}°
Output shaft ⌀      : {d_shaft_out:.2f} mm  twist {defl_out['theta_deg']:.3f}°

--- Planet pin ---
Resultant load      : {stress_res['F_pin']:.1f} N
Pin diameter        : {pin_res['d_pin_mm']:.2f} mm
Bearing pressure    : {pin_res['bearing_pressure_mpa']:.2f} / {allow_bearing_pressure:.2f} MPa

--- Carrier ---
Arm SF              : {carrier_arm_res['sf']:.2f}
Plate SF            : {carrier_plate_res['sf']:.2f}

--- Ring rim ---
mB / YB             : {ring_rim_res['mB']:.2f} / {ring_rim_res['YB']:.2f}   SF={ring_rim_res['sf']:.2f}

--- Keys ---
Sun key SF  (τ/σ)   : {key_sun_res['sf_shear']:.2f} / {key_sun_res['sf_bearing']:.2f}
Output key SF (τ/σ) : {key_out_res['sf_shear']:.2f} / {key_out_res['sf_bearing']:.2f}

--- Bearings ---
Main bearing L10    : {main_bearing_life['L10_h']:.0f} h   (PASS={bearing_pass})
Planet brg L10      : {f"{planet_bearing_life['L10_h']:.0f} h" if planet_bearing_life else 'n/a'}

--- Thermal ---
Power loss          : {power_loss_w:.1f} W
Steady-state temp   : {thermal_res['steady_state_temp_c']:.1f} °C
Lubrication         : {thermal_res['lube_recommendation']}

OVERALL STATUS      : {'PASS ✅' if overall_ok else 'CHECK REQUIRED ⚠️'}
"""
    st.code(summary, language='text')

st.caption("⚠️ Preliminary design tool (ISO 6336-lite / ASME shaft code / "
           "AGMA rim factor / Lundberg–Palmgren bearing life). Verify against "
           "full standards before production release.")
