"""
================================================================================
 PLANETARY GEARBOX DESIGNER v4 - COMBINED PLATFORM
================================================================================
Single-stage 1:9 ratio planetary gearbox (Ring Fixed: Sun = input, Carrier =
output), 3 planets - both LOCKED. Input torque & input speed are user
sliders; motor power is ALWAYS a computed result (never a direct input),
derived from torque, speed, ratio and efficiency. Design (worst-case) output
torque is locked to the 65-75 N.m band.

Run with:   streamlit run app.py
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
# 1. LOCKED PROJECT CONSTANTS
# ================================================================
TARGET_RATIO = 9.0
N_PLANETS = 3
FIXED_CASE = "Ring Fixed (Sun = input, Carrier = output)"
FACE_WIDTH_FACTOR = 12.0
DESIGN_TQ_MIN, DESIGN_TQ_MAX = 65.0, 75.0
MODULES = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]

MATERIALS = {
    "17CrNiMo6 / 18CrNiMo7-6 (Case Carburized)": {
        "E": 210000.0, "nu": 0.30, "tau": 240.0, "sigmaF": 430.0, "sigmaH": 1500.0,
        "sigma_allow_bend": 380.0, "sigma_allow_bearing": 480.0, "density": 7850.0},
    "20MnCr5 / 16MnCr5 (Case Carburized)": {
        "E": 210000.0, "nu": 0.30, "tau": 140.0, "sigmaF": 380.0, "sigmaH": 1350.0,
        "sigma_allow_bend": 320.0, "sigma_allow_bearing": 400.0, "density": 7850.0},
    "EN24 / 4340 (Hardened & Tempered)": {
        "E": 206000.0, "nu": 0.30, "tau": 150.0, "sigmaF": 310.0, "sigmaH": 1150.0,
        "sigma_allow_bend": 260.0, "sigma_allow_bearing": 320.0, "density": 7850.0},
    "SAE 6150 / 51CrV4 (Spring Steel)": {
        "E": 207000.0, "nu": 0.30, "tau": 170.0, "sigmaF": 350.0, "sigmaH": 1250.0,
        "sigma_allow_bend": 290.0, "sigma_allow_bearing": 360.0, "density": 7850.0},
    "Custom": {
        "E": 200000.0, "nu": 0.30, "tau": 180.0, "sigmaF": 300.0, "sigmaH": 1200.0,
        "sigma_allow_bend": 250.0, "sigma_allow_bearing": 300.0, "density": 7850.0},
}

PARAM_GLOSSARY = {
    "Operating / Ratio": [
        ("T_in", "Input (sun) shaft torque", "N.m", "user slider", "Drives mesh loads; scaled by ratio and eta for output torque."),
        ("n_in", "Input shaft speed", "rpm", "user slider", "Sets omega_in and, with ratio, output/planet spin speeds."),
        ("i (ratio)", "Overall ratio - LOCKED", "-", "9 (fixed)", "i=(Zs+Zr)/Zs for Ring-Fixed configuration."),
        ("eta", "Mesh efficiency", "-", "0.95-0.97", "P_out = T_in*i*eta; motor power derived from this."),
        ("Design Torque", "Worst-case output torque - LOCKED band", "N.m", "65-75", "Basis for every strength check."),
        ("Motor Power", "Required input power - ALWAYS a result", "W", "computed", "P=T*omega; shown at operating and design points."),
    ],
    "Tooth Synthesis": [
        ("Zs / Zp / Zr", "Sun / Planet / Ring tooth counts", "-", "manual or auto", "Zr = Zs + 2*Zp keeps sun/planet/ring coaxial."),
        ("Assembly condition", "(Zs+Zr) mod N=0", "-", "must hold", "Lets all N planets phase into mesh simultaneously."),
        ("Clearance condition", "(Zs+Zp)*sin(180/N) > Zp+2", "-", "must hold", "Stops adjacent planet tips overlapping."),
        ("m (module)", "Tooth module", "mm", "0.75-5", "Bigger module = stronger teeth, bigger envelope."),
    ],
    "Gear Geometry": [
        ("d_pitch/base/tip/root", "Reference/base/outer/root circle diameters", "mm", "computed", "Base diameter used in force-arm calcs."),
        ("b (face width)", "Axial gear width", "mm", "12*m (fixed rule)", "Wider face spreads the tooth load thinner."),
        ("Centre distance", "Sun-Planet / Ring-Planet centre spacing", "mm", "computed", "Fixes the physical layout radius."),
    ],
    "Forces & Stress (ISO 6336-lite)": [
        ("Ft / Fr / Fn", "Tangential / radial / normal mesh force", "N", "computed", "Fn drives Hertzian contact stress."),
        ("Kp", "Planet load-sharing factor", "-", "1.05-1.15", "Inflates worst-loaded mesh for unequal load sharing."),
        ("KA/KV/KFbeta/KHbeta", "Application/dynamic/face-load factors", "-", "1.0-1.25", "De-rate nominal load for shock, speed, misalignment."),
        ("sigmaF / sigmaH", "Root bending / flank contact stress", "MPa", "computed", "Compared to material limits for safety factor."),
        ("theta (force angle)", "Angle between Sun-mesh & Ring-mesh loads on planet", "deg", "~120", "Vector-combines the two mesh loads into pin resultant."),
    ],
    "Carrier & Ring Rim": [
        ("Arm bending", "Carrier arm treated as a cantilever beam", "MPa", "computed", "M=F_pin*L; sigma=M/Z; flags undersized arm."),
        ("Plate bending", "Carrier plate sector treated as a cantilever", "MPa", "computed", "Tributary width = pin-circle circumference / N planets."),
        ("Y_B (rim factor)", "AGMA-style ring rim-thickness multiplier", "-", "1.0 if mB>=1.2", "A thin rim behind the teeth is penalised."),
    ],
    "Keys / Splines": [
        ("F_key", "Tangential force at the shaft surface", "N", "2T/d", "Drives both the key shear and bearing/crush checks."),
        ("tau_key", "Key shear stress", "MPa", "F/(w*l)", "Compared to the material's allowable shear stress."),
        ("sigma_bearing", "Key crushing/bearing stress", "MPa", "F/(0.5*h*l)", "Compared to the allowable bearing stress."),
    ],
    "Bearings & Deflection": [
        ("C_dyn", "Bearing dynamic load rating", "N", "catalogue value", "Load a bearing sustains for 1M rev at 90% survival."),
        ("L10", "Basic rating life", "10^6 rev / hours", "L10=(C/P)^p", "p=3 for ball, 10/3 for roller bearings."),
        ("theta (twist)", "Torsional shaft deflection", "deg", "T*L/(G*J)", "Checked against a practical 0.5 deg limit."),
    ],
}


# ================================================================
# 2. TOOTH SYNTHESIS
# ================================================================
def exact_ring_teeth(zs, zp):
    return zs + 2 * zp


def ratio_ring_fixed(zs, zr):
    return (zs + zr) / zs


def assembly_ok(zs, zr, nplanets):
    return (zs + zr) % nplanets == 0


def planet_spacing_ok(zs, zp, nplanets, clearance_mm=1.0):
    return (zs + zp) * math.sin(PI / nplanets) > (zp + 2.0 + clearance_mm)


def suggest_tooth_sets(target_ratio, nplanets, max_od, modules=MODULES):
    rows = []
    for zs in range(17, 61):
        for zp in range(17, 101):
            zr = exact_ring_teeth(zs, zp)
            ratio = ratio_ring_fixed(zs, zr)
            if abs(ratio - target_ratio) > 1e-9:
                continue
            if not (assembly_ok(zs, zr, nplanets) and planet_spacing_ok(zs, zp, nplanets)):
                continue
            for m in modules:
                ring_root_d = zr * m - 2.5 * m
                od = ring_root_d + 2.5 * m + 2.0 * 6.0
                if od <= max_od:
                    rows.append(dict(zs=zs, zp=zp, zr=zr, ratio=ratio, module=m, od=od,
                                     sun_pitch=zs * m, planet_pitch=zp * m, ring_pitch=zr * m))
    return pd.DataFrame(rows)


# ================================================================
# 3. GEAR GEOMETRY
# ================================================================
def gear_geometry(z, m, alpha_deg=20.0, internal=False, beta_deg=0.0):
    beta = math.radians(beta_deg)
    alpha_n = math.radians(alpha_deg)
    alpha_t = math.atan(math.tan(alpha_n) / math.cos(beta))
    d = z * m / math.cos(beta)
    db = d * math.cos(alpha_t)
    if internal:
        da = d - 2 * m / math.cos(beta)
        df = d + 2.5 * m / math.cos(beta)
    else:
        da = d + 2 * m / math.cos(beta)
        df = d - 2.5 * m / math.cos(beta)
    return dict(z=z, module=m, pitch_d=d, base_d=db, tip_d=da, root_d=df,
                addendum=m, dedendum=1.25 * m, alpha_t=math.degrees(alpha_t))


# ================================================================
# 4. KINEMATICS (Ring Fixed)
# ================================================================
def kinematics_ring_fixed(nin, ratio, zs, zp):
    ncarrier = nin / ratio
    nplanet_rel = abs(nin - ncarrier) * zs / zp
    return dict(n_sun=nin, n_ring=0.0, n_carrier=ncarrier,
                n_planet_rel_carrier=nplanet_rel, n_out=ncarrier)


# ================================================================
# 5. MESH LOADS & STRESS (ISO 6336-lite)
# ================================================================
def mesh_loads(Tin_Nm, zs, zr, nplanets, m, alpha_deg, beta_deg, Kp):
    a = math.radians(alpha_deg)
    rb_s = (zs * m / (2 * math.cos(math.radians(beta_deg)))) * math.cos(a)
    rb_r = (zr * m / (2 * math.cos(math.radians(beta_deg)))) * math.cos(a)
    Ft_sp = Tin_Nm * 1000.0 / (nplanets * rb_s) * Kp
    Ft_rp = Ft_sp * (rb_s / rb_r)
    Fr_sp, Fr_rp = Ft_sp * math.tan(a), Ft_rp * math.tan(a)
    Fn_sp, Fn_rp = Ft_sp / math.cos(a), Ft_rp / math.cos(a)
    return dict(Ft_SP=Ft_sp, Ft_RP=Ft_rp, Fr_SP=Fr_sp, Fr_RP=Fr_rp,
                Fn_SP=Fn_sp, Fn_RP=Fn_rp, rbS=rb_s, rbR=rb_r)


def gear_stress(loads, m, b, zs, zp, zr, material, alpha_deg, KA, KV, KFbeta, KFalpha,
                KHbeta, KHalpha, theta_deg):
    YFaS, YSaS = 2.8, 1.55
    YFaP_RP, YSaP_RP = 2.35, 1.60
    Yeps, Ybeta = 0.85, 1.0
    ZH, Zeps, Zbeta = 2.5, 0.90, 1.0
    Ftsp, Ftrp = loads['Ft_SP'], loads['Ft_RP']
    common = KA * KV * KFbeta * KFalpha
    sF_sp = (Ftsp * common / (b * m)) * YFaS * YSaS * Yeps * Ybeta
    sF_rp = (Ftrp * common / (b * m)) * YFaP_RP * YSaP_RP * Yeps * Ybeta
    sF_planet = math.sqrt(max(sF_sp ** 2 + sF_rp ** 2 - 2 * sF_sp * sF_rp * math.cos(math.radians(theta_deg)), 0.0))

    ZE = math.sqrt(1.0 / (PI * (2 * (1 - material['nu'] ** 2) / material['E'])))
    u_sp, u_rp = zp / zs, zr / zp
    dS, dP = zs * m, zp * m
    term_sp = (Ftsp * KA * KV * KHbeta * KHalpha) / (b * dS) * ((u_sp + 1) / u_sp)
    term_rp = (Ftrp * KA * KV * KHbeta * KHalpha) / (b * dP) * (max(u_rp - 1, 1e-9) / u_rp)
    sH_sp = ZH * ZE * Zeps * Zbeta * math.sqrt(max(term_sp, 0.0))
    sH_rp = ZH * ZE * Zeps * Zbeta * math.sqrt(max(term_rp, 0.0))

    Fnsp, Fnrp = loads['Fn_SP'], loads['Fn_RP']
    Fpin = math.sqrt(max(Fnsp ** 2 + Fnrp ** 2 - 2 * Fnsp * Fnrp * math.cos(math.radians(theta_deg)), 0.0))
    return dict(sigmaF_SP=sF_sp, sigmaF_RP=sF_rp, sigmaF_planet=sF_planet,
                sigmaH_SP=sH_sp, sigmaH_RP=sH_rp, F_pin=Fpin, ZE=ZE)


# ================================================================
# 6. SHAFT / PIN SIZING
# ================================================================
def shaft_diameter(T_Nm, M_Nm, tau_allow, Kb, Kt, Kw):
    T, M = T_Nm * 1000.0, M_Nm * 1000.0
    Te = math.sqrt((Kb * M) ** 2 + (Kt * Kw * T) ** 2)
    d = (16 * Te / (PI * tau_allow)) ** (1.0 / 3.0)
    return d, Te / 1000.0


def pin_design(F, support_span, sigma_b, tau, bearing_p):
    Mmax = F * support_span / 4.0
    V = F / 2.0
    db = (32 * Mmax / (PI * sigma_b)) ** (1.0 / 3.0)
    ds = math.sqrt(4 * V / (PI * tau))
    d = max(db, ds)
    p = F / (d * support_span)
    return dict(F_design=F, Mmax=Mmax, V=V, d_bend=db, d_shear=ds,
                d_pin=d, p_bearing=p, pressure_ok=p <= bearing_p)


def torsional_deflection_deg(T_Nmm, L_mm, E_mpa, nu, d_mm):
    G = E_mpa / (2.0 * (1.0 + nu))
    J = PI * d_mm ** 4 / 32.0
    theta_deg = math.degrees(T_Nmm * L_mm / (G * J))
    return {'G_mpa': G, 'J_mm4': J, 'theta_deg': theta_deg}


# ================================================================
# 7. BEARING LIFE
# ================================================================
def bearing_L10_life(C_dyn, P_equiv, n_rpm, bearing_type="Ball"):
    p = 3.0 if bearing_type == "Ball" else 10.0 / 3.0
    if P_equiv <= 0 or n_rpm <= 0:
        return {"L10_Mrev": float('inf'), "L10_h": float('inf')}
    L10_Mrev = (C_dyn / P_equiv) ** p
    L10_h = (L10_Mrev * 1e6) / (60.0 * n_rpm)
    return {"L10_Mrev": L10_Mrev, "L10_h": L10_h}


# ================================================================
# 8. CARRIER / RING RIM / KEYS
# ================================================================
def carrier_arm_bending(F_pin_N, arm_length_mm, arm_width_mm, arm_thickness_mm, sigma_allow_mpa):
    M = F_pin_N * arm_length_mm
    Z = arm_width_mm * arm_thickness_mm ** 2 / 6.0
    sigma = M / Z if Z > 0 else float('inf')
    sf = sigma_allow_mpa / sigma if sigma > 0 else float('inf')
    return {'M_Nmm': M, 'sigma_mpa': sigma, 'sf': sf, 'pass': sf >= 1.0}


def carrier_plate_bending(F_pin_N, n_planets, r_pin_circle_mm, r_bore_mm, plate_thickness_mm, sigma_allow_mpa):
    tributary_width = (2 * PI * r_pin_circle_mm) / n_planets
    arm = max(r_pin_circle_mm - r_bore_mm, 1e-6)
    M = F_pin_N * arm
    Z = tributary_width * plate_thickness_mm ** 2 / 6.0
    sigma = M / Z if Z > 0 else float('inf')
    sf = sigma_allow_mpa / sigma if sigma > 0 else float('inf')
    return {'M_Nmm': M, 'tributary_width_mm': tributary_width, 'sigma_mpa': sigma, 'sf': sf, 'pass': sf >= 1.0}


def ring_rim_strength(sigmaF_ring_mpa, m_use_mm, d_root_ring_mm, d_od_ring_mm, sigmaF_lim_mpa):
    ht = 2.25 * m_use_mm
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


def key_sizing(T_Nmm, d_shaft_mm, key_width_mm, key_height_mm, key_length_mm, tau_allow_mpa, sigma_allow_bearing_mpa):
    F_key = 2.0 * T_Nmm / d_shaft_mm
    tau_key = F_key / (key_width_mm * key_length_mm)
    sigma_bearing = F_key / (0.5 * key_height_mm * key_length_mm)
    sf_shear = tau_allow_mpa / tau_key if tau_key > 0 else float('inf')
    sf_bearing = sigma_allow_bearing_mpa / sigma_bearing if sigma_bearing > 0 else float('inf')
    return {'F_key_N': F_key, 'tau_key_mpa': tau_key, 'sigma_bearing_mpa': sigma_bearing,
            'sf_shear': sf_shear, 'sf_bearing': sf_bearing, 'pass': (sf_shear >= 1.0 and sf_bearing >= 1.0)}


# ================================================================
# 9. COMPONENT DIMENSIONS
# ================================================================
def component_dimensions(zs, zp, zr, m, b, d_sun, d_planet, d_ring, d_in_shaft, d_out_shaft,
                         d_pin, max_od, pin_span, nplanets):
    sun_bore = d_in_shaft + 2.0
    sun_hub_od = max(sun_bore + 6.0, d_in_shaft * 1.6)
    sun_hub_len = max(1.2 * d_in_shaft, b)
    sun_rim_thick = (sun_hub_od - sun_bore) / 2.0

    planet_bore = d_pin + 2.0
    planet_hub_od = max(planet_bore + 4.0, d_pin * 1.45)
    planet_hub_len = max(b, pin_span * 0.75)
    planet_rim_thick = (planet_hub_od - planet_bore) / 2.0

    ring_tooth_tip_d = d_ring - 2 * m
    ring_tooth_root_d = d_ring + 2.5 * m
    ring_radial_rim = max(0.18 * d_ring, 8.0)
    ring_outer_d = ring_tooth_root_d + 2 * ring_radial_rim
    ring_wall = (ring_outer_d - ring_tooth_root_d) / 2
    ring_face_width = b + 2.0

    carrier_pitch_radius = (d_sun + d_planet) / 2.0
    carrier_plate_thk = max(6.0, 0.30 * b, 0.20 * d_pin)
    carrier_od = min(max_od - 4.0, 2 * (carrier_pitch_radius + 1.6 * d_pin))
    carrier_bore = d_out_shaft + 2.0
    carrier_hub_od = max(d_out_shaft + 8.0, d_out_shaft * 1.5)
    carrier_hub_len = max(1.5 * d_out_shaft, 1.5 * b)
    carrier_pin_boss_od = d_pin + 6.0
    carrier_pin_boss_len = max(6.0, 0.5 * d_pin)
    carrier_total_height = carrier_plate_thk + carrier_hub_len + 2 * carrier_pin_boss_len

    input_shaft_len = 40.0 + b + 20.0
    output_shaft_len = 40.0 + b + 20.0 + 30.0

    pin_head_d = d_pin * 1.3
    pin_head_thick = max(3.0, 0.2 * d_pin)
    pin_total_len = pin_span + 2 * pin_head_thick

    housing_clearance = 2.0
    envelope_od = ring_outer_d + 2 * housing_clearance
    housing_wall = max(3.0, 0.06 * envelope_od)
    housing_length = input_shaft_len + b + output_shaft_len
    housing_flange_od = envelope_od + 2 * housing_wall
    housing_base_thick = max(5.0, 0.15 * housing_wall)

    return {
        'sun': {'gear_face_width': b, 'bore_d': sun_bore, 'hub_od': sun_hub_od,
                'hub_length': sun_hub_len, 'rim_thickness': sun_rim_thick, 'total_height': b + sun_hub_len},
        'planet': {'gear_face_width': b, 'bore_d': planet_bore, 'hub_od': planet_hub_od,
                   'hub_length': planet_hub_len, 'rim_thickness': planet_rim_thick, 'total_height': b + planet_hub_len},
        'ring': {'inner_tooth_tip_d': ring_tooth_tip_d, 'inner_tooth_root_d': ring_tooth_root_d,
                 'outer_d': ring_outer_d, 'wall_thickness': ring_wall, 'face_width': ring_face_width,
                 'total_height': ring_face_width},
        'carrier': {'plate_thickness': carrier_plate_thk, 'plate_od': carrier_od, 'output_bore': carrier_bore,
                    'hub_od': carrier_hub_od, 'hub_length': carrier_hub_len,
                    'planet_pin_circle_d': 2 * carrier_pitch_radius, 'pin_boss_od': carrier_pin_boss_od,
                    'pin_boss_length': carrier_pin_boss_len, 'total_height': carrier_total_height},
        'input_shaft': {'diameter': d_in_shaft, 'length': input_shaft_len, 'key_width': 0.0, 'key_depth': 0.0},
        'output_shaft': {'diameter': d_out_shaft, 'length': output_shaft_len, 'key_width': 0.0, 'key_depth': 0.0},
        'planet_pin': {'diameter': d_pin, 'span_length': pin_span, 'head_diameter': pin_head_d,
                       'head_thickness': pin_head_thick, 'total_length': pin_total_len},
        'housing': {'outer_d': envelope_od, 'wall_thickness': housing_wall, 'length': housing_length,
                    'flange_od': housing_flange_od, 'base_thickness': housing_base_thick,
                    'inner_d': envelope_od - 2 * housing_wall},
    }


# ================================================================
# 10. 3D SOLIDS - ADVANCED CAD-STYLE RENDERER
# ================================================================
def _resample_closed_polygon(poly, n):
    poly = np.asarray(poly, dtype=float)
    closed = np.vstack([poly, poly[:1]])
    seg_len = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg_len)]
    if cum[-1] <= 0:
        return np.repeat(poly[:1], n, axis=0)
    cum = cum / cum[-1]
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    x = np.interp(t, cum, closed[:, 0])
    y = np.interp(t, cum, closed[:, 1])
    return np.column_stack([x, y])


def _extrude_disk(outer_xy, z0, z1):
    outer = np.asarray(outer_xy, dtype=float)
    n = len(outer)
    cx = float(outer[:, 0].mean())
    cy = float(outer[:, 1].mean())

    vx = np.concatenate([outer[:, 0], outer[:, 0], [cx, cx]])
    vy = np.concatenate([outer[:, 1], outer[:, 1], [cy, cy]])
    vz = np.concatenate([np.full(n, z0), np.full(n, z1), [z0, z1]])

    c_bot = 2 * n
    c_top = 2 * n + 1
    I, J, K = [], [], []
    for i in range(n):
        ni = (i + 1) % n
        b = i
        t = i + n
        nb = ni
        nt = ni + n
        I.append(b); J.append(nb); K.append(nt)
        I.append(b); J.append(nt); K.append(t)
        I.append(c_bot); J.append(nb); K.append(b)
        I.append(c_top); J.append(t); K.append(nt)
    return vx, vy, vz, I, J, K


def _extrude_ring(outer_xy, inner_xy, z0, z1):
    n = max(len(outer_xy), len(inner_xy))
    n = max(n, 48)
    outer = _resample_closed_polygon(outer_xy, n)
    inner = _resample_closed_polygon(inner_xy, n)

    vx = np.concatenate([outer[:, 0], outer[:, 0], inner[:, 0], inner[:, 0]])
    vy = np.concatenate([outer[:, 1], outer[:, 1], inner[:, 1], inner[:, 1]])
    vz = np.concatenate([np.full(n, z0), np.full(n, z1),
                         np.full(n, z0), np.full(n, z1)])

    I, J, K = [], [], []
    for i in range(n):
        ni = (i + 1) % n
        ob = i
        ot = i + n
        hb = i + 2 * n
        ht = i + 3 * n
        nob = ni
        notp = ni + n
        nhb = ni + 2 * n
        nht = ni + 3 * n
        I.append(ob); J.append(nob); K.append(notp)
        I.append(ob); J.append(notp); K.append(ot)
        I.append(hb); J.append(nht); K.append(nhb)
        I.append(hb); J.append(ht); K.append(nht)
        I.append(ob); J.append(hb); K.append(nhb)
        I.append(ob); J.append(nhb); K.append(nob)
        I.append(ot); J.append(notp); K.append(nht)
        I.append(ot); J.append(nht); K.append(ht)
    return vx, vy, vz, I, J, K


def _circle_poly(r, n=64, offset=0.0):
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False) + offset
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def _involute_point(r, r_base):
    if r <= r_base:
        return np.array([r_base, 0.0])
    th = math.sqrt(max((r / r_base) ** 2 - 1.0, 0.0))
    return np.array([r_base * (math.cos(th) + th * math.sin(th)),
                     r_base * (math.sin(th) - th * math.cos(th))])


def _single_tooth_polygon(z, m, alpha_deg=20.0, points_per_flank=10):
    alpha = math.radians(alpha_deg)
    r_pitch = z * m / 2.0
    r_base = r_pitch * math.cos(alpha)
    r_tip = r_pitch + m
    r_root = r_pitch - 1.25 * m
    r_start = max(r_base, r_root)

    rs = np.linspace(r_start, r_tip, points_per_flank)
    right_flank = np.array([_involute_point(r, r_base) for r in rs])
    left_flank = np.array([[-x, y] for x, y in right_flank[::-1]])

    half_angle_root = (math.pi / (2.0 * z)) * 1.35
    root_arc = []
    for k in range(6):
        ang = -half_angle_root + 2.0 * half_angle_root * k / 5.0
        root_arc.append([r_root * math.cos(ang), r_root * math.sin(ang)])
    root_arc = np.array(root_arc)

    pts = []
    pts.append(root_arc[0])
    pts.extend(right_flank.tolist())
    pts.extend(left_flank.tolist())
    pts.append(root_arc[-1])
    for p in root_arc[::-1][1:]:
        pts.append(p)
    return np.array(pts)


def _gear_outline(z, m, alpha_deg=20.0):
    tooth = _single_tooth_polygon(z, m, alpha_deg)
    period = 2.0 * math.pi / z
    all_pts = []
    for i in range(z):
        a = i * period
        c = math.cos(a)
        s = math.sin(a)
        rot = np.column_stack([tooth[:, 0] * c - tooth[:, 1] * s,
                               tooth[:, 0] * s + tooth[:, 1] * c])
        all_pts.append(rot)
    return np.vstack(all_pts)


def _mesh_trace(mesh, color, name, opacity=1.0):
    vx, vy, vz, I, J, K = mesh
    return go.Mesh3d(
        x=vx, y=vy, z=vz, i=I, j=J, k=K,
        color=color, opacity=opacity, name=name,
        flatshading=False,
        lighting=dict(ambient=0.40, diffuse=0.85, specular=0.75,
                      roughness=0.30, fresnel=0.15),
        lightposition=dict(x=300, y=200, z=400),
    )


def _make_gear_mesh(z, m, alpha_deg, face_width, bore_dia=None):
    outline = _gear_outline(z, m, alpha_deg)
    if bore_dia is not None and bore_dia > 0:
        bore = _circle_poly(bore_dia / 2.0, 64)
        return _extrude_ring(outline, bore, 0.0, face_width)
    return _extrude_disk(outline, 0.0, face_width)


def ring_gear_solid_3d(z, m, alpha_deg, face_width, outer_dia):
    r_pitch = z * m / 2.0
    r_root = r_pitch + 1.25 * m
    outer = _circle_poly(outer_dia / 2.0, 96)
    inner = _circle_poly(r_root, 96)
    return _extrude_ring(outer, inner, 0.0, face_width)


def carrier_solid_3d(pitch_radius, d_pin, n_planets, plate_thickness,
                     hub_diameter, bore_diameter):
    plate_r = pitch_radius + d_pin * 1.5
    outer = _circle_poly(plate_r, 96)
    if bore_diameter is not None and bore_diameter > 0:
        inner = _circle_poly(bore_diameter / 2.0, 96)
        return _extrude_ring(outer, inner, 0.0, plate_thickness)
    return _extrude_disk(outer, 0.0, plate_thickness)


def shaft_solid_3d(diameter, length, offset_z=0.0):
    outline = _circle_poly(diameter / 2.0, 48)
    return _extrude_disk(outline, offset_z, offset_z + length)


def planet_pin_solid_3d(diameter, length, offset_z=0.0):
    return shaft_solid_3d(diameter, length, offset_z)


def create_assembly_3d(zs, zp, zr, m, n_planets, d_pin, face_width,
                       d_in_shaft, d_out_shaft, sun_geom, planet_geom, comp,
                       explode=0.0):
    pitch_radius = (sun_geom['pitch_d'] + planet_geom['pitch_d']) / 2.0
    traces = []
    gap = face_width * 1.6 * explode

    sun_mesh = _make_gear_mesh(int(zs), m, 20.0, face_width, comp['sun']['bore_d'])
    traces.append(_mesh_trace(sun_mesh, '#E85D2A', 'Sun Gear'))

    for k in range(n_planets):
        ang = k * 2.0 * math.pi / n_planets
        c = math.cos(ang)
        s = math.sin(ang)
        vx0, vy0, vz0, I0, J0, K0 = _make_gear_mesh(int(zp), m, 20.0, face_width,
                                                     comp['planet']['bore_d'])
        vx = vx0 * c - vy0 * s + pitch_radius * c
        vy = vx0 * s + vy0 * c + pitch_radius * s
        traces.append(_mesh_trace((vx, vy, vz0 + gap, I0, J0, K0),
                                  '#F2B01E', 'Planet %d' % (k + 1)))

    rvx, rvy, rvz, rI, rJ, rK = ring_gear_solid_3d(int(zr), m, 20.0,
                                                    face_width + 2.0,
                                                    comp['ring']['outer_d'])
    traces.append(_mesh_trace((rvx, rvy, rvz + 2.0 * gap, rI, rJ, rK),
                              '#9AA0A6', 'Ring Gear', opacity=0.45))

    cvx, cvy, cvz, cI, cJ, cK = carrier_solid_3d(pitch_radius, d_pin, n_planets,
                                                  comp['carrier']['plate_thickness'],
                                                  comp['carrier']['hub_od'],
                                                  comp['carrier']['output_bore'])
    traces.append(_mesh_trace((cvx, cvy, cvz + face_width + 3.0 * gap, cI, cJ, cK),
                              '#3F6FB5', 'Carrier', opacity=0.60))

    in_mesh = shaft_solid_3d(d_in_shaft, 40.0, offset_z=-45.0 - gap)
    traces.append(_mesh_trace(in_mesh, '#B0B0B0', 'Input Shaft'))

    out_mesh = shaft_solid_3d(d_out_shaft, 40.0 + face_width,
                               offset_z=face_width + comp['carrier']['plate_thickness'] + 4.0 * gap)
    traces.append(_mesh_trace(out_mesh, '#B0B0B0', 'Output Shaft'))

    for k in range(n_planets):
        ang = k * 2.0 * math.pi / n_planets
        c = math.cos(ang)
        s = math.sin(ang)
        pvx, pvy, pvz, pI, pJ, pK = shaft_solid_3d(d_pin,
                                                    comp['planet_pin']['total_length'],
                                                    offset_z=-2.0 + gap)
        pvx = pvx + pitch_radius * c
        pvy = pvy + pitch_radius * s
        traces.append(_mesh_trace((pvx, pvy, pvz, pI, pJ, pK),
                                  '#2E2E2E', 'Pin %d' % (k + 1)))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (mm)', backgroundcolor='#0e1117',
                       gridcolor='#2a2f3a', showbackground=True),
            yaxis=dict(title='Y (mm)', backgroundcolor='#0e1117',
                       gridcolor='#2a2f3a', showbackground=True),
            zaxis=dict(title='Z (mm)', backgroundcolor='#0e1117',
                       gridcolor='#2a2f3a', showbackground=True),
            aspectmode='data',
            camera=dict(eye=dict(x=1.6, y=1.4, z=1.0)),
        ),
        paper_bgcolor='#0e1117',
        font=dict(color='#e6e6e6'),
        title=dict(text='3D Planetary Gearbox - CAD View',
                   font=dict(size=18, color='#e6e6e6')),
        height=720,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(bgcolor='rgba(20,20,20,0.7)',
                    bordercolor='#444', borderwidth=1),
    )
    return fig


def create_component_3d_view(component_type, params):
    fig = go.Figure()

    if component_type == 'Sun Gear':
        mesh = _make_gear_mesh(params['z'], params['m'], 20.0,
                               params['face_width'], params['bore_d'])
        fig.add_trace(_mesh_trace(mesh, '#E85D2A', 'Sun Gear'))
        title = 'Sun Gear - z=%d, m=%.2f mm' % (params['z'], params['m'])

    elif component_type == 'Planet Gear':
        mesh = _make_gear_mesh(params['z'], params['m'], 20.0,
                               params['face_width'], params['bore_d'])
        fig.add_trace(_mesh_trace(mesh, '#F2B01E', 'Planet Gear'))
        title = 'Planet Gear - z=%d, m=%.2f mm' % (params['z'], params['m'])

    elif component_type == 'Ring Gear':
        mesh = ring_gear_solid_3d(params['z'], params['m'], 20.0,
                                  params['face_width'], params['outer_d'])
        fig.add_trace(_mesh_trace(mesh, '#9AA0A6', 'Ring Gear', opacity=0.55))
        title = 'Ring Gear - z=%d, m=%.2f mm' % (params['z'], params['m'])

    elif component_type == 'Carrier':
        mesh = carrier_solid_3d(params['pitch_radius'], params['d_pin'],
                                params['n_planets'], params['plate_thickness'],
                                params['hub_od'], params['bore_d'])
        fig.add_trace(_mesh_trace(mesh, '#3F6FB5', 'Carrier', opacity=0.70))
        title = 'Carrier Plate'

    elif component_type == 'Input Shaft':
        mesh = shaft_solid_3d(params['diameter'], params['length'])
        fig.add_trace(_mesh_trace(mesh, '#B0B0B0', 'Input Shaft'))
        title = 'Input Shaft - dia %.1f mm' % params['diameter']

    elif component_type == 'Output Shaft':
        mesh = shaft_solid_3d(params['diameter'], params['length'])
        fig.add_trace(_mesh_trace(mesh, '#B0B0B0', 'Output Shaft'))
        title = 'Output Shaft - dia %.1f mm' % params['diameter']

    elif component_type == 'Planet Pin':
        mesh = shaft_solid_3d(params['diameter'], params['length'])
        fig.add_trace(_mesh_trace(mesh, '#2E2E2E', 'Planet Pin'))
        title = 'Planet Pin - dia %.1f mm' % params['diameter']

    else:
        title = component_type

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (mm)', backgroundcolor='#0e1117',
                       gridcolor='#2a2f3a', showbackground=True),
            yaxis=dict(title='Y (mm)', backgroundcolor='#0e1117',
                       gridcolor='#2a2f3a', showbackground=True),
            zaxis=dict(title='Z (mm)', backgroundcolor='#0e1117',
                       gridcolor='#2a2f3a', showbackground=True),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.3, z=0.9)),
        ),
        paper_bgcolor='#0e1117',
        font=dict(color='#e6e6e6'),
        title=dict(text=title, font=dict(size=16, color='#e6e6e6')),
        height=480,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ================================================================
# 11. OPENSCAD EXPORT
# ================================================================
def generate_openscad_assembly(zs, zp, zr, m, n_planets, d_pin, b, sun_hub_od, sun_bore,
                               planet_hub_od, planet_bore, ring_outer_d, ring_face_width,
                               carrier_pitch_radius, carrier_plate_thk, carrier_hub_od, carrier_bore,
                               din, dout, pin_span, alpha_deg=20.0, show="assembly"):
    sun_pitch_d = zs * m
    sun_tip_d = sun_pitch_d + 2 * m
    sun_root_d = sun_pitch_d - 2.5 * m

    planet_pitch_d = zp * m
    planet_tip_d = planet_pitch_d + 2 * m
    planet_root_d = planet_pitch_d - 2.5 * m

    ring_pitch_d = zr * m
    ring_tip_d = ring_pitch_d - 2 * m
    ring_root_d = ring_pitch_d + 2.5 * m
    carrier_od = 2 * (carrier_pitch_radius + d_pin * 1.5)
    in_shaft_len = 40 + b + 20
    out_shaft_len = 40 + b + 20 + 30

    lines = []
    lines.append("// =====================================================")
    lines.append("// PLANETARY GEARBOX - OpenSCAD PARAMETRIC MODEL")
    lines.append("// Zs=%d Zp=%d Zr=%d m=%s mm" % (zs, zp, zr, str(m)))
    lines.append("// ratio=1:%.3f" % ((zs + zr) / float(zs)))
    lines.append("// =====================================================")
    lines.append("PI = 3.14159265358979;")
    lines.append("")
    lines.append("zs = %d;" % zs)
    lines.append("zp = %d;" % zp)
    lines.append("zr = %d;" % zr)
    lines.append("m  = %s;" % str(m))
    lines.append("alpha = %s;" % str(alpha_deg))
    lines.append("n_planets = %d;" % n_planets)
    lines.append("b  = %s;" % str(b))
    lines.append("d_pin = %s;" % str(d_pin))
    lines.append("pin_span = %s;" % str(pin_span))
    lines.append("carrier_pitch_radius = %.4f;" % carrier_pitch_radius)
    lines.append("")
    lines.append("$fn = 24;")
    lines.append("")
    lines.append("module tooth_wedge(r_in, r_out, w_in, w_out, depth) {")
    lines.append("    linear_extrude(height = depth, center = true)")
    lines.append("        polygon(points = [")
    lines.append("            [r_in,  -w_in/2],")
    lines.append("            [r_in,   w_in/2],")
    lines.append("            [r_out,  w_out/2],")
    lines.append("            [r_out, -w_out/2]")
    lines.append("        ]);")
    lines.append("}")
    lines.append("")
    lines.append("module external_gear(z, m, tip_d, root_d, width) {")
    lines.append("    pitch_r = z * m / 2;")
    lines.append("    tooth_pitch_width = (PI * pitch_r / z) * 0.9;")
    lines.append("    overlap = m * 0.5;")
    lines.append("    union() {")
    lines.append("        cylinder(h = width, d = root_d, center = true);")
    lines.append("        for (i = [0 : z - 1])")
    lines.append("            rotate([0, 0, i * 360 / z])")
    lines.append("                tooth_wedge(root_d/2 - overlap, tip_d/2, tooth_pitch_width, tooth_pitch_width * 0.55, width);")
    lines.append("    }")
    lines.append("}")
    lines.append("")
    lines.append("module ring_gear(z, m, tip_d, root_d, outer_d, width) {")
    lines.append("    pitch_r = z * m / 2;")
    lines.append("    gap_width_at_tip  = (PI * pitch_r / z) * 1.05;")
    lines.append("    gap_width_at_root = gap_width_at_tip * 1.6;")
    lines.append("    overlap = m * 0.5;")
    lines.append("    difference() {")
    lines.append("        cylinder(h = width, d = outer_d, center = true);")
    lines.append("        cylinder(h = width + 2, d = tip_d, center = true);")
    lines.append("        for (i = [0 : z - 1])")
    lines.append("            rotate([0, 0, i * 360 / z + (180 / z)])")
    lines.append("                tooth_wedge(tip_d/2 - overlap, root_d/2 + overlap, gap_width_at_tip, gap_width_at_root, width + 4);")
    lines.append("    }")
    lines.append("}")
    lines.append("")
    lines.append("module sun_gear() {")
    lines.append("    difference() {")
    lines.append("        union() {")
    lines.append("            external_gear(zs, m, %.3f, %.3f, b);" % (sun_tip_d, sun_root_d))
    lines.append("            cylinder(h = b, d = %.3f, center = true);" % sun_hub_od)
    lines.append("        }")
    lines.append("        cylinder(h = b + 10, d = %.3f, center = true);" % sun_bore)
    lines.append("    }")
    lines.append("}")
    lines.append("")
    lines.append("module planet_gear() {")
    lines.append("    difference() {")
    lines.append("        union() {")
    lines.append("            external_gear(zp, m, %.3f, %.3f, b);" % (planet_tip_d, planet_root_d))
    lines.append("            cylinder(h = b, d = %.3f, center = true);" % planet_hub_od)
    lines.append("        }")
    lines.append("        cylinder(h = b + 10, d = %.3f, center = true);" % planet_bore)
    lines.append("    }")
    lines.append("}")
    lines.append("")
    lines.append("module ring_gear_part() {")
    lines.append("    ring_gear(zr, m, %.3f, %.3f, %.3f, %.3f);" % (ring_tip_d, ring_root_d, ring_outer_d, ring_face_width))
    lines.append("}")
    lines.append("")
    lines.append("module carrier() {")
    lines.append("    difference() {")
    lines.append("        union() {")
    lines.append("            cylinder(h = %.3f, d = %.3f, center = true);" % (carrier_plate_thk, carrier_od))
    lines.append("            cylinder(h = b + 4, d = %.3f, center = true);" % carrier_hub_od)
    lines.append("        }")
    lines.append("        cylinder(h = b + 10, d = %.3f, center = true);" % carrier_bore)
    lines.append("        for (i = [0 : n_planets - 1])")
    lines.append("            rotate([0, 0, i * (360 / n_planets)])")
    lines.append("                translate([carrier_pitch_radius, 0, 0])")
    lines.append("                cylinder(h = %.3f + 6, d = d_pin + 1.0, center = true);" % carrier_plate_thk)
    lines.append("    }")
    lines.append("}")
    lines.append("")
    lines.append("module input_shaft()  { cylinder(h = %.3f,  d = %.3f,  center = true); }" % (in_shaft_len, din))
    lines.append("module output_shaft() { cylinder(h = %.3f, d = %.3f, center = true); }" % (out_shaft_len, dout))
    lines.append("module planet_pin_part() { cylinder(h = pin_span, d = d_pin, center = true); }")
    lines.append("")
    lines.append("module assembly(exploded = false) {")
    lines.append("    gap = exploded ? b * 1.5 : 0.4;")
    lines.append("    color(\"DarkOrange\") sun_gear();")
    lines.append("    color(\"DimGray\")")
    lines.append("        translate([0, 0, -((%.3f)/2 + b/2 + gap)])" % in_shaft_len)
    lines.append("        input_shaft();")
    lines.append("    for (i = [0 : n_planets - 1])")
    lines.append("        rotate([0, 0, i * (360 / n_planets)])")
    lines.append("            translate([carrier_pitch_radius, 0, 0]) {")
    lines.append("                color(\"Gold\") planet_gear();")
    lines.append("                color(\"DarkSlateGray\")")
    lines.append("                    translate([0, 0, exploded ? gap : 0])")
    lines.append("                    planet_pin_part();")
    lines.append("            }")
    lines.append("    color(\"SlateGray\", 0.55)")
    lines.append("        translate([0, 0, (exploded ? gap * 2 : gap)])")
    lines.append("        ring_gear_part();")
    lines.append("    color(\"SteelBlue\", 0.85)")
    lines.append("        translate([0, 0, b/2 + %.3f/2 + (exploded ? gap * 1.5 : gap)])" % carrier_plate_thk)
    lines.append("        carrier();")
    lines.append("    color(\"DimGray\")")
    lines.append("        translate([0, 0, b/2 + %.3f + (%.3f)/2 + (exploded ? gap * 2.5 : gap)])" % (carrier_plate_thk, out_shaft_len))
    lines.append("        output_shaft();")
    lines.append("}")
    lines.append("")
    lines.append("SHOW = \"%s\";" % show)
    lines.append("if (SHOW == \"assembly\")       assembly(false);")
    lines.append("else if (SHOW == \"exploded\")  assembly(true);")
    lines.append("else if (SHOW == \"sun\")       sun_gear();")
    lines.append("else if (SHOW == \"planet\")    planet_gear();")
    lines.append("else if (SHOW == \"ring\")      ring_gear_part();")
    lines.append("else if (SHOW == \"carrier\")   carrier();")
    lines.append("else if (SHOW == \"input_shaft\")  input_shaft();")
    lines.append("else if (SHOW == \"output_shaft\") output_shaft();")
    lines.append("else if (SHOW == \"planet_pin\")   planet_pin_part();")

    return "\n".join(lines)


# ================================================================
# 12. STREAMLIT APP
# ================================================================
st.set_page_config(page_title="Planetary Gearbox Designer v4", page_icon="⚙️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header{font-size:2.1rem;font-weight:800;color:#1f4e78;text-align:center;margin-bottom:0.3rem;}
.sub-header{font-size:0.95rem;color:#666;text-align:center;margin-bottom:1.2rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">PLANETARY GEARBOX DESIGNER v4</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Single-Stage 1:9 (Ring Fixed) - 3 Planets - Full Strength, Deflection, '
            '3D CAD & OpenSCAD Platform</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR : INPUTS ----------------
with st.sidebar:
    st.header("1. Operating Conditions")
    st.caption("Ratio: **1:%d** (single-stage, locked)  |  Planets: **%d** (locked)  |  Config: %s" % (
        int(TARGET_RATIO), N_PLANETS, FIXED_CASE))
    t_in_max = st.number_input("Max Input Torque available (N.m):", min_value=1.0, max_value=500.0, value=20.0, step=0.5)
    tin = st.slider("Input Torque (N.m):", min_value=1.0, max_value=t_in_max, value=5.0, step=0.1)
    n_in_max = st.number_input("Max Input Speed available (rpm):", min_value=100.0, max_value=20000.0, value=3000.0, step=50.0)
    nin = st.slider("Input Speed (rpm):", min_value=100.0, max_value=n_in_max, value=1500.0, step=10.0)
    eta = st.number_input("Stage Efficiency:", min_value=0.80, max_value=0.99, value=0.97, step=0.005)
    st.caption("Motor Power is **always computed** below - not a direct input.")

    st.header("2. Gear Parameters")
    alpha = st.number_input("Pressure Angle (deg):", min_value=14.5, max_value=25.0, value=20.0, step=0.5)
    beta = st.number_input("Helix Angle (deg):", min_value=0.0, max_value=30.0, value=0.0, step=1.0)
    max_od = st.number_input("Max Ring OD (mm):", min_value=50.0, max_value=500.0, value=200.0, step=1.0)

    st.header("3. Tooth Selection")
    tooth_mode = st.radio("Mode:", ["Manual", "Auto Suggest"], index=0)
    if tooth_mode == "Manual":
        zs = st.number_input("Sun Teeth Zs:", min_value=17, max_value=120, value=20, step=1)
        zp = st.number_input("Planet Teeth Zp:", min_value=17, max_value=150, value=70, step=1)
        module = st.number_input("Module (mm):", min_value=0.5, max_value=10.0, value=1.0, step=0.05)
    else:
        cand = suggest_tooth_sets(TARGET_RATIO, N_PLANETS, max_od)
        if cand.empty:
            st.error("No exact 1:9 candidates found for this OD limit. Increase Max Ring OD.")
            st.stop()
        opt = cand.sort_values(['module', 'zs', 'zp']).iloc[0]
        zs, zp, module = int(opt.zs), int(opt.zp), float(opt.module)
        st.info("Auto-selected: Zs=%d, Zp=%d, m=%.2f mm" % (zs, zp, module))
    zr = exact_ring_teeth(int(zs), int(zp))

    st.header("4. Material")
    mat_name = st.selectbox("Gear / Shaft / Pin / Carrier Material:", list(MATERIALS.keys()))
    mat = dict(MATERIALS[mat_name])
    if mat_name == "Custom":
        with st.expander("Custom Properties", expanded=True):
            mat['E'] = st.number_input("E (MPa):", 50000.0, 300000.0, 200000.0)
            mat['nu'] = st.number_input("nu:", 0.20, 0.40, 0.30, 0.01)
            mat['tau'] = st.number_input("tau (MPa):", 20.0, 500.0, 180.0)
            mat['sigmaF'] = st.number_input("sigmaF (MPa):", 50.0, 1000.0, 300.0)
            mat['sigmaH'] = st.number_input("sigmaH (MPa):", 200.0, 2000.0, 1200.0)
            mat['sigma_allow_bend'] = st.number_input("sigma_allow bend (MPa):", 50.0, 800.0, 250.0)
            mat['sigma_allow_bearing'] = st.number_input("sigma_allow bearing/crush (MPa):", 50.0, 800.0, 300.0)

    st.header("5. Design Factors")
    Kb = st.number_input("ASME Kb:", 1.0, 3.0, 1.5, 0.1)
    Kt = st.number_input("ASME Kt:", 1.0, 2.0, 1.2, 0.1)
    Kw = st.number_input("Keyway Kw:", 1.0, 2.0, 1.3, 0.05)
    Kp = st.number_input("Planet Load-Sharing Kp:", 1.00, 1.30, 1.10, 0.01)
    KA = st.number_input("Application Factor KA:", 1.0, 2.0, 1.25, 0.05)
    KV = st.number_input("Dynamic Factor KV:", 1.0, 1.5, 1.15, 0.05)
    theta = st.number_input("Force Angle theta (deg):", 60.0, 180.0, 120.0, 1.0)

    st.header("6. Design (Worst-Case) Torque - locked 65-75 N.m")
    design_out = st.slider("Design Output Torque (N.m):", DESIGN_TQ_MIN, DESIGN_TQ_MAX, 70.0, 0.5)

    st.header("7. Main Bearings (Sun / Ring shaft)")
    service = st.number_input("Bearing Service Factor:", 1.0, 3.0, 1.5, 0.1)
    cdyn_main = st.number_input("Main Bearing Dynamic Capacity Cdyn (N):", 500.0, 100000.0, 12000.0, 500.0)
    main_bearing_type = st.selectbox("Main Bearing Type:", ["Ball", "Roller"])

    st.header("8. Planet Pin Bearing")
    pin_span = st.number_input("Pin Support Span (mm):", 10.0, 100.0, 30.0, 1.0)
    pin_support = st.selectbox("Pin Support Type:", ["Needle Roller Bearing", "Plain Bronze Bush"])
    if pin_support == "Needle Roller Bearing":
        bearing_p = st.number_input("Allowable Dynamic Pressure (MPa):", 5.0, 80.0, 25.0, 1.0)
        cdyn_planet = st.number_input("Planet Bearing Dynamic Capacity Cdyn (N):", 500.0, 50000.0, 6000.0, 250.0)
    else:
        bearing_p = st.number_input("Allowable Static Bush Pressure (MPa):", 3.0, 40.0, 10.0, 1.0)
        cdyn_planet = None

    st.header("9. Carrier Geometry")
    arm_length_mm = st.number_input("Carrier Arm Length (mm):", 5.0, 200.0, 25.0, 1.0)
    arm_width_mm = st.number_input("Carrier Arm Width (mm):", 5.0, 100.0, 18.0, 1.0)
    arm_thickness_mm = st.number_input("Carrier Arm Thickness (mm):", 3.0, 60.0, 10.0, 1.0)
    plate_thickness_mm = st.number_input("Carrier Plate Thickness (mm):", 3.0, 60.0, 12.0, 1.0)
    plate_bore_margin_mm = st.number_input("Plate Bore Margin over shaft dia (mm):", 0.0, 20.0, 2.0, 0.5)

    st.header("10. Keys / Splines")
    key_width_mm = st.number_input("Key Width w (mm):", 2.0, 40.0, 6.0, 0.5)
    key_height_mm = st.number_input("Key Height h (mm):", 2.0, 40.0, 6.0, 0.5)
    key_length_mm = st.number_input("Key Length l (mm):", 5.0, 150.0, 20.0, 1.0)

    st.header("11. Shaft Length (deflection)")
    shaft_len_in_mm = st.number_input("Input Shaft Length (mm):", 10.0, 500.0, 60.0, 5.0)
    shaft_len_out_mm = st.number_input("Output Shaft Length (mm):", 10.0, 500.0, 60.0, 5.0)


# ================================================================
# CALCULATIONS
# ================================================================
sun_geom = gear_geometry(int(zs), module, alpha, False, beta)
planet_geom = gear_geometry(int(zp), module, alpha, False, beta)
ring_geom = gear_geometry(int(zr), module, alpha, True, beta)
b = FACE_WIDTH_FACTOR * module
sun_geom['face_width'] = b
planet_geom['face_width'] = b
ring_geom['face_width'] = b + 2.0

ratio_actual = ratio_ring_fixed(int(zs), int(zr))
ratio_ok = abs(ratio_actual - TARGET_RATIO) < 1e-6
output_speed = nin / ratio_actual
assembly_pass = assembly_ok(int(zs), int(zr), N_PLANETS)
clearance_pass = planet_spacing_ok(int(zs), int(zp), N_PLANETS)

motor_power_operating_w = (2.0 * PI * nin / 60.0) * tin
Tin_design = design_out / (ratio_actual * eta)
motor_power_design_w = (2.0 * PI * nin / 60.0) * Tin_design

loads = mesh_loads(Tin_design, int(zs), int(zr), N_PLANETS, module, alpha, beta, Kp)
stress = gear_stress(loads, module, b, int(zs), int(zp), int(zr), mat, alpha, KA, KV, 1.20, 1.00, 1.25, 1.00, theta)

sfF = mat['sigmaF'] / max(stress['sigmaF_planet'], 1e-9)
sfH_sp = mat['sigmaH'] / max(stress['sigmaH_SP'], 1e-9)
sfH_rp = mat['sigmaH'] / max(stress['sigmaH_RP'], 1e-9)

din, Te_in = shaft_diameter(Tin_design, 0.0, mat['tau'], Kb, Kt, Kw)
dout, Te_out = shaft_diameter(design_out, 0.0, mat['tau'], Kb, Kt, Kw)
pin = pin_design(stress['F_pin'], pin_span, mat['sigma_allow_bend'], mat['tau'], bearing_p)

defl_in = torsional_deflection_deg(Tin_design * 1000.0, shaft_len_in_mm, mat['E'], mat['nu'], din)
defl_out = torsional_deflection_deg(design_out * 1000.0, shaft_len_out_mm, mat['E'], mat['nu'], dout)
defl_in_ok = defl_in['theta_deg'] <= 0.5
defl_out_ok = defl_out['theta_deg'] <= 0.5

main_bearing_radial = math.hypot(loads['Fr_SP'], loads['Ft_SP']) * service
main_bearing_pass = main_bearing_radial <= cdyn_main
main_bearing_life = bearing_L10_life(cdyn_main, main_bearing_radial / max(service, 1e-9), nin, main_bearing_type)

kin = kinematics_ring_fixed(nin, ratio_actual, int(zs), int(zp))
if pin_support == "Needle Roller Bearing" and cdyn_planet:
    planet_bearing_life = bearing_L10_life(cdyn_planet, stress['F_pin'], max(kin['n_planet_rel_carrier'], 1e-6), "Roller")
else:
    planet_bearing_life = None

comp = component_dimensions(int(zs), int(zp), int(zr), module, b, sun_geom['pitch_d'], planet_geom['pitch_d'],
                            ring_geom['pitch_d'], din, dout, pin['d_pin'], max_od, pin_span, N_PLANETS)
comp['carrier']['plate_thickness'] = plate_thickness_mm
comp['input_shaft']['length'] = shaft_len_in_mm
comp['output_shaft']['length'] = shaft_len_out_mm

r_pin_circle = sun_geom['pitch_d'] / 2 + planet_geom['pitch_d'] / 2
r_bore_carrier = (dout / 2.0) + plate_bore_margin_mm
carrier_arm_res = carrier_arm_bending(stress['F_pin'], arm_length_mm, arm_width_mm, arm_thickness_mm, mat['sigma_allow_bend'])
carrier_plate_res = carrier_plate_bending(stress['F_pin'], N_PLANETS, r_pin_circle, r_bore_carrier,
                                          plate_thickness_mm, mat['sigma_allow_bend'])

ring_rim_res = ring_rim_strength(stress['sigmaF_RP'], module, ring_geom['root_d'], comp['ring']['outer_d'], mat['sigmaF'])

key_sun_res = key_sizing(Tin_design * 1000.0, din, key_width_mm, key_height_mm, key_length_mm, mat['tau'], mat['sigma_allow_bearing'])
key_out_res = key_sizing(design_out * 1000.0, dout, key_width_mm, key_height_mm, key_length_mm, mat['tau'], mat['sigma_allow_bearing'])

overall_checks = {
    'Ratio = 1:9 achieved': ratio_ok,
    'Assembly condition': assembly_pass,
    'Clearance condition': clearance_pass,
    'Ring OD fits envelope': comp['ring']['outer_d'] <= max_od,
    'Combined planet bending SF>=1': sfF >= 1,
    'Sun/Planet contact SF>=1': sfH_sp >= 1,
    'Ring/Planet contact SF>=1': sfH_rp >= 1,
    'Planet pin pressure OK': pin['pressure_ok'],
    'Main bearing capacity OK': main_bearing_pass,
    'Carrier arm bending OK': carrier_arm_res['pass'],
    'Carrier plate bending OK': carrier_plate_res['pass'],
    'Ring rim strength OK': ring_rim_res['pass'],
    'Sun key OK': key_sun_res['pass'],
    'Output key OK': key_out_res['pass'],
    'Input shaft twist <=0.5 deg': defl_in_ok,
    'Output shaft twist <=0.5 deg': defl_out_ok,
}
overall_ok = all(overall_checks.values())


# ================================================================
# TOP METRICS
# ================================================================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ratio", "1:%.3f" % ratio_actual)
c2.metric("Output Speed", "%.1f rpm" % output_speed)
c3.metric("Motor Power (operating)", "%.1f W" % motor_power_operating_w)
c4.metric("Motor Power (design pt)", "%.1f W" % motor_power_design_w)
c5.metric("Overall Status", "PASS" if overall_ok else "CHECK")
st.info("**Teeth:** Zs=%d, Zp=%d, Zr=%d  |  **Module:** %.2f mm  |  **Face Width:** %.1f mm  |  **Design Torque:** %.1f N.m" % (
    zs, zp, zr, module, b, design_out))


# ================================================================
# 15 OUTPUT TABS
# ================================================================
tabs = st.tabs([
    "1 Gear Geometry", "2 Tooth Synthesis", "3 Kinematics", "4 Shafts & Pins", "5 Bearings",
    "6 Planet Load Sharing", "7 Planet Pin", "8 Carrier", "9 Ring Rim", "10 Keys/Splines",
    "11 Component Dimensions", "12 Overall Design Results", "13 Parameter Glossary",
    "14 3D Visualization", "15 OpenSCAD"
])

with tabs[0]:
    st.subheader("Gear Geometry")
    geo_df = pd.DataFrame([
        ["Sun", zs, sun_geom['pitch_d'], sun_geom['base_d'], sun_geom['tip_d'], sun_geom['root_d'], b],
        ["Planet", zp, planet_geom['pitch_d'], planet_geom['base_d'], planet_geom['tip_d'], planet_geom['root_d'], b],
        ["Ring", zr, ring_geom['pitch_d'], ring_geom['base_d'], ring_geom['tip_d'], ring_geom['root_d'], b + 2.0],
    ], columns=["Gear", "Teeth", "Pitch (mm)", "Base (mm)", "Tip (mm)", "Root (mm)", "Face Width (mm)"])
    st.dataframe(geo_df, hide_index=True, use_container_width=True)
    st.write("Working transverse pressure angle: %.2f deg" % sun_geom['alpha_t'])
    st.write("Centre Distance (Sun-Planet): %.3f mm" % ((sun_geom['pitch_d'] + planet_geom['pitch_d']) / 2))
    st.write("Centre Distance (Ring-Planet): %.3f mm" % ((ring_geom['pitch_d'] - planet_geom['pitch_d']) / 2))

with tabs[1]:
    st.subheader("Tooth Synthesis")
    tooth_df = pd.DataFrame([
        ["Sun", str(zs), "Input"], ["Planet", str(zp), "Meshing"], ["Ring", str(zr), "Fixed"],
        ["Zs + 2*Zp = Zr", "%d+2(%d) = %d" % (zs, zp, zs + 2 * zp), "Coaxiality"],
        ["Ratio achieved", "%.4f" % ratio_actual, "PASS" if ratio_ok else "FAIL"],
        ["Assembly", "PASS" if assembly_pass else "FAIL", "(Zs+Zr) mod N = 0"],
        ["Clearance", "PASS" if clearance_pass else "FAIL", "planet tip spacing"],
        ["Module", "%.2f mm" % module, "-"],
    ], columns=["Item", "Value", "Meaning/Status"])
    st.dataframe(tooth_df, hide_index=True, use_container_width=True)
    cand = suggest_tooth_sets(TARGET_RATIO, N_PLANETS, max_od)
    if not cand.empty:
        st.markdown("**Other exact 1:9 candidates within the OD limit:**")
        st.dataframe(cand.sort_values(['module', 'zs']).head(25), hide_index=True, use_container_width=True)

with tabs[2]:
    st.subheader("Kinematics")
    st.dataframe(pd.DataFrame([
        ["Sun (input)", kin['n_sun'], "rpm"], ["Ring (fixed)", 0.0, "rpm"],
        ["Carrier (output)", kin['n_carrier'], "rpm"],
        ["Planet spin rel. carrier", kin['n_planet_rel_carrier'], "rpm"],
    ], columns=["Member", "Speed", "Unit"]), hide_index=True, use_container_width=True)
    st.caption("Output speed = Input speed / ratio = %.1f / %.3f = %.2f rpm" % (nin, ratio_actual, output_speed))

with tabs[3]:
    st.subheader("Shafts & Pins")
    st.dataframe(pd.DataFrame([
        ["Input Shaft (Sun)", Tin_design, 0.0, Te_in, din, defl_in['theta_deg'], 'PASS' if defl_in_ok else 'FAIL'],
        ["Output Shaft (Carrier)", design_out, 0.0, Te_out, dout, defl_out['theta_deg'], 'PASS' if defl_out_ok else 'FAIL'],
    ], columns=["Shaft", "Design Torque (N.m)", "Bending Moment (N.m)", "Equiv. Torque Te (N.m)",
                "Required Dia (mm)", "Torsional Twist (deg)", "Status"]),
        hide_index=True, use_container_width=True)
    st.markdown("**Planet Pin - sizing summary**")
    st.dataframe(pd.DataFrame([
        ["Resultant Pin Load", pin['F_design'], "N"], ["Max Bending Moment", pin['Mmax'], "N.mm"],
        ["Support Shear Force", pin['V'], "N"], ["Dia from Bending", pin['d_bend'], "mm"],
        ["Dia from Shear", pin['d_shear'], "mm"], ["Design Pin Diameter", pin['d_pin'], "mm"],
    ], columns=["Quantity", "Value", "Unit"]), hide_index=True, use_container_width=True)

with tabs[4]:
    st.subheader("Bearings")
    st.markdown("**Main Bearing (Sun / Ring shaft)**")
    st.dataframe(pd.DataFrame([
        ["Resultant Radial Load (xSF)", "%.1f" % main_bearing_radial, "N"],
        ["Dynamic Capacity Cdyn", "%.1f" % cdyn_main, "N"],
        ["Check", "PASS" if main_bearing_pass else "FAIL", "-"],
        ["L10 Life", "%.1f Mrev / %.0f h" % (main_bearing_life['L10_Mrev'], main_bearing_life['L10_h']), "-"],
    ], columns=["Quantity", "Value", "Unit"]), hide_index=True, use_container_width=True)
    st.markdown("**Planet Pin Bearing / Bush**")
    if planet_bearing_life is not None:
        st.dataframe(pd.DataFrame([
            ["Resultant Pin Load", "%.1f" % stress['F_pin'], "N"],
            ["Dynamic Capacity Cdyn", "%.1f" % cdyn_planet, "N"],
            ["Relative Spin Speed", "%.1f" % kin['n_planet_rel_carrier'], "rpm"],
            ["L10 Life", "%.1f Mrev / %.0f h" % (planet_bearing_life['L10_Mrev'], planet_bearing_life['L10_h']), "-"],
        ], columns=["Quantity", "Value", "Unit"]), hide_index=True, use_container_width=True)
    else:
        st.write("Plain bronze bush - bearing pressure = %.2f MPa vs allowable %.2f MPa -> %s" % (
            pin['p_bearing'], bearing_p, 'PASS' if pin['pressure_ok'] else 'FAIL'))

with tabs[5]:
    st.subheader("Planet Load Sharing")
    st.write("**Load-sharing/imbalance factor Kp:** %.2f" % Kp)
    st.write("**Application factor KA:** %.2f  |  **Dynamic factor KV:** %.2f" % (KA, KV))
    st.write("**Worst-case Sun-Planet tangential force:** %.1f N" % loads['Ft_SP'])
    st.write("**Theoretical equal-share per planet:** %.1f N.mm portion of input torque" % (Tin_design * 1000 / N_PLANETS))
    st.caption("Real planetary trains rarely share load perfectly between the 3 planets.")

with tabs[6]:
    st.subheader("Planet Pin - Full Design Check")
    st.dataframe(pd.DataFrame({
        'Quantity': ['Resultant Mesh Load on Pin', 'Support Span', 'Max Bending Moment', 'Support Shear (each)',
                     'Dia required (bending)', 'Dia required (shear)', 'Design Pin Diameter',
                     'Bearing/Bush Pressure', 'Allowable Pressure', 'Pressure Check'],
        'Value': ["%.1f N" % stress['F_pin'], "%.1f mm" % pin_span, "%.1f N.mm" % pin['Mmax'],
                  "%.1f N" % pin['V'], "%.2f mm" % pin['d_bend'], "%.2f mm" % pin['d_shear'],
                  "%.2f mm" % pin['d_pin'], "%.2f MPa" % pin['p_bearing'], "%.2f MPa" % bearing_p,
                  'PASS' if pin['pressure_ok'] else 'FAIL']
    }), hide_index=True, use_container_width=True)

with tabs[7]:
    st.subheader("Carrier - Arm & Plate Bending")
    st.markdown("**Carrier Arm**")
    st.dataframe(pd.DataFrame({
        'Quantity': ['Resultant Pin Load', 'Arm Length', 'Arm Width', 'Arm Thickness',
                     'Bending Moment', 'Bending Stress', 'Allowable Stress', 'Safety Factor', 'Status'],
        'Value': ["%.1f N" % stress['F_pin'], "%.1f mm" % arm_length_mm, "%.1f mm" % arm_width_mm,
                  "%.1f mm" % arm_thickness_mm, "%.1f N.mm" % carrier_arm_res['M_Nmm'],
                  "%.1f MPa" % carrier_arm_res['sigma_mpa'], "%.0f MPa" % mat['sigma_allow_bend'],
                  "%.2f" % carrier_arm_res['sf'], 'PASS' if carrier_arm_res['pass'] else 'FAIL']
    }), hide_index=True, use_container_width=True)
    st.markdown("**Carrier Plate**")
    st.dataframe(pd.DataFrame({
        'Quantity': ['Pin Circle Radius', 'Bore Radius (+margin)', 'Plate Thickness', 'Tributary Width/Planet',
                     'Bending Moment', 'Bending Stress', 'Allowable Stress', 'Safety Factor', 'Status'],
        'Value': ["%.2f mm" % r_pin_circle, "%.2f mm" % r_bore_carrier, "%.1f mm" % plate_thickness_mm,
                  "%.2f mm" % carrier_plate_res['tributary_width_mm'], "%.1f N.mm" % carrier_plate_res['M_Nmm'],
                  "%.1f MPa" % carrier_plate_res['sigma_mpa'], "%.0f MPa" % mat['sigma_allow_bend'],
                  "%.2f" % carrier_plate_res['sf'], 'PASS' if carrier_plate_res['pass'] else 'FAIL']
    }), hide_index=True, use_container_width=True)

with tabs[8]:
    st.subheader("Ring Gear Rim Strength")
    st.dataframe(pd.DataFrame({
        'Quantity': ['Whole Tooth Depth ht', 'Rim Thickness', 'Rim Ratio mB',
                     'Rim Thickness Factor YB', 'Ring sigmaF (unadjusted)', 'Ring sigmaF (rim-adjusted)',
                     'Allowable sigmaF', 'Safety Factor', 'Status'],
        'Value': ["%.2f mm" % ring_rim_res['ht_mm'], "%.2f mm" % ring_rim_res['rim_thickness_mm'],
                  "%.2f" % ring_rim_res['mB'], "%.2f" % ring_rim_res['YB'], "%.1f MPa" % stress['sigmaF_RP'],
                  "%.1f MPa" % ring_rim_res['sigmaF_ring_adj_mpa'], "%.0f MPa" % mat['sigmaF'],
                  "%.2f" % ring_rim_res['sf'], 'PASS' if ring_rim_res['pass'] else 'FAIL']
    }), hide_index=True, use_container_width=True)

with tabs[9]:
    st.subheader("Keys / Splines")
    st.dataframe(pd.DataFrame([
        {'Location': 'Sun/Input shaft (dia %.1fmm)' % din, 'Key Force (N)': "%.1f" % key_sun_res['F_key_N'],
         'Shear (MPa)': "%.1f" % key_sun_res['tau_key_mpa'], 'SF Shear': "%.2f" % key_sun_res['sf_shear'],
         'Bearing (MPa)': "%.1f" % key_sun_res['sigma_bearing_mpa'], 'SF Bearing': "%.2f" % key_sun_res['sf_bearing'],
         'Status': 'PASS' if key_sun_res['pass'] else 'FAIL'},
        {'Location': 'Output/Carrier shaft (dia %.1fmm)' % dout, 'Key Force (N)': "%.1f" % key_out_res['F_key_N'],
         'Shear (MPa)': "%.1f" % key_out_res['tau_key_mpa'], 'SF Shear': "%.2f" % key_out_res['sf_shear'],
         'Bearing (MPa)': "%.1f" % key_out_res['sigma_bearing_mpa'], 'SF Bearing': "%.2f" % key_out_res['sf_bearing'],
         'Status': 'PASS' if key_out_res['pass'] else 'FAIL'},
    ]), hide_index=True, use_container_width=True)

with tabs[10]:
    st.subheader("All Component Dimensions")
    rows = []
    for cname, dims in comp.items():
        for k, v in dims.items():
            rows.append([cname.replace('_', ' ').title(), k.replace('_', ' ').title(), "%.2f" % v, "mm"])
    st.dataframe(pd.DataFrame(rows, columns=["Component", "Dimension", "Value", "Unit"]),
                 hide_index=True, use_container_width=True)

with tabs[11]:
    st.subheader("Overall Design Results")
    summary = pd.DataFrame([
        ["Input Torque", "%.2f" % tin, "N.m"], ["Input Speed", "%.0f" % nin, "rpm"],
        ["Motor Power (operating point)", "%.1f" % motor_power_operating_w, "W"],
        ["Motor Power (design point)", "%.1f" % motor_power_design_w, "W"],
        ["Efficiency", "%.3f" % eta, "-"], ["Achieved Ratio", "%.4f" % ratio_actual, "-"],
        ["Output Speed", "%.2f" % output_speed, "rpm"], ["Design Output Torque", "%.1f" % design_out, "N.m"],
        ["Module", "%.2f" % module, "mm"], ["Face Width", "%.1f" % b, "mm"],
        ["Ring Envelope OD", "%.1f" % comp['ring']['outer_d'], "mm"],
        ["Input Shaft Dia", "%.2f" % din, "mm"], ["Output Shaft Dia", "%.2f" % dout, "mm"],
        ["Planet Pin Dia", "%.2f" % pin['d_pin'], "mm"],
        ["Combined Planet Bending SF", "%.2f" % sfF, "-"],
        ["Sun/Planet Contact SF", "%.2f" % sfH_sp, "-"], ["Ring/Planet Contact SF", "%.2f" % sfH_rp, "-"],
        ["Carrier Arm Bending SF", "%.2f" % carrier_arm_res['sf'], "-"],
        ["Carrier Plate Bending SF", "%.2f" % carrier_plate_res['sf'], "-"],
        ["Ring Rim SF", "%.2f" % ring_rim_res['sf'], "-"],
    ], columns=["Parameter", "Value", "Unit"])
    st.dataframe(summary, hide_index=True, use_container_width=True)

    st.markdown("**Consolidated PASS/FAIL Checks**")
    check_df = pd.DataFrame([(k, "PASS" if v else "FAIL") for k, v in overall_checks.items()],
                            columns=["Design Check", "Status"])
    st.dataframe(check_df, hide_index=True, use_container_width=True)
    st.metric("Overall Design Status", "PASS" if overall_ok else "CHECK REQUIRED")
    if overall_ok:
        st.balloons()

    summary_text = "PLANETARY GEARBOX DESIGN SUMMARY\n"
    summary_text += "Configuration: %s\n" % FIXED_CASE
    summary_text += "Ratio: 1:%.3f\n" % ratio_actual
    summary_text += "Teeth: Zs=%d Zp=%d Zr=%d  Module=%.2fmm\n" % (zs, zp, zr, module)
    summary_text += "Overall Status: %s\n" % ('PASS' if overall_ok else 'CHECK REQUIRED')
    st.download_button("Download Design Summary (TXT)", summary_text, "planetary_gearbox_summary.txt", "text/plain")

with tabs[12]:
    st.subheader("Parameter Glossary")
    for category, entries in PARAM_GLOSSARY.items():
        with st.expander(category, expanded=False):
            st.dataframe(pd.DataFrame(entries, columns=["Symbol", "Meaning", "Unit", "Typical Value", "Role"]),
                         hide_index=True, use_container_width=True)

with tabs[13]:
    st.subheader("3D CAD Visualization")
    st.caption("Drag to orbit. Scroll to zoom. Right-drag to pan. Double-click to reset.")

    explode = st.slider("Explode assembly", 0.0, 1.0, 0.0, 0.05,
                        help="0 = fully assembled, 1 = fully exploded along Z axis")

    fig_asm = create_assembly_3d(int(zs), int(zp), int(zr), module, N_PLANETS,
                                 pin['d_pin'], b, din, dout,
                                 sun_geom, planet_geom, comp, explode=explode)
    st.plotly_chart(fig_asm, use_container_width=True)

    st.markdown("### Individual Components")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(create_component_3d_view('Sun Gear', {
            'z': int(zs), 'm': module, 'face_width': b,
            'hub_od': comp['sun']['hub_od'], 'bore_d': comp['sun']['bore_d']
        }), use_container_width=True)
    with c2:
        st.plotly_chart(create_component_3d_view('Planet Gear', {
            'z': int(zp), 'm': module, 'face_width': b,
            'hub_od': comp['planet']['hub_od'], 'bore_d': comp['planet']['bore_d']
        }), use_container_width=True)
    with c3:
        st.plotly_chart(create_component_3d_view('Ring Gear', {
            'z': int(zr), 'm': module, 'face_width': b + 2.0,
            'outer_d': comp['ring']['outer_d']
        }), use_container_width=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.plotly_chart(create_component_3d_view('Carrier', {
            'pitch_radius': (sun_geom['pitch_d'] + planet_geom['pitch_d']) / 2,
            'd_pin': pin['d_pin'], 'n_planets': N_PLANETS,
            'plate_thickness': plate_thickness_mm,
            'hub_od': comp['carrier']['hub_od'],
            'bore_d': comp['carrier']['output_bore']
        }), use_container_width=True)
    with c5:
        st.plotly_chart(create_component_3d_view('Input Shaft', {
            'diameter': din, 'length': comp['input_shaft']['length']
        }), use_container_width=True)
    with c6:
        st.plotly_chart(create_component_3d_view('Output Shaft', {
            'diameter': dout, 'length': comp['output_shaft']['length']
        }), use_container_width=True)

    st.plotly_chart(create_component_3d_view('Planet Pin', {
        'diameter': pin['d_pin'], 'length': comp['planet_pin']['total_length']
    }), use_container_width=True)

with tabs[14]:
    st.subheader("OpenSCAD CAD Model")
    scad_view = st.selectbox("View to generate:", [
        "assembly", "exploded", "sun", "planet", "ring", "carrier",
        "input_shaft", "output_shaft", "planet_pin"
    ], index=0)
    openscad_code = generate_openscad_assembly(
        zs, zp, zr, module, N_PLANETS, pin['d_pin'], b,
        comp['sun']['hub_od'], comp['sun']['bore_d'], comp['planet']['hub_od'], comp['planet']['bore_d'],
        comp['ring']['outer_d'], comp['ring']['face_width'], r_pin_circle, plate_thickness_mm,
        comp['carrier']['hub_od'], comp['carrier']['output_bore'], din, dout, pin_span,
        alpha_deg=alpha, show=scad_view)
    st.code(openscad_code, language="openscad")
    st.download_button("Download OpenSCAD File (.scad)", openscad_code,
                       "planetary_gearbox_%s.scad" % scad_view, "text/plain", key="download_openscad")


st.divider()
with st.expander("Pastable Input Configuration (JSON)"):
    config_json = json.dumps({
        "input_torque_nm": tin, "input_speed_rpm": nin, "efficiency": eta,
        "design_output_torque_nm": design_out, "target_ratio_locked": TARGET_RATIO,
        "n_planets_locked": N_PLANETS, "pressure_angle_deg": alpha, "helix_angle_deg": beta,
        "max_od_mm": max_od, "sun_teeth": int(zs), "planet_teeth": int(zp), "module_mm": module,
        "material": mat_name, "asme_kb": Kb, "asme_kt": Kt, "keyway_kw": Kw, "Kp": Kp,
        "bearing_service_factor": service, "pin_span_mm": pin_span,
        "allowable_pin_pressure_mpa": bearing_p, "force_angle_deg": theta,
    }, indent=2)
    st.code(config_json, language="json")

st.caption("Simplified sizing tool (ISO 6336-lite / ASME shaft code / AGMA-style rim factor / "
           "Lundberg-Palmgren bearing life). Verify against full standards before production release.")
