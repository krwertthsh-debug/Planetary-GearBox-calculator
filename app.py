"""
================================================================================
 PLANETARY GEARBOX DESIGNER v4 — COMBINED PLATFORM
================================================================================
Single-stage 1:9 ratio planetary gearbox (Ring Fixed: Sun = input, Carrier =
output), 3 planets — both LOCKED. Input torque & input speed are user
sliders; motor power is ALWAYS a computed result (never a direct input),
derived from torque, speed, ratio and efficiency. Design (worst-case) output
torque is locked to the 65-75 N.m band.

INPUT SECTIONS (sidebar):
  Operating Conditions -> Gear Parameters -> Tooth Selection (manual/auto) ->
  Material -> Design Factors -> Design (worst-case) Torque [65-75 N.m] ->
  Main Bearings (Sun/Ring shaft) -> Planet Pin Bearing -> Carrier Geometry ->
  Keys/Splines -> Shaft Length (deflection) -> Layout

OUTPUT TABS:
  1 Gear Geometry | 2 Tooth Synthesis | 3 Kinematics | 4 Shafts & Pins |
  5 Bearings | 6 Planet Load Sharing | 7 Planet Pin | 8 Carrier | 9 Ring Rim |
  10 Keys/Splines | 11 All Component Dimensions | 12 Overall Design Results |
  13 Parameter Glossary | 14 3D Visualization (CAD-style) | 15 OpenSCAD

Run with:   streamlit run planetary_gearbox_app.py
Requires :  streamlit, numpy, matplotlib, pandas, plotly
--------------------------------------------------------------------------------
NOTE ON ENGINEERING RIGOUR
Simplified/representative formulas throughout (ISO 6336-lite, ASME shaft
code, AGMA-style rim factor, Lundberg-Palmgren bearing life, basic
beam-bending approximations for the carrier). First-pass sizing / learning
aid — verify against full ISO 6336 / AGMA 2001 / ISO 281 before production.
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
TARGET_RATIO = 9.0      # single-stage, LOCKED
N_PLANETS = 3            # LOCKED
FIXED_CASE = "Ring Fixed (Sun = input, Carrier = output)"
FACE_WIDTH_FACTOR = 12.0   # b = FACE_WIDTH_FACTOR * module — fixed design rule
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
        ("T_in", "Input (sun) shaft torque", "N·m", "user slider", "Drives mesh loads; scaled by ratio & η for output torque."),
        ("n_in", "Input shaft speed", "rpm", "user slider", "Sets ω_in and, with ratio, output/planet spin speeds."),
        ("i (ratio)", "Overall ratio — LOCKED", "—", "9 (fixed)", "i=(Zs+Zr)/Zs for Ring-Fixed configuration."),
        ("η", "Mesh efficiency", "—", "0.95–0.97", "P_out = T_in·i·η; motor power derived from this, never entered directly."),
        ("Design Torque", "Worst-case output torque — LOCKED band", "N·m", "65–75", "Basis for every strength check (shafts, gears, pin, carrier, keys)."),
        ("Motor Power", "Required input power — ALWAYS a result", "W", "computed", "P=T·ω; shown at both the chosen operating point and the design point."),
    ],
    "Tooth Synthesis": [
        ("Zs / Zp / Zr", "Sun / Planet / Ring tooth counts", "—", "manual or auto", "Zr = Zs + 2·Zp keeps sun/planet/ring coaxial."),
        ("Assembly condition", "(Zs+Zr) mod N=0", "—", "must hold", "Lets all N planets phase into mesh simultaneously."),
        ("Clearance condition", "(Zs+Zp)·sin(180°/N) > Zp+2", "—", "must hold", "Stops adjacent planet tips overlapping."),
        ("m (module)", "Tooth module", "mm", "0.75–5", "Bigger module = stronger teeth, bigger envelope."),
    ],
    "Gear Geometry": [
        ("d_pitch/base/tip/root", "Reference/base/outer/root circle diameters", "mm", "computed", "Base diameter used in force-arm calcs; root governs bending stress."),
        ("b (face width)", "Axial gear width", "mm", f"{FACE_WIDTH_FACTOR:.0f}×m (fixed rule)", "Wider face spreads the tooth load thinner."),
        ("Centre distance", "Sun-Planet / Ring-Planet centre spacing", "mm", "computed", "Fixes the physical layout radius."),
    ],
    "Forces & Stress (ISO 6336-lite)": [
        ("Ft / Fr / Fn", "Tangential / radial / normal mesh force", "N", "computed", "Fn drives Hertzian contact stress; Ft drives root bending."),
        ("Kp", "Planet load-sharing factor", "—", "1.05–1.15", "Inflates the worst-loaded mesh for unequal load sharing."),
        ("KA/KV/KFβ/KHβ", "Application/dynamic/face-load factors", "—", "1.0–1.25", "De-rate the nominal load for shock, speed and misalignment."),
        ("σF / σH", "Root bending / flank contact stress", "MPa", "computed", "Compared to material σF_lim / σH_lim for the safety factor."),
        ("θ (force angle)", "Angle between Sun-mesh & Ring-mesh loads on the planet", "deg", "~120", "Vector-combines the two mesh loads into the planet-pin resultant."),
    ],
    "Carrier & Ring Rim": [
        ("Arm bending", "Carrier arm treated as a cantilever beam", "MPa", "computed", "M=F_pin·L; σ=M/Z; flags an undersized arm section."),
        ("Plate bending", "Carrier plate sector treated as a cantilever", "MPa", "computed", "Tributary width = pin-circle circumference / N planets."),
        ("Y_B (rim factor)", "AGMA-style ring rim-thickness multiplier", "—", "1.0 if mB≥1.2", "A thin rim behind the teeth is penalised with extra stress."),
    ],
    "Keys / Splines": [
        ("F_key", "Tangential force at the shaft surface", "N", "2T/d", "Drives both the key shear and bearing/crush checks."),
        ("τ_key", "Key shear stress", "MPa", "F/(w·l)", "Compared to the material's allowable shear stress."),
        ("σ_bearing", "Key crushing/bearing stress", "MPa", "F/(0.5·h·l)", "Compared to the material's allowable bearing stress."),
    ],
    "Bearings & Deflection": [
        ("C_dyn", "Bearing dynamic load rating", "N", "catalogue value", "Load a bearing sustains for 1M rev at 90% survival."),
        ("L10", "Basic rating life", "10⁶ rev / hours", "L10=(C/P)^p", "p=3 for ball, 10/3 for roller bearings."),
        ("θ (twist)", "Torsional shaft deflection", "deg", "T·L/(G·J)", "Checked against a practical 0.5°/shaft-length limit."),
    ],
}

# ================================================================
# 2. TOOTH SYNTHESIS
# ================================================================
def exact_ring_teeth(zs: int, zp: int) -> int:
    return zs + 2 * zp

def ratio_ring_fixed(zs: int, zr: int) -> float:
    return (zs + zr) / zs

def assembly_ok(zs: int, zr: int, nplanets: int) -> bool:
    return (zs + zr) % nplanets == 0

def planet_spacing_ok(zs: int, zp: int, nplanets: int, clearance_mm: float = 1.0) -> bool:
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
# 5. MESH LOADS & STRESS  (ISO 6336-lite)
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
    d = (16 * Te / (PI * tau_allow)) ** (1 / 3)
    return d, Te / 1000.0

def pin_design(F, support_span, sigma_b, tau, bearing_p):
    Mmax = F * support_span / 4.0
    V = F / 2.0
    db = (32 * Mmax / (PI * sigma_b)) ** (1 / 3)
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
# 10. 3D SOLIDS (true involute) + OPENSCAD
# ================================================================
def involute_tooth_curve(z, m, alpha_deg=20.0, points_per_flank=8):
    alpha = math.radians(alpha_deg)
    r_pitch = z * m / 2.0
    r_base = r_pitch * math.cos(alpha)

    def involute_point(r):
        theta = math.sqrt(max((r / r_base) ** 2 - 1.0, 0.0))
        x = r_base * (math.cos(theta) + theta * math.sin(theta))
        y = r_base * (math.sin(theta) - theta * math.cos(theta))
        return x, y

    r_start = max(r_base, r_pitch - 1.25 * m)
    r_end = r_pitch + m
    r_vals = np.linspace(r_start, r_end, points_per_flank)
    left_flank = [involute_point(r) for r in r_vals]
    right_flank = [(-x, y) for x, y in reversed(left_flank)]

    tip_start_angle = math.atan2(left_flank[-1][1], left_flank[-1][0])
    tip_end_angle = math.atan2(right_flank[0][1], right_flank[0][0])
    tip_angles = np.linspace(tip_start_angle, tip_end_angle, 4)
    tip_points = [(r_end * math.cos(a), r_end * math.sin(a)) for a in tip_angles]

    root_start_angle = math.atan2(right_flank[-1][1], right_flank[-1][0]) + 2 * PI / z
    root_end_angle = math.atan2(left_flank[0][1], left_flank[0][0]) + 2 * PI / z
    root_angles = np.linspace(root_start_angle, root_end_angle, 4)
    root_points = [(r_start * math.cos(a), r_start * math.sin(a)) for a in root_angles]

    all_points = right_flank + tip_points[1:] + left_flank[1:] + root_points[1:]
    return np.array(all_points)

def full_gear_outline(z, m, alpha_deg=20.0):
    tooth_pts = involute_tooth_curve(z, m, alpha_deg)
    period = 2 * PI / z
    all_x, all_y = [], []
    for i in range(z):
        angle = i * period
        c, s = math.cos(angle), math.sin(angle)
        for px, py in tooth_pts:
            all_x.append(px * c - py * s)
            all_y.append(px * s + py * c)
    return np.array(all_x), np.array(all_y)

def _ring_cap(n, x, y, z0, z1):
    vx, vy = np.concatenate([x, x]), np.concatenate([y, y])
    vz = np.concatenate([np.full(n, z0), np.full(n, z1)])
    I, J, K = [], [], []
    for i in range(n):
        ni = (i + 1) % n
        I += [i, i]; J += [ni, ni + n]; K += [ni + n, i + n]
    return vx, vy, vz, I, J, K

def gear_solid_3d(z, m, alpha_deg, face_width, hub_dia=None, bore_dia=None):
    x, y = full_gear_outline(z, m, alpha_deg)
    if hub_dia is not None:
        hub_r = hub_dia / 2.0
        hub_angles = np.linspace(0, 2 * PI, 20, endpoint=False)
        hub_x, hub_y = hub_r * np.cos(hub_angles), hub_r * np.sin(hub_angles)
        if bore_dia is not None:
            bore_r = bore_dia / 2.0
            bore_x = bore_r * np.cos(hub_angles[::-1])
            bore_y = bore_r * np.sin(hub_angles[::-1])
            x = np.concatenate([x, hub_x, bore_x]); y = np.concatenate([y, hub_y, bore_y])
    n = len(x)
    z0, z1 = 0.0, face_width
    vx, vy = np.concatenate([x, x]), np.concatenate([y, y])
    vz = np.concatenate([np.full(n, z0), np.full(n, z1)])
    I, J, K = [], [], []
    for i in range(n):
        ni = (i + 1) % n
        I.append(i); J.append(ni); K.append(ni + n)
        I.append(i); J.append(ni + n); K.append(i + n)
        I.append(2 * n); J.append(i); K.append(ni)
        I.append(2 * n + 1); J.append(ni + n); K.append(i + n)
    vx = np.concatenate([vx, [0.0, 0.0]]); vy = np.concatenate([vy, [0.0, 0.0]])
    vz = np.concatenate([vz, [z0, z1]])
    return vx, vy, vz, I, J, K

def ring_gear_solid_3d(z, m, alpha_deg, face_width, outer_dia):
    x, y = full_gear_outline(z, m, alpha_deg)
    outer_r = outer_dia / 2.0
    outer_angles = np.linspace(0, 2 * PI, 30, endpoint=False)
    outer_x, outer_y = outer_r * np.cos(outer_angles), outer_r * np.sin(outer_angles)
    x, y = np.concatenate([x, outer_x]), np.concatenate([y, outer_y])
    return _ring_cap(len(x), x, y, 0.0, face_width)

def carrier_solid_3d(pitch_radius, d_pin, n_planets, plate_thickness, hub_diameter, bore_diameter):
    plate_radius = pitch_radius + d_pin * 1.5
    angles = np.linspace(0, 2 * PI, 40, endpoint=False)
    x, y = plate_radius * np.cos(angles), plate_radius * np.sin(angles)
    for i in range(n_planets):
        ba = i * 2 * PI / n_planets
        br = d_pin / 2.0 + 2.0
        bx = pitch_radius * math.cos(ba) + br * np.cos(angles)
        by = pitch_radius * math.sin(ba) + br * np.sin(angles)
        x, y = np.concatenate([x, bx]), np.concatenate([y, by])
    hub_r = hub_diameter / 2.0
    hub_x, hub_y = hub_r * np.cos(angles), hub_r * np.sin(angles)
    x, y = np.concatenate([x, hub_x]), np.concatenate([y, hub_y])
    if bore_diameter:
        bore_r = bore_diameter / 2.0
        bore_x, bore_y = bore_r * np.cos(angles[::-1]), bore_r * np.sin(angles[::-1])
        x, y = np.concatenate([x, bore_x]), np.concatenate([y, bore_y])
    return _ring_cap(len(x), x, y, 0.0, plate_thickness)

def shaft_solid_3d(diameter, length):
    radius = diameter / 2.0
    angles = np.linspace(0, 2 * PI, 32, endpoint=False)
    x, y = radius * np.cos(angles), radius * np.sin(angles)
    return _ring_cap(len(x), x, y, 0.0, length)

def planet_pin_solid_3d(diameter, length):
    return shaft_solid_3d(diameter, length)

def _mesh3d(mesh, color, name, opacity=0.9):
    vx, vy, vz, I, J, K = mesh
    return go.Mesh3d(x=vx, y=vy, z=vz, i=I, j=J, k=K, color=color, opacity=opacity,
                      name=name, flatshading=True,
                      lighting=dict(ambient=0.45, diffuse=0.75, specular=0.5, roughness=0.35, fresnel=0.15),
                      lightposition=dict(x=200, y=200, z=250))

def create_assembly_3d(zs, zp, zr, m, n_planets, d_pin, face_width, d_in_shaft, d_out_shaft,
                        sun_geom, planet_geom, comp):
    pitch_radius = (sun_geom['pitch_d'] + planet_geom['pitch_d']) / 2.0
    traces = [_mesh3d(gear_solid_3d(zs, m, 20.0, face_width, comp['sun']['hub_od'], comp['sun']['bore_d']),
                       '#D9531E', 'Sun Gear')]
    for k in range(n_planets):
        angle = k * 2 * PI / n_planets
        vx, vy, vz, I, J, K = gear_solid_3d(zp, m, 20.0, face_width, comp['planet']['hub_od'], comp['planet']['bore_d'])
        c, s = math.cos(angle), math.sin(angle)
        vx = vx + pitch_radius * c; vy = vy + pitch_radius * s
        traces.append(_mesh3d((vx, vy, vz, I, J, K), '#EDB120', f'Planet {k+1}'))
    traces.append(_mesh3d(ring_gear_solid_3d(zr, m, 20.0, face_width + 2.0, comp['ring']['outer_d']),
                           '#8a8a8a', 'Ring Gear', 0.55))
    traces.append(_mesh3d(carrier_solid_3d(pitch_radius, d_pin, n_planets, comp['carrier']['plate_thickness'],
                                            comp['carrier']['hub_od'], comp['carrier']['output_bore']),
                           '#4C72B0', 'Carrier', 0.6))
    traces.append(_mesh3d(shaft_solid_3d(d_in_shaft, 40.0), '#8C8C8C', 'Input Shaft'))
    traces.append(_mesh3d(shaft_solid_3d(d_out_shaft, 40.0 + face_width), '#8C8C8C', 'Output Shaft'))
    for k in range(n_planets):
        angle = k * 2 * PI / n_planets
        vx, vy, vz, I, J, K = planet_pin_solid_3d(d_pin, comp['planet_pin']['total_length'])
        c, s = math.cos(angle), math.sin(angle)
        vx = vx + pitch_radius * c; vy = vy + pitch_radius * s
        traces.append(_mesh3d((vx, vy, vz, I, J, K), '#3B3B3B', f'Pin {k+1}'))

    fig = go.Figure(data=traces)
    fig.update_layout(scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
                                  aspectmode='data', camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))),
                       title='3D Planetary Gearbox Assembly', height=700, margin=dict(l=0, r=0, t=40, b=0))
    return fig

def create_component_3d_view(component_type, params):
    fig = go.Figure()
    if component_type == 'Sun Gear':
        mesh = gear_solid_3d(params['z'], params['m'], 20.0, params['face_width'], params['hub_od'], params['bore_d'])
        fig.add_trace(_mesh3d(mesh, '#D9531E', 'Sun Gear'))
        title = f"Sun Gear — z={params['z']}, m={params['m']:.2f}mm"
    elif component_type == 'Planet Gear':
        mesh = gear_solid_3d(params['z'], params['m'], 20.0, params['face_width'], params['hub_od'], params['bore_d'])
        fig.add_trace(_mesh3d(mesh, '#EDB120', 'Planet Gear'))
        title = f"Planet Gear — z={params['z']}, m={params['m']:.2f}mm"
    elif component_type == 'Ring Gear':
        mesh = ring_gear_solid_3d(params['z'], params['m'], 20.0, params['face_width'], params['outer_d'])
        fig.add_trace(_mesh3d(mesh, '#8a8a8a', 'Ring Gear', 0.55))
        title = f"Ring Gear — z={params['z']}, m={params['m']:.2f}mm"
    elif component_type == 'Carrier':
        mesh = carrier_solid_3d(params['pitch_radius'], params['d_pin'], params['n_planets'],
                                 params['plate_thickness'], params['hub_od'], params['bore_d'])
        fig.add_trace(_mesh3d(mesh, '#4C72B0', 'Carrier', 0.6))
        title = "Carrier Plate"
    elif component_type == 'Input Shaft':
        fig.add_trace(_mesh3d(shaft_solid_3d(params['diameter'], params['length']), '#8C8C8C', 'Input Shaft'))
        title = f"Input Shaft — ⌀{params['diameter']:.1f}mm"
    elif component_type == 'Output Shaft':
        fig.add_trace(_mesh3d(shaft_solid_3d(params['diameter'], params['length']), '#8C8C8C', 'Output Shaft'))
        title = f"Output Shaft — ⌀{params['diameter']:.1f}mm"
    elif component_type == 'Planet Pin':
        fig.add_trace(_mesh3d(planet_pin_solid_3d(params['diameter'], params['length']), '#3B3B3B', 'Planet Pin'))
        title = f"Planet Pin — ⌀{params['diameter']:.1f}mm"
    else:
        title = component_type
    fig.update_layout(scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
                                  aspectmode='data', camera=dict(eye=dict(x=1.3, y=1.3, z=0.7))),
                       title=title, height=460, margin=dict(l=0, r=0, t=40, b=0))
    return fig

def generate_openscad_assembly(zs, zp, zr, m, n_planets, d_pin, b, sun_hub_od, sun_bore,
                                planet_hub_od, planet_bore, ring_outer_d, ring_face_width,
                                carrier_pitch_radius, carrier_plate_thk, carrier_hub_od, carrier_bore,
                                din, dout, pin_span):
    sun_pitch_d = zs * m; sun_tip_d = sun_pitch_d + 2 * m; sun_root_d = sun_pitch_d - 2.5 * m
    planet_pitch_d = zp * m; planet_tip_d = planet_pitch_d + 2 * m; planet_root_d = planet_pitch_d - 2.5 * m
    ring_pitch_d = zr * m; ring_tip_d = ring_pitch_d - 2 * m; ring_root_d = ring_pitch_d + 2.5 * m
    carrier_od = 2 * (carrier_pitch_radius + d_pin * 1.5)
    in_shaft_len = 40 + b + 20
    out_shaft_len = 40 + b + 20 + 30

    return f"""// =====================================================
// PLANETARY GEARBOX - OpenSCAD GENERATED CODE (v4)
// =====================================================
zs = {zs}; zp = {zp}; zr = {zr}; m = {m}; n_planets = {n_planets};
b = {b}; d_pin = {d_pin}; pin_span = {pin_span};

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
                points = [[-tooth_width/2,0,-width/2],[tooth_width/2,0,-width/2],
                          [tooth_width_top/2,0,width/2],[-tooth_width_top/2,0,width/2]],
                faces = [[0,1,2,3]]);
    }}
}}

module sun_gear() {{
    difference() {{
        union() {{
            gear_approx(z={zs}, m={m}, width={b}, tip_d={sun_tip_d:.2f}, root_d={sun_root_d:.2f});
            cylinder(h={b}, d={sun_hub_od:.2f}, center=true);
        }}
        cylinder(h={b + 10}, d={sun_bore:.2f}, center=true);
    }}
}}

module planet_gear() {{
    difference() {{
        union() {{
            gear_approx(z={zp}, m={m}, width={b}, tip_d={planet_tip_d:.2f}, root_d={planet_root_d:.2f});
            cylinder(h={b}, d={planet_hub_od:.2f}, center=true);
        }}
        cylinder(h={b + 10}, d={planet_bore:.2f}, center=true);
    }}
}}

module ring_gear() {{
    difference() {{
        cylinder(h={ring_face_width}, d={ring_outer_d:.2f}, center=true);
        cylinder(h={ring_face_width + 2}, d={ring_root_d:.2f}, center=true);
        for (i = [0 : {zr - 1}])
            rotate([0, 0, i * 360 / {zr}])
                translate([0, {ring_root_d/2 - m:.2f}, 0])
                cube([{m}, {m * 1.5}, {ring_face_width}], center=true);
    }}
}}

module carrier() {{
    difference() {{
        union() {{
            cylinder(h={carrier_plate_thk}, d={carrier_od:.2f}, center=true);
            cylinder(h={carrier_plate_thk * 2}, d={carrier_hub_od:.2f}, center=true);
            for (i = [0 : {n_planets - 1}])
                rotate([0, 0, i * 360 / {n_planets}])
                    translate([{carrier_pitch_radius:.2f}, 0, 0])
                    cylinder(h={carrier_plate_thk + 10}, d={d_pin + 6:.2f}, center=true);
        }}
        cylinder(h={carrier_plate_thk * 3}, d={carrier_bore:.2f}, center=true);
        for (i = [0 : {n_planets - 1}])
            rotate([0, 0, i * 360 / {n_planets}])
                translate([{carrier_pitch_radius:.2f}, 0, 0])
                cylinder(h={carrier_plate_thk + 20}, d={d_pin:.2f}, center=true);
    }}
}}

module input_shaft() {{ cylinder(h={in_shaft_len}, d={din:.2f}, center=true); }}
module output_shaft() {{ cylinder(h={out_shaft_len}, d={dout:.2f}, center=true); }}
module planet_pin() {{ cylinder(h={pin_span + 10}, d={d_pin:.2f}, center=true); }}

module assembly() {{
    translate([0, 0, 0]) sun_gear();
    for (i = [0 : {n_planets - 1}])
        rotate([0, 0, i * 360 / {n_planets}])
            translate([{carrier_pitch_radius:.2f}, 0, 0])
            planet_gear();
    ring_gear();
    carrier();
    translate([0, 0, -{in_shaft_len/2 + b/2:.2f}]) input_shaft();
    translate([0, 0, {carrier_plate_thk/2 + b/2:.2f}]) output_shaft();
    for (i = [0 : {n_planets - 1}])
        rotate([0, 0, i * 360 / {n_planets}])
            translate([{carrier_pitch_radius:.2f}, 0, 0])
            planet_pin();
}}

assembly();
"""

# ================================================================
# 11. STREAMLIT APP
# ================================================================
st.set_page_config(page_title="Planetary Gearbox Designer v4", page_icon="⚙️", layout="wide",
                    initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header{font-size:2.1rem;font-weight:800;color:#1f4e78;text-align:center;margin-bottom:0.3rem;}
.sub-header{font-size:0.95rem;color:#666;text-align:center;margin-bottom:1.2rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚙️ PLANETARY GEARBOX DESIGNER v4</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Single-Stage 1:9 (Ring Fixed) — 3 Planets — Full Strength, Deflection, '
            '3D CAD & OpenSCAD Platform</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR : INPUTS ----------------
with st.sidebar:
    st.header("1. Operating Conditions")
    st.caption(f"Ratio: **1:{TARGET_RATIO:.0f}** (single-stage, locked)  |  Planets: **{N_PLANETS}** (locked)  |  "
               f"Config: {FIXED_CASE}")
    t_in_max = st.number_input("Max Input Torque available (N·m):", min_value=1.0, max_value=500.0, value=20.0, step=0.5)
    tin = st.slider("Input Torque (N·m):", min_value=1.0, max_value=t_in_max, value=5.0, step=0.1)
    n_in_max = st.number_input("Max Input Speed available (rpm):", min_value=100.0, max_value=20000.0, value=3000.0, step=50.0)
    nin = st.slider("Input Speed (rpm):", min_value=100.0, max_value=n_in_max, value=1500.0, step=10.0)
    eta = st.number_input("Stage Efficiency:", min_value=0.80, max_value=0.99, value=0.97, step=0.005)
    st.caption("Motor Power is **always computed** below — not a direct input.")

    st.header("2. Gear Parameters")
    alpha = st.number_input("Pressure Angle (°):", min_value=14.5, max_value=25.0, value=20.0, step=0.5)
    beta = st.number_input("Helix Angle (°):", min_value=0.0, max_value=30.0, value=0.0, step=1.0)
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
        st.info(f"Auto-selected: Zs={zs}, Zp={zp}, m={module:.2f} mm")
    zr = exact_ring_teeth(int(zs), int(zp))

    st.header("4. Material")
    mat_name = st.selectbox("Gear / Shaft / Pin / Carrier Material:", list(MATERIALS.keys()))
    mat = dict(MATERIALS[mat_name])
    if mat_name == "Custom":
        with st.expander("Custom Properties", expanded=True):
            mat['E'] = st.number_input("E (MPa):", 50000.0, 300000.0, 200000.0)
            mat['nu'] = st.number_input("ν:", 0.20, 0.40, 0.30, 0.01)
            mat['tau'] = st.number_input("τ (MPa):", 20.0, 500.0, 180.0)
            mat['sigmaF'] = st.number_input("σF (MPa):", 50.0, 1000.0, 300.0)
            mat['sigmaH'] = st.number_input("σH (MPa):", 200.0, 2000.0, 1200.0)
            mat['sigma_allow_bend'] = st.number_input("σ_allow bend (MPa):", 50.0, 800.0, 250.0)
            mat['sigma_allow_bearing'] = st.number_input("σ_allow bearing/crush (MPa):", 50.0, 800.0, 300.0)

    st.header("5. Design Factors")
    Kb = st.number_input("ASME Kb:", 1.0, 3.0, 1.5, 0.1)
    Kt = st.number_input("ASME Kt:", 1.0, 2.0, 1.2, 0.1)
    Kw = st.number_input("Keyway Kw:", 1.0, 2.0, 1.3, 0.05)
    Kp = st.number_input("Planet Load-Sharing Kp:", 1.00, 1.30, 1.10, 0.01)
    KA = st.number_input("Application Factor KA:", 1.0, 2.0, 1.25, 0.05)
    KV = st.number_input("Dynamic Factor KV:", 1.0, 1.5, 1.15, 0.05)
    theta = st.number_input("Force Angle θ (°):", 60.0, 180.0, 120.0, 1.0)

    st.header("6. Design (Worst-Case) Torque — locked 65-75 N·m")
    design_out = st.slider("Design Output Torque (N·m):", DESIGN_TQ_MIN, DESIGN_TQ_MAX, 70.0, 0.5)

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

    st.header("12. Layout")
    t_sun_phase = st.slider("Sun Rotation (visual only, rad):", 0.0, 2 * PI, 0.0, 0.05)

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

# Motor power — ALWAYS computed
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

od_est = 2 * (ring_geom['pitch_d'] / 2 + max(0.18 * ring_geom['pitch_d'], 8.0)) + 2.5 * module
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
# user overrides
comp['carrier']['plate_thickness'] = plate_thickness_mm
comp['input_shaft']['length'] = shaft_len_in_mm
comp['output_shaft']['length'] = shaft_len_out_mm
comp['input_shaft']['key_width'] = key_width_mm
comp['input_shaft']['key_depth'] = key_height_mm
comp['output_shaft']['key_width'] = key_width_mm
comp['output_shaft']['key_depth'] = key_height_mm

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
    'Combined planet bending SF≥1': sfF >= 1,
    'Sun/Planet contact SF≥1': sfH_sp >= 1,
    'Ring/Planet contact SF≥1': sfH_rp >= 1,
    'Planet pin pressure OK': pin['pressure_ok'],
    'Main bearing capacity OK': main_bearing_pass,
    'Carrier arm bending OK': carrier_arm_res['pass'],
    'Carrier plate bending OK': carrier_plate_res['pass'],
    'Ring rim strength OK': ring_rim_res['pass'],
    'Sun key OK': key_sun_res['pass'],
    'Output key OK': key_out_res['pass'],
    'Input shaft twist ≤0.5°': defl_in_ok,
    'Output shaft twist ≤0.5°': defl_out_ok,
}
overall_ok = all(overall_checks.values())

# ================================================================
# TOP METRICS
# ================================================================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ratio", f"1:{ratio_actual:.3f}")
c2.metric("Output Speed", f"{output_speed:.1f} rpm")
c3.metric("Motor Power (operating)", f"{motor_power_operating_w:.1f} W")
c4.metric("Motor Power (design pt)", f"{motor_power_design_w:.1f} W")
c5.metric("Overall Status", "PASS ✅" if overall_ok else "CHECK ⚠️")
st.info(f"**Teeth:** Zs={zs}, Zp={zp}, Zr={zr}  |  **Module:** {module:.2f} mm  |  "
        f"**Face Width (fixed rule, {FACE_WIDTH_FACTOR:.0f}×m):** {b:.1f} mm  |  "
        f"**Design Torque:** {design_out:.1f} N·m")

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
    st.write(f"Working transverse pressure angle: {sun_geom['alpha_t']:.2f}°")
    st.write(f"Centre Distance (Sun-Planet): {(sun_geom['pitch_d']+planet_geom['pitch_d'])/2:.3f} mm")
    st.write(f"Centre Distance (Ring-Planet): {(ring_geom['pitch_d']-planet_geom['pitch_d'])/2:.3f} mm")

with tabs[1]:
    st.subheader("Tooth Synthesis")
    tooth_df = pd.DataFrame([
        ["Sun", str(zs), "Input"], ["Planet", str(zp), "Meshing"], ["Ring", str(zr), "Fixed"],
        ["Zs + 2·Zp = Zr", f"{zs}+2({zp}) = {zs+2*zp}", "Coaxiality"],
        ["Ratio achieved", f"{ratio_actual:.4f}", "PASS" if ratio_ok else "FAIL (not exactly 1:9)"],
        ["Assembly", "PASS" if assembly_pass else "FAIL", "(Zs+Zr) mod N = 0"],
        ["Clearance", "PASS" if clearance_pass else "FAIL", "planet tip spacing"],
        ["Module", f"{module:.2f} mm", "—"],
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
    st.caption(f"Output speed = Input speed / ratio = {nin:.1f} / {ratio_actual:.3f} = {output_speed:.2f} rpm")

with tabs[3]:
    st.subheader("Shafts & Pins")
    st.dataframe(pd.DataFrame([
        ["Input Shaft (Sun)", Tin_design, 0.0, Te_in, din, defl_in['theta_deg'], 'PASS' if defl_in_ok else 'FAIL'],
        ["Output Shaft (Carrier)", design_out, 0.0, Te_out, dout, defl_out['theta_deg'], 'PASS' if defl_out_ok else 'FAIL'],
    ], columns=["Shaft", "Design Torque (N·m)", "Bending Moment (N·m)", "Equiv. Torque Te (N·m)",
                "Required Dia (mm)", "Torsional Twist (deg)", "Twist ≤0.5° Status"]),
    hide_index=True, use_container_width=True)
    st.markdown("**Planet Pin — sizing summary**")
    st.dataframe(pd.DataFrame([
        ["Resultant Pin Load", pin['F_design'], "N"], ["Max Bending Moment", pin['Mmax'], "N·mm"],
        ["Support Shear Force", pin['V'], "N"], ["Dia from Bending", pin['d_bend'], "mm"],
        ["Dia from Shear", pin['d_shear'], "mm"], ["Design Pin Diameter", pin['d_pin'], "mm"],
    ], columns=["Quantity", "Value", "Unit"]), hide_index=True, use_container_width=True)

with tabs[4]:
    st.subheader("Bearings")
    st.markdown("**Main Bearing (Sun / Ring shaft)**")
    st.dataframe(pd.DataFrame([
        ["Resultant Radial Load (×SF)", f"{main_bearing_radial:.1f}", "N"],
        ["Dynamic Capacity Cdyn", f"{cdyn_main:.1f}", "N"],
        ["Check", "PASS" if main_bearing_pass else "FAIL", "—"],
        ["L10 Life", f"{main_bearing_life['L10_Mrev']:.1f} Mrev / {main_bearing_life['L10_h']:.0f} h", "—"],
    ], columns=["Quantity", "Value", "Unit"]), hide_index=True, use_container_width=True)
    st.markdown("**Planet Pin Bearing / Bush**")
    if planet_bearing_life is not None:
        st.dataframe(pd.DataFrame([
            ["Resultant Pin Load", f"{stress['F_pin']:.1f}", "N"], ["Dynamic Capacity Cdyn", f"{cdyn_planet:.1f}", "N"],
            ["Relative Spin Speed", f"{kin['n_planet_rel_carrier']:.1f}", "rpm"],
            ["L10 Life", f"{planet_bearing_life['L10_Mrev']:.1f} Mrev / {planet_bearing_life['L10_h']:.0f} h", "—"],
        ], columns=["Quantity", "Value", "Unit"]), hide_index=True, use_container_width=True)
    else:
        st.write(f"Plain bronze bush selected — bearing pressure = {pin['p_bearing']:.2f} MPa vs "
                 f"allowable {bearing_p:.2f} MPa → {'PASS' if pin['pressure_ok'] else 'FAIL'}")

with tabs[5]:
    st.subheader("Planet Load Sharing")
    st.write(f"**Load-sharing/imbalance factor Kp:** {Kp:.2f}")
    st.write(f"**Application factor KA:** {KA:.2f}  |  **Dynamic factor KV:** {KV:.2f}")
    st.write(f"**Worst-case (Kp-inflated) Sun-Planet tangential force:** {loads['Ft_SP']:.1f} N")
    st.write(f"**Theoretical equal-share per planet:** {Tin_design*1000/N_PLANETS:.1f} N·mm portion of input torque")
    st.caption("Real planetary trains rarely share load perfectly between the 3 planets — Kp conservatively "
               "loads the worst mesh above the theoretical 1/N share. Raise Kp for looser manufacturing "
               "tolerances or a non-floating sun.")

with tabs[6]:
    st.subheader("Planet Pin — Full Design Check")
    st.caption("Pin modelled as a simply-supported beam spanning the two carrier plates, loaded at "
               "mid-span by the resultant mesh force.")
    st.dataframe(pd.DataFrame({
        'Quantity': ['Resultant Mesh Load on Pin', 'Support Span', 'Max Bending Moment', 'Support Shear (each)',
                     'Dia required (bending)', 'Dia required (shear)', 'Design Pin Diameter',
                     'Bearing/Bush Pressure', 'Allowable Pressure', 'Pressure Check'],
        'Value': [f"{stress['F_pin']:.1f} N", f"{pin_span:.1f} mm", f"{pin['Mmax']:.1f} N·mm",
                  f"{pin['V']:.1f} N", f"{pin['d_bend']:.2f} mm", f"{pin['d_shear']:.2f} mm",
                  f"{pin['d_pin']:.2f} mm", f"{pin['p_bearing']:.2f} MPa", f"{bearing_p:.2f} MPa",
                  'PASS' if pin['pressure_ok'] else 'FAIL']
    }), hide_index=True, use_container_width=True)
    if planet_bearing_life is not None:
        st.write(f"**Needle-bearing L10 life:** {planet_bearing_life['L10_Mrev']:.1f} million rev ≈ "
                 f"{planet_bearing_life['L10_h']:.0f} hours (at {kin['n_planet_rel_carrier']:.1f} rpm relative spin)")

with tabs[7]:
    st.subheader("Carrier — Arm & Plate Bending")
    st.markdown("**Carrier Arm** (cantilever, hub to pin centre)")
    st.dataframe(pd.DataFrame({
        'Quantity': ['Resultant Pin Load', 'Arm Length', 'Arm Width', 'Arm Thickness',
                     'Bending Moment', 'Bending Stress', 'Allowable Stress', 'Safety Factor', 'Status'],
        'Value': [f"{stress['F_pin']:.1f} N", f"{arm_length_mm:.1f} mm", f"{arm_width_mm:.1f} mm",
                  f"{arm_thickness_mm:.1f} mm", f"{carrier_arm_res['M_Nmm']:.1f} N·mm",
                  f"{carrier_arm_res['sigma_mpa']:.1f} MPa", f"{mat['sigma_allow_bend']:.0f} MPa",
                  f"{carrier_arm_res['sf']:.2f}", 'PASS' if carrier_arm_res['pass'] else 'FAIL']
    }), hide_index=True, use_container_width=True)
    st.markdown("**Carrier Plate** (sector-cantilever, bore to pin circle)")
    st.dataframe(pd.DataFrame({
        'Quantity': ['Pin Circle Radius', 'Bore Radius (+margin)', 'Plate Thickness', 'Tributary Width/Planet',
                     'Bending Moment', 'Bending Stress', 'Allowable Stress', 'Safety Factor', 'Status'],
        'Value': [f"{r_pin_circle:.2f} mm", f"{r_bore_carrier:.2f} mm", f"{plate_thickness_mm:.1f} mm",
                  f"{carrier_plate_res['tributary_width_mm']:.2f} mm", f"{carrier_plate_res['M_Nmm']:.1f} N·mm",
                  f"{carrier_plate_res['sigma_mpa']:.1f} MPa", f"{mat['sigma_allow_bend']:.0f} MPa",
                  f"{carrier_plate_res['sf']:.2f}", 'PASS' if carrier_plate_res['pass'] else 'FAIL']
    }), hide_index=True, use_container_width=True)
    st.caption("Simplified beam-sector model — verify a production carrier with FEA for plate torsion "
               "and local pin-boss stress concentration.")

with tabs[8]:
    st.subheader("Ring Gear Rim Strength")
    st.dataframe(pd.DataFrame({
        'Quantity': ['Whole Tooth Depth ht', 'Rim Thickness (OD − root radius)', 'Rim Ratio mB',
                     'Rim Thickness Factor YB', 'Ring σF (unadjusted)', 'Ring σF (rim-adjusted)',
                     'Allowable σF', 'Safety Factor', 'Status'],
        'Value': [f"{ring_rim_res['ht_mm']:.2f} mm", f"{ring_rim_res['rim_thickness_mm']:.2f} mm",
                  f"{ring_rim_res['mB']:.2f}", f"{ring_rim_res['YB']:.2f}", f"{stress['sigmaF_RP']:.1f} MPa",
                  f"{ring_rim_res['sigmaF_ring_adj_mpa']:.1f} MPa", f"{mat['sigmaF']:.0f} MPa",
                  f"{ring_rim_res['sf']:.2f}", 'PASS' if ring_rim_res['pass'] else 'FAIL']
    }), hide_index=True, use_container_width=True)
    st.caption("AGMA guidance recommends rim ratio mB ≥ 1.2 (rim thickness ≥ 1.2× whole tooth depth) so "
               "the ring doesn't flex/crack behind the teeth before the standard tooth-root formula governs.")

with tabs[9]:
    st.subheader("Keys / Splines")
    st.dataframe(pd.DataFrame([
        {'Location': f'Sun/Input shaft (⌀{din:.1f}mm)', 'Key Force (N)': f"{key_sun_res['F_key_N']:.1f}",
         'Shear (MPa)': f"{key_sun_res['tau_key_mpa']:.1f}", 'SF Shear': f"{key_sun_res['sf_shear']:.2f}",
         'Bearing (MPa)': f"{key_sun_res['sigma_bearing_mpa']:.1f}", 'SF Bearing': f"{key_sun_res['sf_bearing']:.2f}",
         'Status': 'PASS' if key_sun_res['pass'] else 'FAIL'},
        {'Location': f'Output/Carrier shaft (⌀{dout:.1f}mm)', 'Key Force (N)': f"{key_out_res['F_key_N']:.1f}",
         'Shear (MPa)': f"{key_out_res['tau_key_mpa']:.1f}", 'SF Shear': f"{key_out_res['sf_shear']:.2f}",
         'Bearing (MPa)': f"{key_out_res['sigma_bearing_mpa']:.1f}", 'SF Bearing': f"{key_out_res['sf_bearing']:.2f}",
         'Status': 'PASS' if key_out_res['pass'] else 'FAIL'},
    ]), hide_index=True, use_container_width=True)
    st.caption(f"Key section: w={key_width_mm:.1f} × h={key_height_mm:.1f} × l={key_length_mm:.1f} mm "
               f"(standard flat/parallel key, shear + crush checked).")

with tabs[10]:
    st.subheader("All Component Dimensions")
    rows = []
    for cname, dims in comp.items():
        for k, v in dims.items():
            rows.append([cname.replace('_', ' ').title(), k.replace('_', ' ').title(), f"{v:.2f}", "mm"])
    st.dataframe(pd.DataFrame(rows, columns=["Component", "Dimension", "Value", "Unit"]),
                 hide_index=True, use_container_width=True)
    st.caption("Bore/hub/rim/flange dimensions use representative sizing rules based on the calculated "
               "shaft, pin and gear diameters — confirm against actual bearing/bush/key catalogue parts.")

with tabs[11]:
    st.subheader("Overall Design Results")
    summary = pd.DataFrame([
        ["Input Torque", f"{tin:.2f}", "N·m"], ["Input Speed", f"{nin:.0f}", "rpm"],
        ["Motor Power (operating point)", f"{motor_power_operating_w:.1f}", "W"],
        ["Motor Power (design point)", f"{motor_power_design_w:.1f}", "W"],
        ["Efficiency", f"{eta:.3f}", "—"], ["Achieved Ratio", f"{ratio_actual:.4f}", "—"],
        ["Output Speed", f"{output_speed:.2f}", "rpm"], ["Design Output Torque", f"{design_out:.1f}", "N·m"],
        ["Module", f"{module:.2f}", "mm"], ["Face Width", f"{b:.1f}", "mm"],
        ["Ring Envelope OD", f"{comp['ring']['outer_d']:.1f}", "mm"],
        ["Input Shaft Dia", f"{din:.2f}", "mm"], ["Output Shaft Dia", f"{dout:.2f}", "mm"],
        ["Planet Pin Dia", f"{pin['d_pin']:.2f}", "mm"],
        ["Combined Planet Bending SF", f"{sfF:.2f}", "—"],
        ["Sun/Planet Contact SF", f"{sfH_sp:.2f}", "—"], ["Ring/Planet Contact SF", f"{sfH_rp:.2f}", "—"],
        ["Carrier Arm Bending SF", f"{carrier_arm_res['sf']:.2f}", "—"],
        ["Carrier Plate Bending SF", f"{carrier_plate_res['sf']:.2f}", "—"],
        ["Ring Rim SF", f"{ring_rim_res['sf']:.2f}", "—"],
    ], columns=["Parameter", "Value", "Unit"])
    st.dataframe(summary, hide_index=True, use_container_width=True)

    st.markdown("**Consolidated PASS/FAIL Checks**")
    check_df = pd.DataFrame([(k, "PASS ✅" if v else "FAIL ❌") for k, v in overall_checks.items()],
                             columns=["Design Check", "Status"])
    st.dataframe(check_df, hide_index=True, use_container_width=True)
    st.metric("Overall Design Status", "PASS ✅" if overall_ok else "CHECK REQUIRED ⚠️")
    if overall_ok:
        st.balloons()

    summary_text = f"""PLANETARY GEARBOX DESIGN SUMMARY
================================
Configuration: {FIXED_CASE}
Ratio: 1:{ratio_actual:.3f} (locked target 1:{TARGET_RATIO:.0f})

OPERATING
Input Torque/Speed : {tin:.2f} N·m @ {nin:.0f} rpm
Motor Power (oper.) : {motor_power_operating_w:.1f} W  (computed)
Motor Power (design): {motor_power_design_w:.1f} W  (computed)
Output Speed        : {output_speed:.1f} rpm
Design Output Torque: {design_out:.1f} N·m  (locked 65-75 band)

TEETH: Zs={zs} Zp={zp} Zr={zr}  Module={module:.2f}mm  Face Width={b:.1f}mm
Ring Envelope OD: {comp['ring']['outer_d']:.1f} mm (limit {max_od:.0f} mm)

SHAFTS: Input dia {din:.2f}mm (twist {defl_in['theta_deg']:.3f}°) | Output dia {dout:.2f}mm (twist {defl_out['theta_deg']:.3f}°)
PLANET PIN: dia {pin['d_pin']:.2f}mm, load {stress['F_pin']:.1f}N

STRESS SF: Combined planet bending {sfF:.2f} | Sun/Planet contact {sfH_sp:.2f} | Ring/Planet contact {sfH_rp:.2f}
CARRIER SF: Arm {carrier_arm_res['sf']:.2f} | Plate {carrier_plate_res['sf']:.2f}
RING RIM SF: {ring_rim_res['sf']:.2f}
KEYS SF (shear/bearing): Sun {key_sun_res['sf_shear']:.2f}/{key_sun_res['sf_bearing']:.2f} | Output {key_out_res['sf_shear']:.2f}/{key_out_res['sf_bearing']:.2f}

OVERALL STATUS: {'PASS' if overall_ok else 'CHECK REQUIRED'}
"""
    st.download_button("⬇️ Download Design Summary (TXT)", summary_text, "planetary_gearbox_summary.txt", "text/plain")

with tabs[12]:
    st.subheader("📖 Parameter Glossary")
    for category, entries in PARAM_GLOSSARY.items():
        with st.expander(category, expanded=False):
            st.dataframe(pd.DataFrame(entries, columns=["Symbol", "Meaning", "Unit", "Typical Value", "Role"]),
                         hide_index=True, use_container_width=True)

with tabs[13]:
    st.subheader("🧊 3D Visualization — True-Involute CAD-style Solids")
    st.caption("Drag to orbit, scroll to zoom, double-click to reset. Full assembly + every component below.")
    fig_asm = create_assembly_3d(int(zs), int(zp), int(zr), module, N_PLANETS, pin['d_pin'], b, din, dout,
                                  sun_geom, planet_geom, comp)
    st.plotly_chart(fig_asm, use_container_width=True)

    st.markdown("### Individual Components")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(create_component_3d_view('Sun Gear', {
            'z': int(zs), 'm': module, 'face_width': b, 'hub_od': comp['sun']['hub_od'], 'bore_d': comp['sun']['bore_d']
        }), use_container_width=True)
    with col2:
        st.plotly_chart(create_component_3d_view('Planet Gear', {
            'z': int(zp), 'm': module, 'face_width': b, 'hub_od': comp['planet']['hub_od'], 'bore_d': comp['planet']['bore_d']
        }), use_container_width=True)
    with col3:
        st.plotly_chart(create_component_3d_view('Ring Gear', {
            'z': int(zr), 'm': module, 'face_width': b + 2.0, 'outer_d': comp['ring']['outer_d']
        }), use_container_width=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.plotly_chart(create_component_3d_view('Carrier', {
            'pitch_radius': r_pin_circle, 'd_pin': pin['d_pin'], 'n_planets': N_PLANETS,
            'plate_thickness': plate_thickness_mm, 'hub_od': comp['carrier']['hub_od'],
            'bore_d': comp['carrier']['output_bore']
        }), use_container_width=True)
    with col5:
        st.plotly_chart(create_component_3d_view('Input Shaft', {
            'diameter': din, 'length': comp['input_shaft']['length']
        }), use_container_width=True)
    with col6:
        st.plotly_chart(create_component_3d_view('Output Shaft', {
            'diameter': dout, 'length': comp['output_shaft']['length']
        }), use_container_width=True)

    st.plotly_chart(create_component_3d_view('Planet Pin', {
        'diameter': pin['d_pin'], 'length': comp['planet_pin']['total_length']
    }), use_container_width=True)

with tabs[14]:
    st.subheader("📐 OpenSCAD CAD Model")
    st.caption("Simplified trapezoidal-tooth OpenSCAD model — copy/paste or download the .scad file.")
    openscad_code = generate_openscad_assembly(
        zs, zp, zr, module, N_PLANETS, pin['d_pin'], b,
        comp['sun']['hub_od'], comp['sun']['bore_d'], comp['planet']['hub_od'], comp['planet']['bore_d'],
        comp['ring']['outer_d'], comp['ring']['face_width'], r_pin_circle, plate_thickness_mm,
        comp['carrier']['hub_od'], comp['carrier']['output_bore'], din, dout, pin_span)
    st.code(openscad_code, language="openscad")
    st.download_button("⬇️ Download OpenSCAD File (.scad)", openscad_code, "planetary_gearbox.scad",
                        "text/plain", key="download_openscad")
    st.info("For precise involute profiles use an OpenSCAD library such as `gears.scad`.")

st.divider()
with st.expander("📋 Pastable Input Configuration (JSON)"):
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

st.caption("⚠️ Simplified sizing tool (ISO 6336-lite / ASME shaft code / AGMA-style rim factor / "
           "Lundberg-Palmgren bearing life). Verify against full standards before production release.")
