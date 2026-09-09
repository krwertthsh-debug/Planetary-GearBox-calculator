"""
PLANETARY GEARBOX DESIGNER v3 — COMPLETE PLATFORM
=====================================================
Combined version incorporating:
- Your original Streamlit calculator structure
- Enhanced component dimension calculations
- 3D visualization of all components (actual involute tooth geometry)
- Pastable input configuration format
- Comprehensive output tabs with full calculations
"""

import math
import io
import struct
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import streamlit as st
import plotly.graph_objects as go

# ================================================================
# CONSTANTS & DEFAULTS
# ================================================================
PI = math.pi
DEFAULT_RATIO = 9.0
DEFAULT_MAX_OD = 200.0
DEFAULT_PLANETS = 3
DEFAULT_ETA = 0.97
DEFAULT_ALPHA = 20.0
DEFAULT_BETA = 0.0
MODULES = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]

# Material properties (MPa unless noted)
MATERIALS = {
    "17CrNiMo6 / 18CrNiMo7-6 (Case Carburized)": {
        "E": 210000.0, "nu": 0.30, "tau": 240.0, 
        "sigmaF": 430.0, "sigmaH": 1500.0, "sigma_allow_bend": 380.0,
        "density": 7850.0, "hardness_HB": 620.0
    },
    "20MnCr5 / 16MnCr5 (Case Carburized)": {
        "E": 210000.0, "nu": 0.30, "tau": 140.0,
        "sigmaF": 380.0, "sigmaH": 1350.0, "sigma_allow_bend": 320.0,
        "density": 7850.0, "hardness_HB": 580.0
    },
    "EN24 / 4340 (Hardened & Tempered)": {
        "E": 206000.0, "nu": 0.30, "tau": 150.0,
        "sigmaF": 310.0, "sigmaH": 1150.0, "sigma_allow_bend": 260.0,
        "density": 7850.0, "hardness_HB": 450.0
    },
    "SAE 6150 / 51CrV4 (Spring Steel)": {
        "E": 207000.0, "nu": 0.30, "tau": 170.0,
        "sigmaF": 350.0, "sigmaH": 1250.0, "sigma_allow_bend": 290.0,
        "density": 7850.0, "hardness_HB": 520.0
    },
    "Custom": {
        "E": 200000.0, "nu": 0.30, "tau": 180.0,
        "sigmaF": 300.0, "sigmaH": 1200.0, "sigma_allow_bend": 250.0,
        "density": 7850.0, "hardness_HB": 500.0
    }
}

# ================================================================
# 3D GEAR TOOTH GEOMETRY (TRUE INVOLUTE)
# ================================================================
def involute_tooth_curve(z, m, alpha_deg=20.0, internal=False, points_per_flank=8):
    """Generate true involute tooth profile for one tooth period"""
    alpha = math.radians(alpha_deg)
    r_pitch = z * m / 2.0
    r_base = r_pitch * math.cos(alpha)
    
    def involute_point(r):
        theta = math.sqrt(max((r / r_base)**2 - 1.0, 0.0))
        x = r_base * (math.cos(theta) + theta * math.sin(theta))
        y = r_base * (math.sin(theta) - theta * math.cos(theta))
        return x, y
    
    r_start = max(r_base, r_pitch - 1.25 * m)
    r_end = r_pitch + m
    
    r_vals = np.linspace(r_start, r_end, points_per_flank)
    
    left_flank = []
    for r in r_vals:
        x, y = involute_point(r)
        left_flank.append((x, y))
    
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

def full_gear_outline(z, m, alpha_deg=20.0, internal=False):
    """Generate full gear outline"""
    tooth_pts = involute_tooth_curve(z, m, alpha_deg, internal)
    period = 2 * PI / z
    
    all_x, all_y = [], []
    for i in range(z):
        angle = i * period
        c, s = math.cos(angle), math.sin(angle)
        for px, py in tooth_pts:
            all_x.append(px * c - py * s)
            all_y.append(px * s + py * c)
    
    return np.array(all_x), np.array(all_y)

def gear_solid_3d(z, m, alpha_deg=20.0, face_width=10.0, hub_dia=None, bore_dia=None, internal=False):
    """Create 3D mesh for a gear"""
    x, y = full_gear_outline(z, m, alpha_deg, internal)
    
    if hub_dia is not None and not internal:
        hub_r = hub_dia / 2.0
        hub_pts = 20
        hub_angles = np.linspace(0, 2*PI, hub_pts, endpoint=False)
        hub_x = hub_r * np.cos(hub_angles)
        hub_y = hub_r * np.sin(hub_angles)
        
        if bore_dia is not None:
            bore_r = bore_dia / 2.0
            bore_x = bore_r * np.cos(hub_angles[::-1])
            bore_y = bore_r * np.sin(hub_angles[::-1])
            
            x = np.concatenate([x, hub_x, bore_x])
            y = np.concatenate([y, hub_y, bore_y])
    
    n = len(x)
    z0, z1 = 0.0, face_width
    
    vx = np.concatenate([x, x])
    vy = np.concatenate([y, y])
    vz = np.concatenate([np.full(n, z0), np.full(n, z1)])
    
    I, J, K = [], [], []
    for i in range(n):
        next_i = (i + 1) % n
        I.append(i); J.append(next_i); K.append(next_i + n)
        I.append(i); J.append(next_i + n); K.append(i + n)
        
        I.append(n * 2); J.append(i); K.append(next_i)
        I.append(n * 2 + 1); J.append(next_i + n); K.append(i + n)
    
    vx = np.concatenate([vx, [0.0, 0.0]])
    vy = np.concatenate([vy, [0.0, 0.0]])
    vz = np.concatenate([vz, [z0, z1]])
    
    return vx, vy, vz, I, J, K

def ring_gear_solid_3d(z, m, alpha_deg=20.0, face_width=10.0, outer_dia=None):
    """Create 3D mesh for internal ring gear"""
    x, y = full_gear_outline(z, m, alpha_deg, internal=True)
    
    if outer_dia is not None:
        outer_r = outer_dia / 2.0
        outer_pts = 30
        outer_angles = np.linspace(0, 2*PI, outer_pts, endpoint=False)
        outer_x = outer_r * np.cos(outer_angles)
        outer_y = outer_r * np.sin(outer_angles)
        
        x = np.concatenate([x, outer_x])
        y = np.concatenate([y, outer_y])
    
    n = len(x)
    z0, z1 = 0.0, face_width
    
    vx = np.concatenate([x, x])
    vy = np.concatenate([y, y])
    vz = np.concatenate([np.full(n, z0), np.full(n, z1)])
    
    I, J, K = [], [], []
    for i in range(n):
        next_i = (i + 1) % n
        I.append(i); J.append(next_i); K.append(next_i + n)
        I.append(i); J.append(next_i + n); K.append(i + n)
    
    return vx, vy, vz, I, J, K

def carrier_solid_3d(pitch_radius, d_pin, n_planets, plate_thickness, hub_diameter, bore_diameter):
    """Create 3D mesh for carrier plate"""
    plate_radius = pitch_radius + d_pin * 1.5
    plate_pts = 40
    angles = np.linspace(0, 2*PI, plate_pts, endpoint=False)
    
    x = plate_radius * np.cos(angles)
    y = plate_radius * np.sin(angles)
    
    for i in range(n_planets):
        boss_angle = i * 2 * PI / n_planets
        boss_r = d_pin / 2.0 + 2.0
        boss_x = pitch_radius * math.cos(boss_angle) + boss_r * np.cos(angles)
        boss_y = pitch_radius * math.sin(boss_angle) + boss_r * np.sin(angles)
        x = np.concatenate([x, boss_x])
        y = np.concatenate([y, boss_y])
    
    hub_r = hub_diameter / 2.0
    hub_x = hub_r * np.cos(angles)
    hub_y = hub_r * np.sin(angles)
    
    x = np.concatenate([x, hub_x])
    y = np.concatenate([y, hub_y])
    
    if bore_diameter:
        bore_r = bore_diameter / 2.0
        bore_x = bore_r * np.cos(angles[::-1])
        bore_y = bore_r * np.sin(angles[::-1])
        x = np.concatenate([x, bore_x])
        y = np.concatenate([y, bore_y])
    
    n = len(x)
    z0, z1 = 0.0, plate_thickness
    
    vx = np.concatenate([x, x])
    vy = np.concatenate([y, y])
    vz = np.concatenate([np.full(n, z0), np.full(n, z1)])
    
    I, J, K = [], [], []
    for i in range(n):
        next_i = (i + 1) % n
        I.append(i); J.append(next_i); K.append(next_i + n)
        I.append(i); J.append(next_i + n); K.append(i + n)
    
    return vx, vy, vz, I, J, K

def shaft_solid_3d(diameter, length):
    """Create 3D mesh for shaft"""
    radius = diameter / 2.0
    n = 32
    angles = np.linspace(0, 2*PI, n, endpoint=False)
    
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    
    z0, z1 = 0.0, length
    
    vx = np.concatenate([x, x])
    vy = np.concatenate([y, y])
    vz = np.concatenate([np.full(n, z0), np.full(n, z1)])
    
    I, J, K = [], [], []
    for i in range(n):
        next_i = (i + 1) % n
        I.append(i); J.append(next_i); K.append(next_i + n)
        I.append(i); J.append(next_i + n); K.append(i + n)
    
    return vx, vy, vz, I, J, K

def planet_pin_solid_3d(diameter, length):
    return shaft_solid_3d(diameter, length)

# ================================================================
# GEOMETRY CALCULATIONS
# ================================================================
def exact_ring_teeth(zs: int, zp: int) -> int:
    return zs + 2 * zp

def ratio_ring_fixed(zs: int, zr: int) -> float:
    return (zs + zr) / zs

def assembly_ok(zs: int, zr: int, nplanets: int) -> bool:
    return (zs + zr) % nplanets == 0

def planet_spacing_ok(zs: int, zp: int, nplanets: int, clearance_mm: float = 1.0) -> bool:
    return (zs + zp) * math.sin(PI / nplanets) > (zp + 2.0 + clearance_mm)

def suggest_tooth_sets(target_ratio=9.0, nplanets=3, max_od=200.0, modules=MODULES):
    rows = []
    for zs in range(17, 61):
        for zp in range(17, 101):
            zr = exact_ring_teeth(zs, zp)
            ratio = ratio_ring_fixed(zs, zr)
            if abs(ratio - target_ratio) > 1e-9:
                continue
            aok = assembly_ok(zs, zr, nplanets)
            cok = planet_spacing_ok(zs, zp, nplanets)
            if not (aok and cok):
                continue
            for m in modules:
                ring_root_d = zr * m - 2.5 * m
                od = ring_root_d + 2.5 * m + 2.0 * 6.0
                if od <= max_od:
                    rows.append(dict(zs=zs, zp=zp, zr=zr, ratio=ratio, module=m, od=od,
                                     sun_pitch=zs*m, planet_pitch=zp*m, ring_pitch=zr*m))
    return pd.DataFrame(rows)

def gear_geometry(z, m, alpha_deg=20.0, internal=False, beta_deg=0.0):
    beta = math.radians(beta_deg)
    alpha_n = math.radians(alpha_deg)
    alpha_t = math.atan(math.tan(alpha_n) / math.cos(beta))
    
    d = z * m / math.cos(beta)
    db = d * math.cos(alpha_t)
    
    if internal:
        da = d - 2*m / math.cos(beta)
        df = d + 2.5*m / math.cos(beta)
    else:
        da = d + 2*m / math.cos(beta)
        df = d - 2.5*m / math.cos(beta)
    
    return dict(z=z, module=m, pitch_d=d, base_d=db, tip_d=da, root_d=df,
                addendum=m, dedendum=1.25*m, alpha_t=math.degrees(alpha_t))

# ================================================================
# LOADS & STRESS CALCULATIONS
# ================================================================
def kinematics_ring_fixed(nin, ratio, zs, zp):
    ncarrier = nin / ratio
    nplanet_rel = abs(nin - ncarrier) * zs / zp
    return dict(n_sun=nin, n_ring=0.0, n_carrier=ncarrier,
                n_planet_rel_carrier=nplanet_rel, n_out=ncarrier)

def mesh_loads(Tin_Nm, zs, zr, nplanets, m, alpha_deg, beta_deg, Kp=1.10):
    a = math.radians(alpha_deg)
    rb_s = (zs*m/(2*math.cos(math.radians(beta_deg)))) * math.cos(a)
    rb_r = (zr*m/(2*math.cos(math.radians(beta_deg)))) * math.cos(a)
    Ft_sp = Tin_Nm*1000.0 / (nplanets * rb_s) * Kp
    Ft_rp = Ft_sp * (rb_s/rb_r)
    Fr_sp = Ft_sp * math.tan(a)
    Fr_rp = Ft_rp * math.tan(a)
    Fn_sp = Ft_sp / math.cos(a)
    Fn_rp = Ft_rp / math.cos(a)
    return dict(Ft_SP=Ft_sp, Ft_RP=Ft_rp, Fr_SP=Fr_sp, Fr_RP=Fr_rp,
                Fn_SP=Fn_sp, Fn_RP=Fn_rp, rbS=rb_s, rbR=rb_r)

def gear_stress(loads, m, b, zs, zp, zr, material, alpha_deg=20.0,
                KA=1.25, KV=1.15, KFbeta=1.20, KFalpha=1.00,
                KHbeta=1.25, KHalpha=1.00, Kp=1.10, theta_deg=120.0):
    YFaS, YSaS = 2.8, 1.55
    YFaP_SP, YSaP_SP = 2.5, 1.60
    YFaP_RP, YSaP_RP = 2.35, 1.60
    Yeps, Ybeta = 0.85, 1.0
    ZH, Zeps, Zbeta = 2.5, 0.90, 1.0
    a = math.radians(alpha_deg)
    Ftsp, Ftrp = loads['Ft_SP'], loads['Ft_RP']
    common = KA*KV*KFbeta*KFalpha
    sF_sp = (Ftsp*common/(b*m))*YFaS*YSaS*Yeps*Ybeta
    sF_rp = (Ftrp*common/(b*m))*YFaP_RP*YSaP_RP*Yeps*Ybeta
    sF_planet = math.sqrt(max(sF_sp**2 + sF_rp**2 - 2*sF_sp*sF_rp*math.cos(math.radians(theta_deg)), 0.0))

    ZE = math.sqrt(1.0/(PI*((1-material['nu']**2)/material['E'] + (1-material['nu']**2)/material['E'])))
    u_sp = zp/zs
    u_rp = zr/zp
    dS, dP = zs*m, zp*m
    term_sp = (Ftsp*KA*KV*KHbeta*KHalpha)/(b*dS) * ((u_sp+1)/u_sp)
    term_rp = (Ftrp*KA*KV*KHbeta*KHalpha)/(b*dP) * (max(u_rp-1, 1e-9)/u_rp)
    sH_sp = ZH*ZE*Zeps*Zbeta*math.sqrt(max(term_sp, 0.0))
    sH_rp = ZH*ZE*Zeps*Zbeta*math.sqrt(max(term_rp, 0.0))

    Fnsp, Fnrp = loads['Fn_SP'], loads['Fn_RP']
    Fpin = math.sqrt(max(Fnsp**2 + Fnrp**2 - 2*Fnsp*Fnrp*math.cos(math.radians(theta_deg)), 0.0))
    return dict(sigmaF_SP=sF_sp, sigmaF_RP=sF_rp, sigmaF_planet=sF_planet,
                sigmaH_SP=sH_sp, sigmaH_RP=sH_rp, F_pin=Fpin)

# ================================================================
# SHAFT & PIN SIZING
# ================================================================
def shaft_diameter(T_Nm, M_Nm, tau_allow, Kb=1.5, Kt=1.2, Kw=1.3):
    T = T_Nm*1000.0
    M = M_Nm*1000.0
    Te = math.sqrt((Kb*M)**2 + (Kt*Kw*T)**2)
    d = (16*Te/(PI*tau_allow))**(1/3)
    return d, Te/1000.0

def pin_design(F, support_span, sigma_b, tau, bearing_p, safety=1.5):
    Fd = F*safety
    Mmax = Fd*support_span/4.0
    V = Fd/2.0
    db = (32*Mmax/(PI*sigma_b))**(1/3)
    ds = math.sqrt(4*V/(PI*tau))
    d = max(db, ds)
    p = Fd/(d*support_span*0.55)
    return dict(F_design=Fd, Mmax=Mmax, V=V, d_bend=db, d_shear=ds,
                d_pin=d, p_bearing=p, pressure_ok=p <= bearing_p)

# ================================================================
# COMPONENT DIMENSIONS
# ================================================================
def component_dimensions(zs, zp, zr, m, b, d_sun, d_planet, d_ring,
                         d_in_shaft, d_out_shaft, d_pin, max_od=200.0,
                         pin_span=30.0, design_out=75.0, nplanets=3):
    sun_bore = d_in_shaft + 2.0
    sun_hub_od = max(sun_bore + 6.0, d_in_shaft*1.6)
    sun_hub_len = max(1.2*d_in_shaft, b)
    sun_rim_thick = (sun_hub_od - sun_bore)/2.0
    sun_web_thick = max(4.0, 0.15*b)
    
    planet_bore = d_pin + 2.0
    planet_hub_od = max(planet_bore + 4.0, d_pin*1.45)
    planet_hub_len = max(b, pin_span*0.75)
    planet_rim_thick = (planet_hub_od - planet_bore)/2.0
    
    ring_tooth_tip_d = d_ring - 2*m
    ring_tooth_root_d = d_ring + 2.5*m
    ring_radial_rim = max(0.18*d_ring, 8.0)
    ring_outer_d = ring_tooth_root_d + 2*ring_radial_rim
    ring_wall = (ring_outer_d - ring_tooth_root_d)/2
    ring_face_width = b + 2.0
    ring_stiffener_thick = max(4.0, 0.12*ring_face_width)
    ring_flange_od = ring_outer_d + 8.0
    ring_flange_thick = max(5.0, 0.08*ring_face_width)
    
    carrier_pitch_radius = (d_sun + d_planet)/2.0
    carrier_plate_thk = max(6.0, 0.30*b, 0.20*d_pin)
    carrier_od = min(max_od-4.0, 2*(carrier_pitch_radius + 1.6*d_pin))
    carrier_bore = d_out_shaft + 2.0
    carrier_hub_od = max(d_out_shaft + 8.0, d_out_shaft*1.5)
    carrier_hub_len = max(1.5*d_out_shaft, 1.5*b)
    carrier_pin_boss_od = d_pin + 6.0
    carrier_pin_boss_len = max(6.0, 0.5*d_pin)
    carrier_arm_width = max(6.0, 0.25*carrier_plate_thk)
    carrier_arm_depth = max(8.0, 0.8*d_pin)
    carrier_total_height = carrier_plate_thk + carrier_hub_len + 2*carrier_pin_boss_len
    
    input_shaft_len = 40.0 + b + 20.0
    input_key_width = 0.25*d_in_shaft if d_in_shaft < 30.0 else 8.0
    input_key_depth = 0.15*d_in_shaft if d_in_shaft < 30.0 else 5.0
    
    output_shaft_len = 40.0 + b + 20.0 + 30.0
    output_key_width = 0.25*d_out_shaft if d_out_shaft < 30.0 else 10.0
    output_key_depth = 0.15*d_out_shaft if d_out_shaft < 30.0 else 6.0
    
    pin_head_d = d_pin*1.3
    pin_head_thick = max(3.0, 0.2*d_pin)
    pin_total_len = pin_span + 2*pin_head_thick
    
    housing_clearance = 2.0
    envelope_od = ring_outer_d + 2*housing_clearance
    housing_wall = max(3.0, 0.06*envelope_od)
    housing_length = input_shaft_len + b + output_shaft_len
    housing_flange_od = envelope_od + 2*housing_wall
    housing_base_thick = max(5.0, 0.15*housing_wall)
    
    return {
        'sun': {'gear_face_width': b, 'bore_d': sun_bore, 'hub_od': sun_hub_od,
                'hub_length': sun_hub_len, 'rim_thickness': sun_rim_thick,
                'web_thickness': sun_web_thick, 'total_height': b + sun_hub_len},
        'planet': {'gear_face_width': b, 'bore_d': planet_bore, 'hub_od': planet_hub_od,
                   'hub_length': planet_hub_len, 'rim_thickness': planet_rim_thick,
                   'total_height': b + planet_hub_len},
        'ring': {'inner_tooth_tip_d': ring_tooth_tip_d, 'inner_tooth_root_d': ring_tooth_root_d,
                 'outer_d': ring_outer_d, 'wall_thickness': ring_wall,
                 'face_width': ring_face_width, 'total_height': ring_face_width,
                 'stiffener_thickness': ring_stiffener_thick,
                 'flange_od': ring_flange_od, 'flange_thickness': ring_flange_thick},
        'carrier': {'plate_thickness': carrier_plate_thk, 'plate_od': carrier_od,
                    'output_bore': carrier_bore, 'hub_od': carrier_hub_od,
                    'hub_length': carrier_hub_len, 'planet_pin_circle_d': 2*carrier_pitch_radius,
                    'pin_boss_od': carrier_pin_boss_od, 'pin_boss_length': carrier_pin_boss_len,
                    'arm_width': carrier_arm_width, 'arm_depth': carrier_arm_depth,
                    'total_height': carrier_total_height},
        'input_shaft': {'diameter': d_in_shaft, 'length': input_shaft_len,
                        'key_width': input_key_width, 'key_depth': input_key_depth},
        'output_shaft': {'diameter': d_out_shaft, 'length': output_shaft_len,
                         'key_width': output_key_width, 'key_depth': output_key_depth},
        'planet_pin': {'diameter': d_pin, 'span_length': pin_span,
                       'head_diameter': pin_head_d, 'head_thickness': pin_head_thick,
                       'total_length': pin_total_len},
        'housing': {'outer_d': envelope_od, 'wall_thickness': housing_wall,
                    'length': housing_length, 'flange_od': housing_flange_od,
                    'base_thickness': housing_base_thick,
                    'inner_d': envelope_od - 2*housing_wall}
    }

# ================================================================
# DEFLECTION & BEARING & THERMAL
# ================================================================
def calculate_deflections(loads, b, m, zs, zr, material, d_in_shaft, d_out_shaft,
                          pin_span, d_pin, carrier_plate_thk):
    E = material["E"]
    nu = material["nu"]
    
    rb_s = (zs * m / 2.0) * math.cos(math.radians(20.0))
    Ft_sp = loads["Ft_SP"]
    sun_span = b * 0.5
    I_sun = PI * rb_s**4 / 4.0
    sun_deflection = (Ft_sp * sun_span**3) / (3 * E * I_sun) * 1000.0
    
    rb_r = (zr * m / 2.0) * math.cos(math.radians(20.0))
    I_ring = PI * (rb_r**4 - (rb_r - 0.18 * rb_r)**4) / 4.0
    Ft_rp = loads["Ft_RP"]
    ring_deflection = (Ft_rp * (b*0.5)**3) / (3 * E * I_ring) * 1000.0
    
    F_pin = loads["Fn_SP"] + loads["Fn_RP"]
    I_carrier = (carrier_plate_thk**3 / 12.0) * 20.0
    carrier_deflection = (F_pin * (pin_span*0.75)**3) / (48 * E * I_carrier) * 1000.0
    
    G = E / (2 * (1 + nu))
    T_in = loads["Ft_SP"] * (zs * m / 2.0) / 1000.0
    T_out = loads["Ft_SP"] * (zr * m / 2.0) / 1000.0
    
    J_in = PI * d_in_shaft**4 / 32.0
    J_out = PI * d_out_shaft**4 / 32.0
    
    theta_in = (T_in * 1000.0 * 40.0) / (G * J_in)
    theta_out = (T_out * 1000.0 * (40.0 + b)) / (G * J_out)
    
    F_n = loads["Fn_SP"]
    tooth_height = 2.5 * m
    tooth_width = m
    I_tooth = tooth_width * tooth_height**3 / 12.0
    tooth_deflection = (F_n * tooth_height**3) / (3 * E * I_tooth) * 1000.0
    
    return {
        "sun_deflection": sun_deflection,
        "ring_deflection": ring_deflection,
        "carrier_deflection": carrier_deflection,
        "shaft_torsional_angle_input": math.degrees(theta_in),
        "shaft_torsional_angle_output": math.degrees(theta_out),
        "tooth_deflection": tooth_deflection
    }

def bearing_L10_life(C_dyn, P_equiv, n_rpm, bearing_type="Ball"):
    p = 3.0 if bearing_type == "Ball" else 10.0/3.0
    if P_equiv <= 0 or n_rpm <= 0:
        return {"L10_Mrev": float('inf'), "L10_h": float('inf')}
    L10_Mrev = (C_dyn / P_equiv) ** p
    L10_h = (L10_Mrev * 1e6) / (60.0 * n_rpm)
    return {"L10_Mrev": L10_Mrev, "L10_h": L10_h}

def calculate_thermal(power_loss_W, housing_area_m2, ambient_C=25.0, h_coeff=15.0):
    delta_T = power_loss_W / (h_coeff * housing_area_m2)
    T_steady = ambient_C + delta_T
    return {
        "power_loss_W": power_loss_W,
        "housing_area_m2": housing_area_m2,
        "delta_T": delta_T,
        "steady_state_T": T_steady,
        "thermal_ok": T_steady < 90.0
    }

# ================================================================
# 3D VISUALIZATION FUNCTIONS
# ================================================================
def create_assembly_3d(zs, zp, zr, m, n_planets, d_pin, face_width, d_in_shaft, d_out_shaft,
                        sun_geometry, planet_geometry, ring_geometry, comp_dims):
    sun_mesh = gear_solid_3d(zs, m, 20.0, face_width,
                            hub_dia=comp_dims['sun']['hub_od'],
                            bore_dia=comp_dims['sun']['bore_d'])
    
    planet_meshes = []
    pitch_radius = (sun_geometry['pitch_d'] + planet_geometry['pitch_d']) / 2.0
    
    for k in range(n_planets):
        angle = k * 2 * PI / n_planets
        planet_mesh = gear_solid_3d(zp, m, 20.0, face_width,
                                   hub_dia=comp_dims['planet']['hub_od'],
                                   bore_dia=comp_dims['planet']['bore_d'])
        c, s = math.cos(angle), math.sin(angle)
        px, py = pitch_radius * c, pitch_radius * s
        vx, vy, vz, I, J, K = planet_mesh
        vx = vx + px
        vy = vy + py
        planet_meshes.append((vx, vy, vz, I, J, K))
    
    ring_mesh = ring_gear_solid_3d(zr, m, 20.0, face_width + 2.0,
                                   outer_dia=comp_dims['ring']['outer_d'])
    
    carrier_mesh = carrier_solid_3d(
        pitch_radius, d_pin, n_planets, comp_dims['carrier']['plate_thickness'],
        comp_dims['carrier']['hub_od'], comp_dims['carrier']['output_bore']
    )
    
    input_shaft = shaft_solid_3d(d_in_shaft, 40.0)
    output_shaft = shaft_solid_3d(d_out_shaft, 40.0 + face_width)
    
    pins = []
    for k in range(n_planets):
        angle = k * 2 * PI / n_planets
        pin_mesh = planet_pin_solid_3d(d_pin, comp_dims['planet_pin']['total_length'])
        c, s = math.cos(angle), math.sin(angle)
        px, py = pitch_radius * c, pitch_radius * s
        vx, vy, vz, I, J, K = pin_mesh
        vx = vx + px
        vy = vy + py
        pins.append((vx, vy, vz, I, J, K))
    
    traces = []
    
    traces.append(go.Mesh3d(
        x=sun_mesh[0], y=sun_mesh[1], z=sun_mesh[2],
        i=sun_mesh[3], j=sun_mesh[4], k=sun_mesh[5],
        color='#D9531E', opacity=0.9, name='Sun Gear', flatshading=True
    ))
    
    for idx, planet_mesh in enumerate(planet_meshes):
        traces.append(go.Mesh3d(
            x=planet_mesh[0], y=planet_mesh[1], z=planet_mesh[2],
            i=planet_mesh[3], j=planet_mesh[4], k=planet_mesh[5],
            color='#EDB120', opacity=0.9, name=f'Planet {idx+1}', flatshading=True
        ))
    
    traces.append(go.Mesh3d(
        x=ring_mesh[0], y=ring_mesh[1], z=ring_mesh[2],
        i=ring_mesh[3], j=ring_mesh[4], k=ring_mesh[5],
        color='#8a8a8a', opacity=0.7, name='Ring Gear', flatshading=True
    ))
    
    traces.append(go.Mesh3d(
        x=carrier_mesh[0], y=carrier_mesh[1], z=carrier_mesh[2],
        i=carrier_mesh[3], j=carrier_mesh[4], k=carrier_mesh[5],
        color='#4C72B0', opacity=0.6, name='Carrier', flatshading=True
    ))
    
    traces.append(go.Mesh3d(
        x=input_shaft[0], y=input_shaft[1], z=input_shaft[2],
        i=input_shaft[3], j=input_shaft[4], k=input_shaft[5],
        color='#8C8C8C', opacity=0.9, name='Input Shaft', flatshading=True
    ))
    
    traces.append(go.Mesh3d(
        x=output_shaft[0], y=output_shaft[1], z=output_shaft[2],
        i=output_shaft[3], j=output_shaft[4], k=output_shaft[5],
        color='#8C8C8C', opacity=0.9, name='Output Shaft', flatshading=True
    ))
    
    for idx, pin_mesh in enumerate(pins):
        traces.append(go.Mesh3d(
            x=pin_mesh[0], y=pin_mesh[1], z=pin_mesh[2],
            i=pin_mesh[3], j=pin_mesh[4], k=pin_mesh[5],
            color='#3B3B3B', opacity=0.9, name=f'Pin {idx+1}', flatshading=True
        ))
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        title='3D Planetary Gearbox Assembly',
        height=700,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_component_3d_view(component_type, params):
    fig = go.Figure()
    
    if component_type == 'Sun Gear':
        mesh = gear_solid_3d(
            params['z'], params['m'], 20.0,
            params['face_width'], params['hub_od'], params['bore_d']
        )
        fig.add_trace(go.Mesh3d(
            x=mesh[0], y=mesh[1], z=mesh[2],
            i=mesh[3], j=mesh[4], k=mesh[5],
            color='#D9531E', opacity=0.9, name='Sun Gear', flatshading=True
        ))
        title = f"Sun Gear — z={params['z']}, m={params['m']:.2f}mm"
    elif component_type == 'Planet Gear':
        mesh = gear_solid_3d(
            params['z'], params['m'], 20.0,
            params['face_width'], params['hub_od'], params['bore_d']
        )
        fig.add_trace(go.Mesh3d(
            x=mesh[0], y=mesh[1], z=mesh[2],
            i=mesh[3], j=mesh[4], k=mesh[5],
            color='#EDB120', opacity=0.9, name='Planet Gear', flatshading=True
        ))
        title = f"Planet Gear — z={params['z']}, m={params['m']:.2f}mm"
    elif component_type == 'Ring Gear':
        mesh = ring_gear_solid_3d(
            params['z'], params['m'], 20.0,
            params['face_width'], params['outer_d']
        )
        fig.add_trace(go.Mesh3d(
            x=mesh[0], y=mesh[1], z=mesh[2],
            i=mesh[3], j=mesh[4], k=mesh[5],
            color='#8a8a8a', opacity=0.7, name='Ring Gear', flatshading=True
        ))
        title = f"Ring Gear — z={params['z']}, m={params['m']:.2f}mm"
    elif component_type == 'Carrier':
        mesh = carrier_solid_3d(
            params['pitch_radius'], params['d_pin'], params['n_planets'],
            params['plate_thickness'], params['hub_od'], params['bore_d']
        )
        fig.add_trace(go.Mesh3d(
            x=mesh[0], y=mesh[1], z=mesh[2],
            i=mesh[3], j=mesh[4], k=mesh[5],
            color='#4C72B0', opacity=0.6, name='Carrier', flatshading=True
        ))
        title = "Carrier Plate"
    elif component_type == 'Input Shaft':
        mesh = shaft_solid_3d(params['diameter'], params['length'])
        fig.add_trace(go.Mesh3d(
            x=mesh[0], y=mesh[1], z=mesh[2],
            i=mesh[3], j=mesh[4], k=mesh[5],
            color='#8C8C8C', opacity=0.9, name='Input Shaft', flatshading=True
        ))
        title = f"Input Shaft — ⌀{params['diameter']:.1f}mm"
    elif component_type == 'Output Shaft':
        mesh = shaft_solid_3d(params['diameter'], params['length'])
        fig.add_trace(go.Mesh3d(
            x=mesh[0], y=mesh[1], z=mesh[2],
            i=mesh[3], j=mesh[4], k=mesh[5],
            color='#8C8C8C', opacity=0.9, name='Output Shaft', flatshading=True
        ))
        title = f"Output Shaft — ⌀{params['diameter']:.1f}mm"
    elif component_type == 'Planet Pin':
        mesh = planet_pin_solid_3d(params['diameter'], params['length'])
        fig.add_trace(go.Mesh3d(
            x=mesh[0], y=mesh[1], z=mesh[2],
            i=mesh[3], j=mesh[4], k=mesh[5],
            color='#3B3B3B', opacity=0.9, name='Planet Pin', flatshading=True
        ))
        title = f"Planet Pin — ⌀{params['diameter']:.1f}mm"
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.3, y=1.3, z=0.7))
        ),
        title=title,
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

# ================================================================
# STREAMLIT APP
# ================================================================
st.set_page_config(
    page_title="Planetary Gearbox Designer v3",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1f4e78;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .pass-badge {
        background: #28a745;
        color: white;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
    }
    .fail-badge {
        background: #dc3545;
        color: white;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚙️ PLANETARY GEARBOX DESIGNER v3</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Complete Single-Stage 1:9 Reduction — All Component Calculations + 3D Visualization</div>', unsafe_allow_html=True)

# ================================================================
# SIDEBAR — INPUTS
# ================================================================
with st.sidebar:
    st.header("🔧 Operating Conditions")
    
    tin = st.number_input("Input Torque (N·m)", 0.1, 1000.0, 5.0, 0.1)
    nin = st.number_input("Input Speed (rpm)", 10.0, 20000.0, 1500.0, 50.0)
    motor_power_kw = st.number_input("Motor Power (kW)", 0.05, 20.0, 0.785, 0.01)
    eta = st.number_input("Stage Efficiency", 0.80, 0.99, 0.97, 0.005)
    design_out = st.slider("Design Output Torque (N·m)", 65.0, 75.0, 75.0, 1.0)
    
    st.header("⚙️ Gear Parameters")
    ratio_target = st.number_input("Target Ratio", 1.0, 20.0, 9.0, 0.1)
    nplanets = st.selectbox("Number of Planets", [3, 4, 5], index=0)
    alpha = st.number_input("Pressure Angle (°)", 14.5, 25.0, 20.0, 0.5)
    beta = st.number_input("Helix Angle (°)", 0.0, 30.0, 0.0, 1.0)
    max_od = st.number_input("Max OD (mm)", 50.0, 500.0, 200.0, 1.0)
    
    tooth_mode = st.radio("Tooth Selection:", ["Manual", "Auto Suggest"], index=0)
    
    if tooth_mode == "Manual":
        zs = st.number_input("Sun Teeth Zs", 17, 120, 20, 1)
        zp = st.number_input("Planet Teeth Zp", 17, 150, 70, 1)
        module = st.number_input("Module (mm)", 0.5, 10.0, 1.0, 0.05)
    else:
        cand = suggest_tooth_sets(ratio_target, nplanets, max_od)
        if cand.empty:
            st.error("No exact candidates found. Adjust parameters.")
            st.stop()
        opt = cand.sort_values(['module', 'zs', 'zp']).iloc[0]
        zs, zp, module = int(opt.zs), int(opt.zp), float(opt.module)
        st.info(f"Auto: Zs={zs}, Zp={zp}, m={module:.2f}mm")
    
    zr = exact_ring_teeth(int(zs), int(zp))
    
    st.header("📐 Material")
    mat_name = st.selectbox("Material", list(MATERIALS.keys()))
    mat = dict(MATERIALS[mat_name])
    
    if mat_name == "Custom":
        with st.expander("Custom Properties", expanded=True):
            mat['E'] = st.number_input("E (MPa)", 50000.0, 300000.0, 200000.0)
            mat['nu'] = st.number_input("ν", 0.20, 0.40, 0.30, 0.01)
            mat['tau'] = st.number_input("τ (MPa)", 20.0, 500.0, 180.0)
            mat['sigmaF'] = st.number_input("σF (MPa)", 50.0, 1000.0, 300.0)
            mat['sigmaH'] = st.number_input("σH (MPa)", 200.0, 2000.0, 1200.0)
            mat['sigma_allow_bend'] = st.number_input("σBend (MPa)", 50.0, 800.0, 250.0)
    
    st.header("🔩 Design Factors")
    Kb = st.number_input("ASME Kb", 1.0, 3.0, 1.5, 0.1)
    Kt = st.number_input("ASME Kt", 1.0, 2.0, 1.2, 0.1)
    Kw = st.number_input("Keyway Kw", 1.0, 2.0, 1.3, 0.05)
    service = st.number_input("Bearing SF", 1.0, 3.0, 1.5, 0.1)
    pin_span = st.number_input("Pin Span (mm)", 10.0, 100.0, 30.0, 1.0)
    bearing_p = st.number_input("Pin Pressure (MPa)", 5.0, 80.0, 25.0, 1.0)
    theta = st.number_input("Force Angle θ (°)", 60.0, 180.0, 120.0, 1.0)
    
    st.header("📏 Face Width")
    b_factor = st.slider("b/m Ratio", 6.0, 20.0, 12.0, 0.5)
    b = b_factor * module

# ================================================================
# CALCULATIONS
# ================================================================
sun_geom = gear_geometry(int(zs), module, alpha, False, beta)
planet_geom = gear_geometry(int(zp), module, alpha, False, beta)
ring_geom = gear_geometry(int(zr), module, alpha, True, beta)

sun_geom['face_width'] = b
planet_geom['face_width'] = b
ring_geom['face_width'] = b + 2.0

ratio_actual = ratio_ring_fixed(int(zs), int(zr))
ratio_ok = abs(ratio_actual - ratio_target) < 1e-9
output_speed = nin / ratio_actual

assembly = assembly_ok(int(zs), int(zr), nplanets)
clearance = planet_spacing_ok(int(zs), int(zp), nplanets)

Tin_design = design_out / (ratio_actual * eta)
loads = mesh_loads(Tin_design, int(zs), int(zr), nplanets, module, alpha, beta)

stress = gear_stress(loads, module, b, int(zs), int(zp), int(zr), mat, alpha, theta_deg=theta)

din, Te_in = shaft_diameter(Tin_design, 0.0, mat['tau'], Kb, Kt, Kw)
dout, Te_out = shaft_diameter(design_out, 0.0, mat['tau'], Kb, Kt, Kw)
pin = pin_design(stress['F_pin'], pin_span, mat['sigma_allow_bend'], mat['tau'], bearing_p)

comp = component_dimensions(
    int(zs), int(zp), int(zr), module, b,
    sun_geom['pitch_d'], planet_geom['pitch_d'], ring_geom['pitch_d'],
    din, dout, pin['d_pin'], max_od, pin_span, design_out, nplanets
)

deflections = calculate_deflections(
    loads, b, module, int(zs), int(zr), mat,
    din, dout, pin_span, pin['d_pin'], comp['carrier']['plate_thickness']
)

main_bearing_radial = math.hypot(loads['Fr_SP'], loads['Ft_SP']) * service
n_planet_rel = abs(nin - output_speed) * int(zs) / int(zp)
main_bearing_life = bearing_L10_life(12000.0, main_bearing_radial, nin, "Ball")
planet_bearing_life = bearing_L10_life(6000.0, stress['F_pin'] * service, n_planet_rel, "Roller")

power_loss = (1 - eta) * motor_power_kw * 1000.0
housing_area = PI * comp['housing']['outer_d'] / 1000.0 * comp['housing']['length'] / 1000.0
thermal = calculate_thermal(power_loss, housing_area)

sfF = mat['sigmaF'] / max(stress['sigmaF_planet'], 1e-9)
sfH_sp = mat['sigmaH'] / max(stress['sigmaH_SP'], 1e-9)
sfH_rp = mat['sigmaH'] / max(stress['sigmaH_RP'], 1e-9)

od_ok = comp['housing']['outer_d'] <= max_od
checks = [
    ("Ratio", ratio_ok),
    ("Assembly", assembly),
    ("Clearance", clearance),
    ("OD Envelope", od_ok),
    ("Gear Bending SF", sfF >= 1.0),
    ("Sun Contact SF", sfH_sp >= 1.0),
    ("Ring Contact SF", sfH_rp >= 1.0),
    ("Planet Pin Pressure", pin['pressure_ok']),
    ("Deflection < 0.1mm", deflections['sun_deflection'] < 0.1),
    ("Thermal < 90°C", thermal['thermal_ok']),
]
overall = all(ok for _, ok in checks)

# ================================================================
# DASHBOARD
# ================================================================
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
col_m1.metric("Ratio", f"1:{ratio_actual:.3f}")
col_m2.metric("Output Speed", f"{output_speed:.1f} rpm")
col_m3.metric("Motor Power", f"{motor_power_kw:.3f} kW")
col_m4.metric("Design Torque", f"{design_out:.1f} N·m")
col_m5.metric("Status", "PASS ✅" if overall else "CHECK ⚠️")

st.info(f"**Teeth:** Zs={zs}, Zp={zp}, Zr={zr}  |  **Module:** {module:.2f} mm  |  **Face Width:** {b:.1f} mm")

# ================================================================
# TABS
# ================================================================
tabs = st.tabs([
    "📊 Results",
    "⚙️ Tooth Synthesis",
    "📐 Geometry",
    "🔄 Kinematics",
    "🧲 Forces",
    "🧮 Stress",
    "🔩 Shafts & Pins",
    "🧱 Component Dimensions",
    "📏 Deflections",
    "🛞 Bearings",
    "🔥 Thermal",
    "✅ Checks",
    "🧊 3D Visualization"
])

# TAB 1: Results
with tabs[0]:
    st.subheader("Overall Design Results")
    summary = pd.DataFrame([
        ["Input Torque", tin, "N·m"],
        ["Input Speed", nin, "rpm"],
        ["Motor Power", motor_power_kw, "kW"],
        ["Efficiency", eta, "—"],
        ["Actual Ratio", ratio_actual, "—"],
        ["Output Speed", output_speed, "rpm"],
        ["Design Output Torque", design_out, "N·m"],
        ["Input Design Torque", Tin_design, "N·m"],
        ["Module", module, "mm"],
        ["Face Width", b, "mm"],
        ["Ring Envelope OD", comp['housing']['outer_d'], "mm"],
        ["Input Shaft Dia", din, "mm"],
        ["Output Shaft Dia", dout, "mm"],
        ["Planet Pin Dia", pin['d_pin'], "mm"],
        ["Gear Bending SF", sfF, "—"],
        ["Sun Contact SF", sfH_sp, "—"],
        ["Ring Contact SF", sfH_rp, "—"],
    ], columns=["Parameter", "Value", "Unit"])
    st.dataframe(summary, hide_index=True, use_container_width=True)

# TAB 2: Tooth Synthesis
with tabs[1]:
    st.subheader("Tooth Synthesis")
    tooth_df = pd.DataFrame([
        ["Sun", zs, "Input"],
        ["Planet", zp, "Meshing"],
        ["Ring", zr, "Fixed"],
        ["S + 2P = R", f"{zs}+2({zp}) = {zs+2*zp}", "Geometry"],
        ["Ratio", f"{ratio_actual:.3f}", "(S+R)/S"],
        ["Assembly", "PASS" if assembly else "FAIL", "(S+R) mod N"],
        ["Clearance", "PASS" if clearance else "CHECK", "Pitch-circle check"],
        ["Module", f"{module:.2f}", "mm"],
    ], columns=["Item", "Value", "Meaning"])
    st.dataframe(tooth_df, hide_index=True, use_container_width=True)
    
    cand = suggest_tooth_sets(ratio_target, nplanets, max_od)
    if not cand.empty:
        st.markdown("**Other 1:9 candidates:**")
        st.dataframe(cand.head(20), hide_index=True, use_container_width=True)

# TAB 3: Geometry
with tabs[2]:
    st.subheader("Gear Geometry")
    geo_df = pd.DataFrame([
        ["Sun", zs, sun_geom['pitch_d'], sun_geom['base_d'], sun_geom['tip_d'], sun_geom['root_d'], b],
        ["Planet", zp, planet_geom['pitch_d'], planet_geom['base_d'], planet_geom['tip_d'], planet_geom['root_d'], b],
        ["Ring", zr, ring_geom['pitch_d'], ring_geom['base_d'], ring_geom['tip_d'], ring_geom['root_d'], b+2.0],
    ], columns=["Gear", "Teeth", "Pitch (mm)", "Base (mm)", "Tip (mm)", "Root (mm)", "Face (mm)"])
    st.dataframe(geo_df, hide_index=True, use_container_width=True)
    st.write(f"Centre Distance (S-P): {(sun_geom['pitch_d']+planet_geom['pitch_d'])/2:.3f} mm")
    st.write(f"Centre Distance (R-P): {(ring_geom['pitch_d']-planet_geom['pitch_d'])/2:.3f} mm")

# TAB 4: Kinematics
with tabs[3]:
    kin = kinematics_ring_fixed(nin, ratio_actual, int(zs), int(zp))
    st.dataframe(pd.DataFrame([
        ["Sun", kin['n_sun'], "rpm"],
        ["Ring", 0.0, "rpm"],
        ["Carrier", kin['n_carrier'], "rpm"],
        ["Planet (rel)", kin['n_planet_rel_carrier'], "rpm"],
    ], columns=["Member", "Speed", "Unit"]), hide_index=True, use_container_width=True)

# TAB 5: Forces
with tabs[4]:
    st.dataframe(pd.DataFrame([
        ["Ft SP", loads['Ft_SP'], "N"],
        ["Ft RP", loads['Ft_RP'], "N"],
        ["Fr SP", loads['Fr_SP'], "N"],
        ["Fr RP", loads['Fr_RP'], "N"],
        ["Fn SP", loads['Fn_SP'], "N"],
        ["Fn RP", loads['Fn_RP'], "N"],
        ["Pin Load", stress['F_pin'], "N"],
    ], columns=["Force", "Value", "Unit"]), hide_index=True, use_container_width=True)

# TAB 6: Stress
with tabs[5]:
    st.dataframe(pd.DataFrame([
        ["Bending SP", stress['sigmaF_SP'], mat['sigmaF'], mat['sigmaF']/max(stress['sigmaF_SP'],1e-9)],
        ["Bending RP", stress['sigmaF_RP'], mat['sigmaF'], mat['sigmaF']/max(stress['sigmaF_RP'],1e-9)],
        ["Combined Planet", stress['sigmaF_planet'], mat['sigmaF'], sfF],
        ["Contact SP", stress['sigmaH_SP'], mat['sigmaH'], sfH_sp],
        ["Contact RP", stress['sigmaH_RP'], mat['sigmaH'], sfH_rp],
    ], columns=["Check", "Actual MPa", "Allowable MPa", "SF"]), hide_index=True, use_container_width=True)
    st.warning("Screening equations — verify with full ISO/AGMA.")

# TAB 7: Shafts & Pins
with tabs[6]:
    st.dataframe(pd.DataFrame([
        ["Input Shaft", Tin_design, 0.0, Te_in, din],
        ["Output Shaft", design_out, 0.0, Te_out, dout],
        ["Planet Pin", pin['F_design'], pin['Mmax'], "—", pin['d_pin']],
    ], columns=["Component", "Torque/Load", "Moment", "Te", "Dia (mm)"]), 
    hide_index=True, use_container_width=True)
    st.caption("Pin bending/shear/pressure details below.")
    st.dataframe(pd.DataFrame([
        ["Pin Design Load", pin['F_design'], "N"],
        ["Max Bending Moment", pin['Mmax'], "N·mm"],
        ["Support Shear", pin['V'], "N"],
        ["d from Bending", pin['d_bend'], "mm"],
        ["d from Shear", pin['d_shear'], "mm"],
        ["Bearing Pressure", pin['p_bearing'], "MPa"],
        ["Pressure Check", "PASS" if pin['pressure_ok'] else "FAIL", "—"],
    ], columns=["Parameter", "Value", "Unit"]), hide_index=True, use_container_width=True)

# TAB 8: Component Dimensions
with tabs[7]:
    st.subheader("All Component Dimensions")
    rows = []
    for cname, dims in comp.items():
        if isinstance(dims, dict):
            for k, v in dims.items():
                rows.append([cname.replace('_', ' ').title(), k.replace('_', ' ').title(), v, "mm"])
    st.dataframe(pd.DataFrame(rows, columns=["Component", "Dimension", "Value", "Unit"]), 
                 hide_index=True, use_container_width=True)

# TAB 9: Deflections
with tabs[8]:
    st.dataframe(pd.DataFrame([
        ["Sun Deflection", deflections['sun_deflection'], "mm", "< 0.10", "✅" if deflections['sun_deflection'] < 0.10 else "⚠️"],
        ["Ring Deflection", deflections['ring_deflection'], "mm", "< 0.10", "✅" if deflections['ring_deflection'] < 0.10 else "⚠️"],
        ["Carrier Deflection", deflections['carrier_deflection'], "mm", "< 0.15", "✅" if deflections['carrier_deflection'] < 0.15 else "⚠️"],
        ["Torsion Input", deflections['shaft_torsional_angle_input'], "°", "< 1.0", "✅" if deflections['shaft_torsional_angle_input'] < 1.0 else "⚠️"],
        ["Torsion Output", deflections['shaft_torsional_angle_output'], "°", "< 1.0", "✅" if deflections['shaft_torsional_angle_output'] < 1.0 else "⚠️"],
        ["Tooth Deflection", deflections['tooth_deflection'], "mm", "< 0.05", "✅" if deflections['tooth_deflection'] < 0.05 else "⚠️"],
    ], columns=["Component", "Value", "Unit", "Limit", "Status"]), 
    hide_index=True, use_container_width=True)

# TAB 10: Bearings
with tabs[9]:
    st.dataframe(pd.DataFrame([
        ["Main Bearing", main_bearing_radial, 12000.0, main_bearing_life['L10_Mrev'], main_bearing_life['L10_h'], "Ball"],
        ["Planet Bearing", stress['F_pin']*service, 6000.0, planet_bearing_life['L10_Mrev'], planet_bearing_life['L10_h'], "Roller"],
    ], columns=["Bearing", "Load N", "C N", "L10 Mrev", "L10 hours", "Type"]), 
    hide_index=True, use_container_width=True)

# TAB 11: Thermal
with tabs[10]:
    st.dataframe(pd.DataFrame([
        ["Power Loss", thermal['power_loss_W'], "W"],
        ["Housing Area", thermal['housing_area_m2'], "m²"],
        ["Temp Rise", thermal['delta_T'], "°C"],
        ["Steady Temp", thermal['steady_state_T'], "°C"],
        ["Status", "PASS" if thermal['thermal_ok'] else "NEEDS COOLING", "—"],
    ], columns=["Parameter", "Value", "Unit"]), 
    hide_index=True, use_container_width=True)

# TAB 12: Checks
with tabs[11]:
    check_df = pd.DataFrame(
        [(name, "PASS ✅" if ok else "FAIL ❌") for name, ok in checks],
        columns=["Design Check", "Status"]
    )
    st.dataframe(check_df, hide_index=True, use_container_width=True)
    if overall:
        st.balloons()

# TAB 13: 3D Visualization
with tabs[12]:
    st.subheader("🧊 3D Visualization")
    
    st.markdown("### Full Assembly")
    fig_asm = create_assembly_3d(
        int(zs), int(zp), int(zr), module, nplanets, pin['d_pin'], b,
        din, dout, sun_geom, planet_geom, ring_geom, comp
    )
    st.plotly_chart(fig_asm, use_container_width=True)
    
    st.markdown("### Individual Components")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Sun Gear**")
        fig_sun = create_component_3d_view('Sun Gear', {
            'z': int(zs), 'm': module, 'face_width': b,
            'hub_od': comp['sun']['hub_od'], 'bore_d': comp['sun']['bore_d']
        })
        st.plotly_chart(fig_sun, use_container_width=True)
    
    with col2:
        st.markdown("**Planet Gear**")
        fig_planet = create_component_3d_view('Planet Gear', {
            'z': int(zp), 'm': module, 'face_width': b,
            'hub_od': comp['planet']['hub_od'], 'bore_d': comp['planet']['bore_d']
        })
        st.plotly_chart(fig_planet, use_container_width=True)
    
    with col3:
        st.markdown("**Ring Gear**")
        fig_ring = create_component_3d_view('Ring Gear', {
            'z': int(zr), 'm': module, 'face_width': b+2.0,
            'outer_d': comp['ring']['outer_d']
        })
        st.plotly_chart(fig_ring, use_container_width=True)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("**Carrier**")
        fig_carrier = create_component_3d_view('Carrier', {
            'pitch_radius': (sun_geom['pitch_d'] + planet_geom['pitch_d'])/2.0,
            'd_pin': pin['d_pin'], 'n_planets': nplanets,
            'plate_thickness': comp['carrier']['plate_thickness'],
            'hub_od': comp['carrier']['hub_od'],
            'bore_d': comp['carrier']['output_bore']
        })
        st.plotly_chart(fig_carrier, use_container_width=True)
    
    with col5:
        st.markdown("**Input Shaft**")
        fig_in = create_component_3d_view('Input Shaft', {
            'diameter': din, 'length': comp['input_shaft']['length']
        })
        st.plotly_chart(fig_in, use_container_width=True)
    
    with col6:
        st.markdown("**Output Shaft**")
        fig_out = create_component_3d_view('Output Shaft', {
            'diameter': dout, 'length': comp['output_shaft']['length']
        })
        st.plotly_chart(fig_out, use_container_width=True)
    
    col7, col8 = st.columns(2)
    with col7:
        st.markdown("**Planet Pin**")
        fig_pin = create_component_3d_view('Planet Pin', {
            'diameter': pin['d_pin'], 'length': comp['planet_pin']['total_length']
        })
        st.plotly_chart(fig_pin, use_container_width=True)

# ================================================================
# PASTABLE INPUT CONFIGURATION (FIXED)
# ================================================================
st.divider()
with st.expander("📋 Pastable Input Configuration"):
    st.markdown("**Copy this JSON format to save/load your design configuration:**")
    
    # Use st.code() instead of st.markdown() to avoid triple-quote conflicts
    config_json = json.dumps({
        "input_torque_nm": 5.0,
        "input_speed_rpm": 1500.0,
        "motor_power_kw": 0.785,
        "efficiency": 0.97,
        "design_output_torque_nm": 75.0,
        "target_ratio": 9.0,
        "n_planets": 3,
        "pressure_angle_deg": 20.0,
        "helix_angle_deg": 0.0,
        "max_od_mm": 200.0,
        "sun_teeth": 20,
        "planet_teeth": 70,
        "module_mm": 1.0,
        "face_width_ratio": 12.0,
        "material": "17CrNiMo6 / 18CrNiMo7-6 (Case Carburized)",
        "asme_kb": 1.5,
        "asme_kt": 1.2,
        "keyway_kw": 1.3,
        "bearing_sf": 1.5,
        "pin_span_mm": 30.0,
        "allowable_pin_pressure_mpa": 25.0,
        "force_angle_deg": 120.0
    }, indent=2)
    st.code(config_json, language="json")

# ================================================================
# EXPORT
# ================================================================
st.divider()
st.subheader("📋 Export Design Data")

# Create summary text
summary_text = f"""PLANETARY GEARBOX DESIGN SUMMARY
================================
Configuration: Single-Stage Planetary (Ring Fixed)
Ratio: 1:{ratio_actual:.3f} (Target 1:{ratio_target:.1f})

OPERATING CONDITIONS
- Input Torque: {tin:.2f} N·m
- Input Speed: {nin:.0f} rpm
- Motor Power: {motor_power_kw:.3f} kW
- Output Speed: {output_speed:.1f} rpm
- Design Output Torque: {design_out:.1f} N·m

GEAR TOOTHING
- Sun: {zs} teeth
- Planet: {zp} teeth × {nplanets}
- Ring: {zr} teeth
- Module: {module:.2f} mm
- Face Width: {b:.1f} mm

DIAMETERS (mm)
- Sun Pitch/Base: {sun_geom['pitch_d']:.2f}/{sun_geom['base_d']:.2f}
- Planet Pitch/Base: {planet_geom['pitch_d']:.2f}/{planet_geom['base_d']:.2f}
- Ring Pitch/Base: {ring_geom['pitch_d']:.2f}/{ring_geom['base_d']:.2f}

FORCES (N)
- Tangential SP: {loads['Ft_SP']:.2f}
- Tangential RP: {loads['Ft_RP']:.2f}
- Normal SP: {loads['Fn_SP']:.2f}
- Normal RP: {loads['Fn_RP']:.2f}
- Planet Pin Load: {stress['F_pin']:.2f}

STRESSES (MPa)
- Bending SP: {stress['sigmaF_SP']:.1f}
- Bending RP: {stress['sigmaF_RP']:.1f}
- Combined Planet: {stress['sigmaF_planet']:.1f}
- Contact SP: {stress['sigmaH_SP']:.1f}
- Contact RP: {stress['sigmaH_RP']:.1f}

COMPONENT SIZES
- Input Shaft: ⌀{din:.2f} mm
- Output Shaft: ⌀{dout:.2f} mm
- Planet Pin: ⌀{pin['d_pin']:.2f} mm
- Ring Gear OD: {comp['ring']['outer_d']:.1f} mm
- Carrier OD: {comp['carrier']['plate_od']:.1f} mm
- Housing OD: {comp['housing']['outer_d']:.1f} mm

STATUS: {'PASS ✅' if overall else 'REVIEW REQUIRED ⚠️'}
"""

st.download_button(
    "⬇️ Download Design Summary (TXT)",
    summary_text,
    "planetary_gearbox_summary.txt",
    "text/plain"
)

st.markdown("---")
st.caption("""
**⚠️ DISCLAIMER:** Preliminary design tool using simplified ISO-6336-lite, ASME shaft code, 
and Lundberg-Palmgren bearing life equations. For production use, verify with:
- ISO 6336 (Gear Rating)
- AGMA 2001-B88
- ISO 281 (Bearing Life)
- ASME B106.1M (Shaft Design)
""")
# ================================================================
# OPENSCAD CAD GENERATION
# ================================================================
def generate_openscad_sun_gear(z, m, width, hub_od, bore_d, alpha=20.0):
    """Generate OpenSCAD code for a sun gear (external gear with hub)"""
    pitch_d = z * m
    tip_d = pitch_d + 2 * m
    root_d = pitch_d - 2.5 * m
    return f"""
module sun_gear() {{
    // Sun gear: z={z}, m={m}, face width={width}mm
    difference() {{
        union() {{
            gear_approx(z={z}, m={m}, width={width}, tip_d={tip_d:.2f}, root_d={root_d:.2f});
            // Hub
            cylinder(h={hub_len}, d={hub_od:.2f}, center=true);
        }}
        // Central bore
        cylinder(h={width + 10}, d={bore_d:.2f}, center=true);
    }}
}}
"""

def generate_openscad_planet_gear(z, m, width, hub_od, bore_d):
    """OpenSCAD for planet gear"""
    pitch_d = z * m
    tip_d = pitch_d + 2 * m
    root_d = pitch_d - 2.5 * m
    hub_len = max(width, 30.0)  # approximate
    return f"""
module planet_gear() {{
    difference() {{
        union() {{
            gear_approx(z={z}, m={m}, width={width}, tip_d={tip_d:.2f}, root_d={root_d:.2f});
            cylinder(h={hub_len}, d={hub_od:.2f}, center=true);
        }}
        cylinder(h={width + 10}, d={bore_d:.2f}, center=true);
    }}
}}
"""

def generate_openscad_ring_gear(z, m, width, outer_d, inner_d):
    """OpenSCAD for internal ring gear"""
    pitch_d = z * m
    # For internal gear, tip is smaller, root is larger
    tip_d = pitch_d - 2 * m
    root_d = pitch_d + 2.5 * m
    return f"""
module ring_gear() {{
    difference() {{
        // Outer cylinder
        cylinder(h={width}, d={outer_d:.2f}, center=true);
        // Remove internal teeth cavity (simplified as a cylinder)
        cylinder(h={width + 2}, d={root_d:.2f}, center=true);
        // Add teeth as extrusions? We'll use a gear_approx with negative space
        // For simplicity we just show the ring as a hollow cylinder with rectangular teeth cut.
        // You can replace with a proper internal gear module from OpenSCAD libraries.
        // We'll add a placeholder for teeth.
    }}
}}
"""

def generate_openscad_carrier(pitch_radius, d_pin, n_planets, plate_thickness, hub_od, bore_d):
    """OpenSCAD for carrier plate"""
    pin_d = d_pin
    return f"""
module carrier() {{
    // Carrier plate
    difference() {{
        union() {{
            // Main plate (simplified as a cylinder)
            cylinder(h={plate_thickness}, d={2 * (pitch_radius + d_pin * 1.5):.2f}, center=true);
            // Hub
            cylinder(h={plate_thickness * 2}, d={hub_od:.2f}, center=true);
            // Pin bosses
            for (i = [0 : {n_planets - 1}]) {{
                rotate([0, 0, i * 360 / {n_planets}])
                    translate([{pitch_radius:.2f}, 0, 0])
                    cylinder(h={plate_thickness + 10}, d={pin_d + 6:.2f}, center=true);
            }}
        }}
        // Central bore
        cylinder(h={plate_thickness * 3}, d={bore_d:.2f}, center=true);
        // Pin holes (through bosses)
        for (i = [0 : {n_planets - 1}]) {{
            rotate([0, 0, i * 360 / {n_planets}])
                translate([{pitch_radius:.2f}, 0, 0])
                cylinder(h={plate_thickness + 20}, d={pin_d:.2f}, center=true);
        }}
    }}
}}
"""

def generate_openscad_shaft(diameter, length, key_width=None, key_depth=None):
    """OpenSCAD for a shaft with optional keyway"""
    key_str = ""
    if key_width and key_depth:
        key_str = f"""
    // Keyway
    translate([0, {-diameter/2:.2f}, 0])
        cube([{key_width:.2f}, {key_depth:.2f}, {length}]);
"""
    return f"""
module shaft() {{
    difference() {{
        cylinder(h={length}, d={diameter:.2f}, center=true);
        {key_str}
    }}
}}
"""

def generate_openscad_pin(diameter, length):
    """OpenSCAD for a planet pin"""
    return f"""
module pin() {{
    cylinder(h={length}, d={diameter:.2f}, center=true);
}}
"""

def generate_openscad_assembly(zs, zp, zr, m, n_planets, d_pin, b, 
                               sun_hub_od, sun_bore, 
                               planet_hub_od, planet_bore,
                               ring_outer_d, ring_face_width,
                               carrier_pitch_radius, carrier_plate_thk, carrier_hub_od, carrier_bore,
                               din, dout, pin_span):
    """Generate the full OpenSCAD assembly"""
    # Sun gear params
    sun_tip_d = zs * m + 2 * m
    sun_root_d = zs * m - 2.5 * m
    sun_hub_len = b  # simplified
    # Planet gear params
    planet_tip_d = zp * m + 2 * m
    planet_root_d = zp * m - 2.5 * m
    planet_hub_len = b  # simplified
    # Ring gear params
    ring_tip_d = zr * m - 2 * m
    ring_root_d = zr * m + 2.5 * m
    # Carrier
    carrier_od = 2 * (carrier_pitch_radius + d_pin * 1.5)
    # Shaft lengths
    in_shaft_len = 40 + b + 20
    out_shaft_len = 40 + b + 20 + 30

    code = f"""
// =====================================================
// PLANETARY GEARBOX - OpenSCAD GENERATED CODE
// =====================================================

// Parameters (all in mm)
zs = {zs};       // Sun teeth
zp = {zp};       // Planet teeth
zr = {zr};       // Ring teeth
m = {m};         // Module
n_planets = {n_planets};   // Number of planets
b = {b};         // Face width
d_pin = {d_pin}; // Planet pin diameter
pin_span = {pin_span};     // Pin span

// Gear module (simplified involute approximation)
module gear_approx(z, m, width, tip_d, root_d) {{
    pitch_r = z * m / 2;
    // Tooth width at pitch circle (approximate)
    tooth_angle = 360 / z;
    // Create each tooth
    for (i = [0 : z-1]) {{
        rotate([0, 0, i * tooth_angle])
            translate([pitch_r, 0, 0])
            // Trapezoid tooth shape
            polyhedron(
                points = [
                    [ -tooth_width/2, 0, 0 ],
                    [ tooth_width/2, 0, 0 ],
                    [ tooth_width_top/2, 0, width ],
                    [ -tooth_width_top/2, 0, width ]
                ],
                faces = [[0,1,2,3]]
            );
    }}
    // Add basic cylinder body
    difference() {{
        cylinder(h=width, d=tip_d, center=true);
        cylinder(h=width+2, d=root_d, center=true);
    }}
}}

// Sun gear
module sun_gear() {{
    difference() {{
        union() {{
            gear_approx(z={zs}, m={m}, width={b}, tip_d={sun_tip_d:.2f}, root_d={sun_root_d:.2f});
            cylinder(h={b}, d={sun_hub_od:.2f}, center=true);
        }}
        cylinder(h={b + 10}, d={sun_bore:.2f}, center=true);
    }}
}}

// Planet gear
module planet_gear() {{
    difference() {{
        union() {{
            gear_approx(z={zp}, m={m}, width={b}, tip_d={planet_tip_d:.2f}, root_d={planet_root_d:.2f});
            cylinder(h={b}, d={planet_hub_od:.2f}, center=true);
        }}
        cylinder(h={b + 10}, d={planet_bore:.2f}, center=true);
    }}
}}

// Ring gear (internal)
module ring_gear() {{
    difference() {{
        // Outer body
        cylinder(h={ring_face_width}, d={ring_outer_d:.2f}, center=true);
        // Inner cavity (root diameter)
        cylinder(h={ring_face_width + 2}, d={ring_root_d:.2f}, center=true);
        // Add teeth as extrusions (simplified)
        for (i = [0 : {zr - 1}]) {{
            rotate([0, 0, i * 360 / {zr}])
                translate([0, 0, 0])
                // Add internal tooth block
                translate([0, {ring_root_d/2 - 2:.2f}, 0])
                    cube([ {m}, 4, {ring_face_width} ], center=true);
        }}
    }}
}}

// Carrier plate
module carrier() {{
    difference() {{
        union() {{
            cylinder(h={carrier_plate_thk}, d={carrier_od:.2f}, center=true);
            cylinder(h={carrier_plate_thk * 2}, d={carrier_hub_od:.2f}, center=true);
            for (i = [0 : {n_planets - 1}]) {{
                rotate([0, 0, i * 360 / {n_planets}])
                    translate([{carrier_pitch_radius:.2f}, 0, 0])
                    cylinder(h={carrier_plate_thk + 10}, d={d_pin + 6:.2f}, center=true);
            }}
        }}
        cylinder(h={carrier_plate_thk * 3}, d={carrier_bore:.2f}, center=true);
        for (i = [0 : {n_planets - 1}]) {{
            rotate([0, 0, i * 360 / {n_planets}])
                translate([{carrier_pitch_radius:.2f}, 0, 0])
                cylinder(h={carrier_plate_thk + 20}, d={d_pin:.2f}, center=true);
        }}
    }}
}}

// Input shaft
module input_shaft() {{
    cylinder(h={in_shaft_len}, d={din:.2f}, center=true);
}}

// Output shaft
module output_shaft() {{
    cylinder(h={out_shaft_len}, d={dout:.2f}, center=true);
}}

// Planet pin
module planet_pin() {{
    cylinder(h={pin_span + 10}, d={d_pin:.2f}, center=true);
}}

// Assembly
module assembly() {{
    // Place sun gear at origin
    sun_gear();

    // Place planets around
    for (i = [0 : {n_planets - 1}]) {{
        rotate([0, 0, i * 360 / {n_planets}])
            translate([{carrier_pitch_radius:.2f}, 0, 0])
            planet_gear();
    }}

    // Ring gear (outer)
    ring_gear();

    // Carrier
    carrier();

    // Input shaft (placed below sun)
    translate([0, 0, -{in_shaft_len/2 + b/2:.2f}])
        input_shaft();

    // Output shaft (placed above carrier)
    translate([0, 0, {carrier_plate_thk/2 + b/2:.2f}])
        output_shaft();

    // Planet pins
    for (i = [0 : {n_planets - 1}]) {{
        rotate([0, 0, i * 360 / {n_planets}])
            translate([{carrier_pitch_radius:.2f}, 0, 0])
            planet_pin();
    }}
}}

// Uncomment to render only individual parts
// sun_gear();
// planet_gear();
// ring_gear();
// carrier();
// input_shaft();
// output_shaft();
// planet_pin();

// Render full assembly
assembly();
"""
    return code
