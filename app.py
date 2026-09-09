"""
PLANETARY GEARBOX DESIGNER — COMPLETE PLATFORM
================================================
Single-stage 1:9 ratio planetary gearbox with:
- Customisable sun/planet/ring teeth and module
- Complete component dimensions (thickness, diameters, heights)
- Full force, stress, deflection, and life calculations
- Streamlit web interface with results tabs
"""

import math
import io
import struct
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

# Bearing catalogues (simplified - C_dyn in N, max speed in rpm)
BEARING_CATALOGUE = {
    "SKF 61804 (20×32×7)": {"C": 4200.0, "n_max": 30000.0, "type": "Ball"},
    "SKF 61806 (30×42×7)": {"C": 5100.0, "n_max": 24000.0, "type": "Ball"},
    "SKF 61904 (20×37×9)": {"C": 7800.0, "n_max": 22000.0, "type": "Ball"},
    "SKF 61906 (30×47×9)": {"C": 9900.0, "n_max": 18000.0, "type": "Ball"},
    "SKF 6004 (20×42×12)": {"C": 9400.0, "n_max": 26000.0, "type": "Ball"},
    "SKF 6204 (20×47×14)": {"C": 12800.0, "n_max": 22000.0, "type": "Ball"},
    "INA NKI12/16 (12×24×16)": {"C": 6500.0, "n_max": 18000.0, "type": "Roller"},
    "INA NKI15/20 (15×27×20)": {"C": 8900.0, "n_max": 15000.0, "type": "Roller"},
    "INA NKI20/20 (20×32×20)": {"C": 11200.0, "n_max": 12000.0, "type": "Roller"},
}

# ================================================================
# DATA CLASSES
# ================================================================
@dataclass
class GearGeometry:
    """Complete geometry for a single gear"""
    z: int
    module: float
    pitch_d: float
    base_d: float
    tip_d: float
    root_d: float
    addendum: float
    dedendum: float
    circular_pitch: float
    face_width: float
    
    def to_dict(self):
        return asdict(self)

@dataclass
class ComponentDimensions:
    """Physical dimensions for every component"""
    # Sun gear
    sun_gear: dict
    # Planet gear
    planet_gear: dict
    # Ring gear
    ring_gear: dict
    # Carrier
    carrier: dict
    # Shafts
    input_shaft: dict
    output_shaft: dict
    # Planet pin
    planet_pin: dict
    # Housing
    housing: dict
    
    def to_dataframe(self):
        rows = []
        for component, dims in asdict(self).items():
            for param, value in dims.items():
                rows.append({"Component": component.replace('_', ' ').title(),
                            "Parameter": param.replace('_', ' ').title(),
                            "Value": value, "Unit": "mm"})
        return pd.DataFrame(rows)

# ================================================================
# GEOMETRY CALCULATIONS
# ================================================================
def exact_ring_teeth(zs: int, zp: int) -> int:
    """Ring teeth from coaxiality condition"""
    return zs + 2 * zp

def ratio_ring_fixed(zs: int, zr: int) -> float:
    """Reduction ratio with ring fixed"""
    return (zs + zr) / zs

def assembly_ok(zs: int, zr: int, nplanets: int) -> bool:
    """Planetary assembly condition"""
    return (zs + zr) % nplanets == 0

def planet_spacing_ok(zs: int, zp: int, nplanets: int) -> bool:
    """Check adjacent planet clearance"""
    return (zs + zp) * math.sin(PI / nplanets) > (zp + 2.0)

def gear_geometry(z, m, alpha_deg=20.0, internal=False, beta_deg=0.0):
    """Complete gear geometry including all diameters"""
    beta = math.radians(beta_deg)
    alpha_n = math.radians(alpha_deg)
    alpha_t = math.atan(math.tan(alpha_n) / math.cos(beta))
    
    d = z * m / math.cos(beta)
    db = d * math.cos(alpha_t)
    
    if internal:
        da = d - 2.0 * m / math.cos(beta)
        df = d + 2.5 * m / math.cos(beta)
        addendum = m
        dedendum = 1.25 * m
    else:
        da = d + 2.0 * m / math.cos(beta)
        df = d - 2.5 * m / math.cos(beta)
        addendum = m
        dedendum = 1.25 * m
    
    return {
        "z": z, "module": m, "pitch_d": d, "base_d": db,
        "tip_d": da, "root_d": df, "addendum": addendum,
        "dedendum": dedendum, "alpha_t_deg": math.degrees(alpha_t),
        "circular_pitch": PI * m,
        "face_width": 0.0  # to be set later
    }

def calculate_face_width(module, bm_ratio=12.0):
    """Calculate face width from module and b/m ratio"""
    return bm_ratio * module

# ================================================================
# LOAD & STRESS CALCULATIONS
# ================================================================
def calculate_loads(Tin_Nm, zs, zr, nplanets, m, alpha_deg=20.0, beta_deg=0.0, Kp=1.10):
    """Calculate all mesh forces"""
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    
    # Base circle radii
    rb_s = (zs * m / (2 * math.cos(beta))) * math.cos(alpha)
    rb_r = (zr * m / (2 * math.cos(beta))) * math.cos(alpha)
    
    # Tangential forces (per planet)
    Ft_sp = Tin_Nm * 1000.0 / (nplanets * rb_s) * Kp
    Ft_rp = Ft_sp * (rb_s / rb_r)
    
    # Radial forces
    Fr_sp = Ft_sp * math.tan(alpha)
    Fr_rp = Ft_rp * math.tan(alpha)
    
    # Normal (line-of-action) forces
    Fn_sp = Ft_sp / math.cos(alpha)
    Fn_rp = Ft_rp / math.cos(alpha)
    
    return {
        "Ft_SP": Ft_sp, "Ft_RP": Ft_rp,
        "Fr_SP": Fr_sp, "Fr_RP": Fr_rp,
        "Fn_SP": Fn_sp, "Fn_RP": Fn_rp
    }

def calculate_stress(loads, m, b, zs, zp, zr, material, alpha_deg=20.0,
                     KA=1.25, KV=1.15, KFbeta=1.20, KFalpha=1.00,
                     KHbeta=1.25, KHalpha=1.00, theta_deg=120.0):
    """Calculate bending and contact stresses (ISO-6336 simplified)"""
    
    # Tooth form factors (simplified Lewis-type values)
    YFaS = 2.8; YSaS = 1.55
    YFaP_SP = 2.5; YSaP_SP = 1.60
    YFaP_RP = 2.35; YSaP_RP = 1.60
    Yeps, Ybeta = 0.85, 1.0
    
    # Contact zone factors
    ZH, Zeps, Zbeta = 2.5, 0.90, 1.0
    
    alpha = math.radians(alpha_deg)
    Ftsp, Ftrp = loads["Ft_SP"], loads["Ft_RP"]
    
    # Bending stresses
    common = KA * KV * KFbeta * KFalpha
    sF_sp = (Ftsp * common / (b * m)) * YFaS * YSaS * Yeps * Ybeta
    sF_rp = (Ftrp * common / (b * m)) * YFaP_RP * YSaP_RP * Yeps * Ybeta
    
    # Combined planet bending stress
    theta = math.radians(theta_deg)
    sF_planet = math.sqrt(max(sF_sp**2 + sF_rp**2 - 
                              2 * sF_sp * sF_rp * math.cos(theta), 0.0))
    
    # Contact stresses
    ZE = math.sqrt(1.0 / (PI * ((1 - material["nu"]**2) / material["E"] +
                                (1 - material["nu"]**2) / material["E"])))
    
    u_sp = zp / zs
    u_rp = zr / zp
    dS, dP = zs * m, zp * m
    
    term_sp = (Ftsp * KA * KV * KHbeta * KHalpha) / (b * dS) * ((u_sp + 1) / u_sp)
    term_rp = (Ftrp * KA * KV * KHbeta * KHalpha) / (b * dP) * (max(u_rp - 1, 1e-9) / u_rp)
    
    sH_sp = ZH * ZE * Zeps * Zbeta * math.sqrt(max(term_sp, 0.0))
    sH_rp = ZH * ZE * Zeps * Zbeta * math.sqrt(max(term_rp, 0.0))
    
    # Resultant planet pin load
    Fnsp, Fnrp = loads["Fn_SP"], loads["Fn_RP"]
    Fpin = math.sqrt(max(Fnsp**2 + Fnrp**2 - 
                         2 * Fnsp * Fnrp * math.cos(theta), 0.0))
    
    # Safety factors
    sf_bending_sun = material["sigmaF"] / max(sF_sp, 1e-9)
    sf_bending_planet = material["sigmaF"] / max(sF_planet, 1e-9)
    sf_contact_sun = material["sigmaH"] / max(sH_sp, 1e-9)
    sf_contact_ring = material["sigmaH"] / max(sH_rp, 1e-9)
    
    return {
        "sigmaF_SP": sF_sp, "sigmaF_RP": sF_rp, "sigmaF_planet": sF_planet,
        "sigmaH_SP": sH_sp, "sigmaH_RP": sH_rp,
        "SF_bending_sun": sf_bending_sun, "SF_bending_planet": sf_bending_planet,
        "SF_contact_sun": sf_contact_sun, "SF_contact_ring": sf_contact_ring,
        "F_pin": Fpin
    }

# ================================================================
# COMPONENT SIZING CALCULATIONS
# ================================================================
def calculate_component_dimensions(zs, zp, zr, m, b, d_sun, d_planet, d_ring,
                                   d_in_shaft, d_out_shaft, d_pin, max_od=200.0,
                                   pin_span=30.0, nplanets=3):
    """Calculate complete physical dimensions for every component"""
    
    result = {}
    
    # ---- Sun Gear ----
    sun_bore = d_in_shaft + 2.0
    sun_hub_od = max(sun_bore + 6.0, d_in_shaft * 1.6)
    sun_hub_len = max(1.2 * d_in_shaft, b)
    sun_face_width = b
    sun_rim_thick = (sun_hub_od - sun_bore) / 2.0
    sun_web_thick = max(4.0, 0.15 * b)
    sun_hub_diameter = sun_hub_od
    result["sun_gear"] = {
        "face_width": sun_face_width,
        "bore_d": sun_bore,
        "hub_od": sun_hub_od,
        "hub_length": sun_hub_len,
        "rim_thickness": sun_rim_thick,
        "web_thickness": sun_web_thick,
        "hub_diameter": sun_hub_diameter,
        "total_height": b + sun_hub_len
    }
    
    # ---- Planet Gear ----
    planet_bore = d_pin + 2.0
    planet_hub_od = max(planet_bore + 4.0, d_pin * 1.45)
    planet_hub_len = max(b, pin_span * 0.75)
    planet_face_width = b
    planet_rim_thick = (planet_hub_od - planet_bore) / 2.0
    planet_hub_diameter = planet_hub_od
    result["planet_gear"] = {
        "face_width": planet_face_width,
        "bore_d": planet_bore,
        "hub_od": planet_hub_od,
        "hub_length": planet_hub_len,
        "rim_thickness": planet_rim_thick,
        "hub_diameter": planet_hub_diameter,
        "total_height": b + planet_hub_len
    }
    
    # ---- Ring Gear ----
    ring_tooth_tip_d = d_ring - 2.0 * m
    ring_tooth_root_d = d_ring + 2.5 * m
    ring_radial_rim = max(0.18 * d_ring, 8.0)
    ring_outer_d = ring_tooth_root_d + 2.0 * ring_radial_rim
    ring_inner_d = ring_tooth_root_d
    ring_wall = (ring_outer_d - ring_inner_d) / 2.0
    ring_face_width = b + 2.0
    ring_height = ring_face_width
    ring_stiffener_thick = max(4.0, 0.12 * ring_face_width)
    ring_bolting_flange_od = ring_outer_d + 8.0
    ring_flange_thick = max(5.0, 0.08 * ring_face_width)
    result["ring_gear"] = {
        "inner_tooth_tip_d": ring_tooth_tip_d,
        "inner_tooth_root_d": ring_tooth_root_d,
        "outer_d": ring_outer_d,
        "wall_thickness": ring_wall,
        "face_width": ring_face_width,
        "total_height": ring_height,
        "stiffener_thickness": ring_stiffener_thick,
        "flange_od": ring_bolting_flange_od,
        "flange_thickness": ring_flange_thick
    }
    
    # ---- Carrier ----
    carrier_pitch_radius = (d_sun + d_planet) / 2.0
    carrier_arm_radius = carrier_pitch_radius
    carrier_plate_thk = max(6.0, 0.30 * b, 0.20 * d_pin)
    carrier_od = min(max_od - 4.0, 2.0 * (carrier_pitch_radius + 1.6 * d_pin))
    carrier_bore = d_out_shaft + 2.0
    carrier_hub_od = max(d_out_shaft + 8.0, d_out_shaft * 1.5)
    carrier_hub_len = max(1.5 * d_out_shaft, 1.5 * b)
    carrier_pin_boss_od = d_pin + 6.0
    carrier_pin_boss_len = max(6.0, 0.5 * d_pin)
    carrier_arm_width = max(6.0, 0.25 * carrier_plate_thk)
    carrier_arm_depth = max(8.0, 0.8 * d_pin)
    carrier_total_height = carrier_plate_thk + carrier_hub_len + 2 * carrier_pin_boss_len
    result["carrier"] = {
        "plate_thickness": carrier_plate_thk,
        "plate_od": carrier_od,
        "output_bore": carrier_bore,
        "hub_od": carrier_hub_od,
        "hub_length": carrier_hub_len,
        "planet_pin_circle_d": 2.0 * carrier_pitch_radius,
        "arm_center_radius": carrier_arm_radius,
        "pin_boss_od": carrier_pin_boss_od,
        "pin_boss_length": carrier_pin_boss_len,
        "arm_width": carrier_arm_width,
        "arm_depth": carrier_arm_depth,
        "total_height": carrier_total_height
    }
    
    # ---- Input Shaft ----
    input_shaft_len = 40.0 + b + 20.0
    input_shaft_d = d_in_shaft
    input_key_width = 0.25 * d_in_shaft if d_in_shaft < 30.0 else 8.0
    input_key_depth = 0.15 * d_in_shaft if d_in_shaft < 30.0 else 5.0
    input_shaft_shoulder_d = d_in_shaft * 1.2
    result["input_shaft"] = {
        "diameter": input_shaft_d,
        "length": input_shaft_len,
        "key_width": input_key_width,
        "key_depth": input_key_depth,
        "shoulder_diameter": input_shaft_shoulder_d
    }
    
    # ---- Output Shaft ----
    output_shaft_len = 40.0 + b + 20.0 + 30.0
    output_shaft_d = d_out_shaft
    output_key_width = 0.25 * d_out_shaft if d_out_shaft < 30.0 else 10.0
    output_key_depth = 0.15 * d_out_shaft if d_out_shaft < 30.0 else 6.0
    output_shaft_shoulder_d = d_out_shaft * 1.2
    result["output_shaft"] = {
        "diameter": output_shaft_d,
        "length": output_shaft_len,
        "key_width": output_key_width,
        "key_depth": output_key_depth,
        "shoulder_diameter": output_shaft_shoulder_d
    }
    
    # ---- Planet Pin ----
    pin_head_d = d_pin * 1.3
    pin_head_thick = max(3.0, 0.2 * d_pin)
    pin_total_len = pin_span + 2.0 * pin_head_thick
    result["planet_pin"] = {
        "diameter": d_pin,
        "span_length": pin_span,
        "head_diameter": pin_head_d,
        "head_thickness": pin_head_thick,
        "total_length": pin_total_len
    }
    
    # ---- Housing ----
    housing_clearance = 2.0
    envelope_od = ring_outer_d + 2.0 * housing_clearance
    housing_wall = max(3.0, 0.06 * envelope_od)
    housing_length = input_shaft_len + b + output_shaft_len
    housing_flange_od = envelope_od + 2.0 * housing_wall
    housing_base_thick = max(5.0, 0.15 * housing_wall)
    result["housing"] = {
        "outer_d": envelope_od,
        "wall_thickness": housing_wall,
        "length": housing_length,
        "flange_od": housing_flange_od,
        "base_thickness": housing_base_thick,
        "inner_d": envelope_od - 2.0 * housing_wall
    }
    
    return result

# ================================================================
# DEFLECTION CALCULATIONS
# ================================================================
def calculate_deflections(loads, b, m, zs, zr, material, d_in_shaft, d_out_shaft,
                          pin_span, d_pin, carrier_plate_thk):
    """Calculate deflections for critical components"""
    
    E = material["E"]
    
    # Sun gear deflection (simplified - radial deflection under load)
    rb_s = (zs * m / 2.0) * math.cos(math.radians(DEFAULT_ALPHA))
    Ft_sp = loads["Ft_SP"]
    # Simplification: sun deflection approximated as cantilever
    sun_span = b * 0.5
    I_sun = PI * rb_s**4 / 4.0
    sun_deflection = (Ft_sp * sun_span**3) / (3 * E * I_sun) * 1000.0  # mm
    
    # Ring gear deflection
    ring_span = b * 0.5
    rb_r = (zr * m / 2.0) * math.cos(math.radians(DEFAULT_ALPHA))
    I_ring = PI * (rb_r**4 - (rb_r - 0.18 * rb_r)**4) / 4.0
    Ft_rp = loads["Ft_RP"]
    ring_deflection = (Ft_rp * ring_span**3) / (3 * E * I_ring) * 1000.0  # mm
    
    # Carrier deflection
    carrier_span = pin_span * 0.75
    F_pin = loads["Fn_SP"] + loads["Fn_RP"]  # approximation
    I_carrier = (carrier_plate_thk**3 / 12.0) * 20.0  # approximate
    carrier_deflection = (F_pin * carrier_span**3) / (48 * E * I_carrier) * 1000.0  # mm
    
    # Shaft torsional deflection
    G = E / (2 * (1 + material["nu"]))
    T_in = loads["Ft_SP"] * (zs * m / 2.0) / 1000.0  # N.m
    T_out = loads["Ft_SP"] * (zr * m / 2.0) / 1000.0  # N.m
    
    J_in = PI * d_in_shaft**4 / 32.0
    J_out = PI * d_out_shaft**4 / 32.0
    
    L_in = 40.0  # mm
    L_out = 40.0 + b  # mm
    
    theta_in = (T_in * 1000.0 * L_in) / (G * J_in)  # rad
    theta_out = (T_out * 1000.0 * L_out) / (G * J_out)  # rad
    
    theta_in_deg = math.degrees(theta_in)
    theta_out_deg = math.degrees(theta_out)
    
    # Tooth deflection (simplified - elastic deformation at contact)
    F_n = loads["Fn_SP"]
    tooth_height = 2.5 * m
    tooth_width = m
    I_tooth = tooth_width * tooth_height**3 / 12.0
    tooth_deflection = (F_n * tooth_height**3) / (3 * E * I_tooth) * 1000.0  # mm
    
    return {
        "sun_deflection": sun_deflection,
        "ring_deflection": ring_deflection,
        "carrier_deflection": carrier_deflection,
        "shaft_torsional_angle_input": theta_in_deg,
        "shaft_torsional_angle_output": theta_out_deg,
        "tooth_deflection": tooth_deflection
    }

# ================================================================
# SHAFT & PIN SIZING
# ================================================================
def shaft_diameter(T_Nm, M_Nm, tau_allow, Kb=1.5, Kt=1.2, Kw=1.3):
    """ASME combined torsion + bending shaft sizing"""
    T = T_Nm * 1000.0
    M = M_Nm * 1000.0
    Te = math.sqrt((Kb * M)**2 + (Kt * Kw * T)**2)
    d = (16.0 * Te / (PI * tau_allow)) ** (1.0 / 3.0)
    return d, Te / 1000.0

def pin_design(F, support_span, sigma_b, tau, bearing_p, safety=1.5):
    """Planet pin sizing"""
    Fd = F * safety
    Mmax = Fd * support_span / 4.0
    V = Fd / 2.0
    
    db = (32.0 * Mmax / (PI * sigma_b)) ** (1.0 / 3.0)
    ds = math.sqrt(4.0 * V / (PI * tau))
    d = max(db, ds)
    
    p = Fd / (d * support_span * 0.55)
    
    return {
        "F_design": Fd, "Mmax": Mmax, "V": V,
        "d_bend": db, "d_shear": ds, "d_pin": d,
        "p_bearing": p, "pressure_ok": p <= bearing_p
    }

# ================================================================
# BEARING LIFE CALCULATIONS
# ================================================================
def bearing_L10_life(C_dyn, P_equiv, n_rpm, bearing_type="Ball"):
    """Lundberg-Palmgren L10 life calculation"""
    p = 3.0 if bearing_type == "Ball" else 10.0 / 3.0
    if P_equiv <= 0 or n_rpm <= 0:
        return {"L10_Mrev": float('inf'), "L10_h": float('inf')}
    L10_Mrev = (C_dyn / P_equiv) ** p
    L10_h = (L10_Mrev * 1e6) / (60.0 * n_rpm)
    return {"L10_Mrev": L10_Mrev, "L10_h": L10_h}

# ================================================================
# THERMAL CALCULATION
# ================================================================
def calculate_thermal(power_loss_W, housing_area_m2, ambient_C=25.0, h_coeff=15.0):
    """Calculate steady-state temperature rise"""
    # Power loss in Watts, housing area in m²
    delta_T = power_loss_W / (h_coeff * housing_area_m2)
    T_steady = ambient_C + delta_T
    return {
        "power_loss_W": power_loss_W,
        "housing_area_m2": housing_area_m2,
        "delta_T": delta_T,
        "steady_state_T": T_steady,
        "thermal_ok": T_steady < 90.0  # Typical limit for mineral oil
    }

# ================================================================
# PLOTTING FUNCTIONS
# ================================================================
def create_gearbox_schematic(geom_sun, geom_planet, geom_ring, nplanets, d_pin):
    """Create a 2D schematic of the planetary gearbox"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    rS = geom_sun["pitch_d"] / 2.0
    rP = geom_planet["pitch_d"] / 2.0
    rR = geom_ring["pitch_d"] / 2.0
    rCarrier = rS + rP
    
    # Draw ring gear
    theta = np.linspace(0, 2*PI, 200)
    ring_outer = rR + 8.0
    ring_inner = rR - 8.0
    ax.fill(ring_outer * np.cos(theta), ring_outer * np.sin(theta),
            color='gray', alpha=0.7, label='Ring Gear')
    ax.fill(ring_inner * np.cos(theta), ring_inner * np.sin(theta),
            color='white', alpha=0.9)
    
    # Draw sun gear
    ax.fill(rS * np.cos(theta), rS * np.sin(theta), color='#D9531E', label='Sun')
    
    # Draw planets
    for k in range(nplanets):
        angle = k * 2 * PI / nplanets
        pX = rCarrier * math.cos(angle)
        pY = rCarrier * math.sin(angle)
        
        # Planet gear
        planet_theta = np.linspace(0, 2*PI, 100)
        planet_x = pX + rP * np.cos(planet_theta)
        planet_y = pY + rP * np.sin(planet_theta)
        ax.fill(planet_x, planet_y, color='#EDB120', edgecolor='k', label='Planet' if k == 0 else "")
        
        # Planet pin
        pin_r = d_pin / 2.0
        pin_x = pX + pin_r * np.cos(planet_theta)
        pin_y = pY + pin_r * np.sin(planet_theta)
        ax.fill(pin_x, pin_y, color='#3B3B3B', label='Pin' if k == 0 else "")
        
        # Carrier arm
        ax.plot([0, pX], [0, pY], 'b-', linewidth=2)
    
    # Carrier
    ax.plot(0, 0, 'k+', markersize=12, markeredgewidth=2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_xlim([-ring_outer*1.15, ring_outer*1.15])
    ax.set_ylim([-ring_outer*1.15, ring_outer*1.15])
    ax.set_title(f"Planetary Gearbox - 1:{DEFAULT_RATIO:.0f} (Ring Fixed)")
    
    return fig

# ================================================================
# HELPER FUNCTION TO CONVERT DICTIONARY TO DATAFRAME
# ================================================================
def components_to_dataframe(components_dict):
    """Convert component dimensions dictionary to DataFrame"""
    rows = []
    for component, dims in components_dict.items():
        for param, value in dims.items():
            rows.append({
                "Component": component.replace('_', ' ').title(),
                "Parameter": param.replace('_', ' ').title(),
                "Value": value,
                "Unit": "mm"
            })
    return pd.DataFrame(rows)

# ================================================================
# STREAMLIT APP
# ================================================================
st.set_page_config(
    page_title="Planetary Gearbox Designer",
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
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border: 1px solid #e0e0e0;
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

st.markdown('<div class="main-header">⚙️ PLANETARY GEARBOX DESIGN PLATFORM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Complete Single-Stage 1:9 Reduction — All Component Calculations</div>', unsafe_allow_html=True)

# ================================================================
# SIDEBAR — INPUTS
# ================================================================
with st.sidebar:
    st.header("🔧 Operating Conditions")
    
    # Operating inputs
    tin = st.number_input("Input Torque (N·m)", min_value=1.0, max_value=500.0, 
                          value=5.0, step=0.5, help="Torque at the input shaft")
    nin = st.number_input("Input Speed (rpm)", min_value=100, max_value=20000,
                          value=1500, step=50, help="Speed at the input shaft")
    
    # Calculated motor power
    motor_power = tin * 2 * PI * nin / 60.0 / 1000.0  # kW
    st.metric("Calculated Motor Power", f"{motor_power:.3f} kW")
    
    motor_power_manual = st.number_input("Override Motor Power (kW)", 
                                         min_value=0.1, max_value=100.0,
                                         value=round(motor_power, 3), step=0.01)
    
    design_out = st.slider("Design Output Torque (N·m)", 65.0, 75.0, 75.0, 1.0,
                           help="Design basis for all strength calculations")
    
    st.header("⚙️ Gear Parameters")
    ratio_target = st.number_input("Target Ratio", 5.0, 15.0, 9.0, 0.1)
    
    # Auto tooth suggestion or manual
    auto_tooth = st.checkbox("Auto-select teeth", value=True)
    
    if auto_tooth:
        # Try common combinations for 1:9
        candidates = []
        for zs in [18, 19, 20, 21, 22, 24, 25, 26, 28, 30]:
            for zp in range(17, 100):
                zr = exact_ring_teeth(zs, zp)
                ratio = ratio_ring_fixed(zs, zr)
                if abs(ratio - ratio_target) < 1e-9:
                    if assembly_ok(zs, zr, DEFAULT_PLANETS) and planet_spacing_ok(zs, zp, DEFAULT_PLANETS):
                        candidates.append((zs, zp, zr))
        
        if candidates:
            st.write("**Available tooth combinations:**")
            combo_df = pd.DataFrame(candidates, columns=["Sun", "Planet", "Ring"])
            selected_idx = st.selectbox(
                "Select combination",
                range(len(candidates)),
                format_func=lambda x: f"S={candidates[x][0]}, P={candidates[x][1]}, R={candidates[x][2]}"
            )
            zs, zp, zr = candidates[selected_idx]
        else:
            st.warning("No exact 1:9 combo found. Using closest match.")
            zs, zp = 20, 70
            zr = exact_ring_teeth(zs, zp)
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            zs = st.number_input("Sun Teeth", 17, 120, 20, 1)
        with col2:
            zp = st.number_input("Planet Teeth", 17, 150, 70, 1)
        with col3:
            zr = exact_ring_teeth(int(zs), int(zp))
            st.metric("Ring Teeth", zr)
    
    # Module selection
    st.subheader("Module Selection")
    module = st.select_slider("Module (mm)", options=MODULES, value=1.0)
    
    # Show resulting diameters
    st.write(f"**Sun pitch dia:** {zs * module:.1f} mm")
    st.write(f"**Planet pitch dia:** {zp * module:.1f} mm")
    st.write(f"**Ring pitch dia:** {zr * module:.1f} mm")
    
    # Face width
    bm_ratio = st.slider("Face Width / Module Ratio", 8.0, 20.0, 12.0, 0.5)
    b = bm_ratio * module
    st.write(f"**Face Width:** {b:.1f} mm")
    
    st.header("📐 Material Selection")
    mat_name = st.selectbox("Material", list(MATERIALS.keys()))
    mat = dict(MATERIALS[mat_name])
    
    if mat_name == "Custom":
        with st.expander("Custom Material Properties", expanded=True):
            mat["E"] = st.number_input("Young's Modulus E (MPa)", 100000.0, 300000.0, 200000.0)
            mat["nu"] = st.number_input("Poisson's Ratio ν", 0.2, 0.4, 0.3, 0.01)
            mat["tau"] = st.number_input("Allowable Shear τ (MPa)", 50.0, 500.0, 180.0)
            mat["sigmaF"] = st.number_input("Bending Fatigue Limit σF (MPa)", 100.0, 1000.0, 300.0)
            mat["sigmaH"] = st.number_input("Contact Fatigue Limit σH (MPa)", 300.0, 2500.0, 1200.0)
            mat["sigma_allow_bend"] = st.number_input("Pin Allowable Bending (MPa)", 50.0, 800.0, 250.0)
            mat["density"] = st.number_input("Density (kg/m³)", 7000.0, 9000.0, 7850.0)
            mat["hardness_HB"] = st.number_input("Hardness (HB)", 100.0, 800.0, 500.0)
    
    st.header("🔩 Pin & Bearing Parameters")
    pin_span = st.number_input("Planet Pin Span (mm)", 10.0, 100.0, 30.0, 1.0)
    bearing_p = st.number_input("Allowable Bearing Pressure (MPa)", 5.0, 80.0, 25.0, 1.0)
    service_factor = st.number_input("Bearing Service Factor", 1.0, 3.0, 1.5, 0.1)
    
    st.header("🔧 Design Factors")
    Kb = st.number_input("ASME Bending Factor Kb", 1.0, 3.0, 1.5, 0.1)
    Kt = st.number_input("ASME Torsion Factor Kt", 1.0, 2.5, 1.2, 0.1)
    Kw = st.number_input("Keyway Factor Kw", 1.0, 2.0, 1.3, 0.05)
    theta_deg = st.slider("Mesh Force Angle θ (deg)", 60.0, 180.0, 120.0, 5.0)
    
    st.header("📏 Envelope Constraints")
    max_od = st.number_input("Max OD (mm)", 100.0, 500.0, 200.0, 10.0)

# ================================================================
# CALCULATION ENGINE
# ================================================================
# Geometry
geom_sun = gear_geometry(zs, module, DEFAULT_ALPHA, False)
geom_planet = gear_geometry(zp, module, DEFAULT_ALPHA, False)
geom_ring = gear_geometry(zr, module, DEFAULT_ALPHA, True)

# Set face widths
geom_sun["face_width"] = b
geom_planet["face_width"] = b
geom_ring["face_width"] = b + 2.0

# Ratio check
ratio_actual = ratio_ring_fixed(zs, zr)
ratio_ok = abs(ratio_actual - ratio_target) < 1e-9

# Assembly checks
assembly = assembly_ok(zs, zr, DEFAULT_PLANETS)
clearance = planet_spacing_ok(zs, zp, DEFAULT_PLANETS)

# Loads
Tin_design = design_out / (ratio_actual * DEFAULT_ETA)
loads = calculate_loads(Tin_design, zs, zr, DEFAULT_PLANETS, module, DEFAULT_ALPHA)

# Stresses
stress = calculate_stress(loads, module, b, zs, zp, zr, mat, 
                          alpha_deg=DEFAULT_ALPHA, theta_deg=theta_deg)

# Shaft sizing
din, Te_in = shaft_diameter(Tin_design, 0.0, mat["tau"], Kb, Kt, Kw)
dout, Te_out = shaft_diameter(design_out, 0.0, mat["tau"], Kb, Kt, Kw)

# Pin design
pin = pin_design(stress["F_pin"], pin_span, mat["sigma_allow_bend"], 
                 mat["tau"], bearing_p, safety=service_factor)

# Component dimensions
comps = calculate_component_dimensions(
    zs, zp, zr, module, b,
    geom_sun["pitch_d"], geom_planet["pitch_d"], geom_ring["pitch_d"],
    din, dout, pin["d_pin"], max_od, pin_span, DEFAULT_PLANETS
)

# Deflections
deflections = calculate_deflections(loads, b, module, zs, zr, mat,
                                    din, dout, pin_span, pin["d_pin"],
                                    comps["carrier"]["plate_thickness"])

# Bearing life
main_bearing_radial = math.hypot(loads["Fr_SP"], loads["Ft_SP"]) * service_factor
n_planet_rel = abs(nin - nin / ratio_actual) * zs / zp

# Thermal calculation
power_loss = (1 - DEFAULT_ETA) * motor_power_manual * 1000.0  # W
housing_area = PI * comps["housing"]["outer_d"] / 1000.0 * comps["housing"]["length"] / 1000.0
thermal = calculate_thermal(power_loss, housing_area)

# Bearing life (main and planet)
main_bearing_life = bearing_L10_life(12000.0, main_bearing_radial, nin, "Ball")
planet_bearing_life = bearing_L10_life(6000.0, stress["F_pin"] * service_factor, 
                                       n_planet_rel, "Roller")

# Overall status
sfF = mat["sigmaF"] / max(stress["sigmaF_planet"], 1e-9)
sfH_sp = mat["sigmaH"] / max(stress["sigmaH_SP"], 1e-9)
sfH_rp = mat["sigmaH"] / max(stress["sigmaH_RP"], 1e-9)

od_ok = comps["housing"]["outer_d"] <= max_od
checks = [
    ("Ratio (1:9)", ratio_ok),
    ("Assembly Condition", assembly),
    ("Planet Clearance", clearance),
    ("OD Envelope", od_ok),
    ("Gear Bending SF", sfF >= 1.0),
    ("Sun Contact SF", sfH_sp >= 1.0),
    ("Ring Contact SF", sfH_rp >= 1.0),
    ("Planet Pin Pressure", pin["pressure_ok"]),
    ("Deflection Limit", deflections["sun_deflection"] < 0.1),
    ("Thermal", thermal["thermal_ok"]),
]

overall_ok = all(ok for _, ok in checks)

# ================================================================
# MAIN DASHBOARD
# ================================================================
# Header metrics
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
with col_m1:
    st.metric("Achieved Ratio", f"1:{ratio_actual:.3f}")
with col_m2:
    st.metric("Output Speed", f"{nin / ratio_actual:.0f} rpm")
with col_m3:
    st.metric("Motor Power", f"{motor_power_manual:.3f} kW")
with col_m4:
    st.metric("Design Torque", f"{design_out:.1f} N·m")
with col_m5:
    if overall_ok:
        st.markdown('<div class="pass-badge">PASS ✓</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="fail-badge">CHECK ⚠️</div>', unsafe_allow_html=True)

st.divider()

# ================================================================
# TABS
# ================================================================
tabs = st.tabs([
    "📊 Summary",
    "⚙️ Tooth Sizing",
    "📐 Geometry",
    "🔄 Kinematics",
    "🧲 Forces & Stress",
    "🔩 Shafts & Pins",
    "🧱 Component Dimensions",
    "📏 Deflections",
    "🛞 Bearings",
    "🔥 Thermal",
    "✅ Design Checks",
    "📊 Charts & Plots"
])

# ---------------- TAB 1: SUMMARY ----------------
with tabs[0]:
    st.subheader("📊 Complete Design Summary")
    
    summary_data = {
        "Parameter": [
            "Configuration", "Input Member", "Output Member", "Fixed Member",
            "Target Ratio", "Actual Ratio", "Ratio Error",
            "Input Torque (design)", "Input Speed", "Output Speed",
            "Module", "Face Width", "Number of Planets",
            "Material", "Envelope OD", "Status"
        ],
        "Value": [
            "Single-Stage Planetary", "Sun", "Carrier", "Ring",
            f"1:{ratio_target:.1f}", f"1:{ratio_actual:.3f}", 
            f"{abs(ratio_actual - ratio_target) * 100:.2f}%",
            f"{Tin_design:.2f} N·m", f"{nin:.0f} rpm", f"{nin / ratio_actual:.1f} rpm",
            f"{module:.2f} mm", f"{b:.1f} mm", f"{DEFAULT_PLANETS}",
            mat_name, f"{comps['housing']['outer_d']:.1f} mm",
            "✅ PASS" if overall_ok else "⚠️ REVIEW"
        ]
    }
    
    st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)
    
    # Key component sizes
    st.subheader("Key Component Sizes")
    key_sizes = {
        "Component": ["Sun Gear", "Planet Gear", "Ring Gear", "Carrier", 
                      "Input Shaft", "Output Shaft", "Planet Pin", "Housing"],
        "Key Dimension": [
            f"⌀{comps['sun_gear']['hub_od']:.1f} × {comps['sun_gear']['total_height']:.1f}",
            f"⌀{comps['planet_gear']['hub_od']:.1f} × {comps['planet_gear']['total_height']:.1f}",
            f"⌀{comps['ring_gear']['outer_d']:.1f} × {comps['ring_gear']['total_height']:.1f}",
            f"⌀{comps['carrier']['plate_od']:.1f} × {comps['carrier']['total_height']:.1f}",
            f"⌀{comps['input_shaft']['diameter']:.1f} × {comps['input_shaft']['length']:.1f}",
            f"⌀{comps['output_shaft']['diameter']:.1f} × {comps['output_shaft']['length']:.1f}",
            f"⌀{comps['planet_pin']['diameter']:.1f} × {comps['planet_pin']['total_length']:.1f}",
            f"⌀{comps['housing']['outer_d']:.1f} × {comps['housing']['length']:.1f}"
        ]
    }
    st.dataframe(pd.DataFrame(key_sizes), hide_index=True, use_container_width=True)

# ---------------- TAB 2: TOOTH SIZING ----------------
with tabs[1]:
    st.subheader("⚙️ Gear Tooth Selection")
    
    st.markdown(f"""
    **Selected Configuration:**
    - Sun: **{zs} teeth**
    - Planet: **{zp} teeth** (× {DEFAULT_PLANETS})
    - Ring: **{zr} teeth**
    - Module: **{module:.2f} mm**
    
    **Verification Equations:**
    - Coaxiality: R = S + 2P → {zs} + 2×{zp} = {zs + 2*zp} ✓
    - Ratio (Ring Fixed): i = (S + R)/S = ({zs} + {zr})/{zs} = {ratio_actual:.3f}
    - Assembly: (S + R) mod N = ({zs} + {zr}) mod {DEFAULT_PLANETS} = {(zs + zr) % DEFAULT_PLANETS} → {'✓' if assembly else '✗'}
    - Clearance: (S + P)sin(180°/N) > P + 2 → {(zs + zp) * math.sin(PI / DEFAULT_PLANETS):.1f} > {zp + 2} → {'✓' if clearance else '✗'}
    """)
    
    # Show other possible tooth combinations
    st.subheader("Other Valid 1:9 Combinations")
    candidates = []
    for zs_try in range(17, 60):
        for zp_try in range(17, 100):
            zr_try = exact_ring_teeth(zs_try, zp_try)
            ratio_try = ratio_ring_fixed(zs_try, zr_try)
            if abs(ratio_try - ratio_target) < 1e-9:
                if assembly_ok(zs_try, zr_try, DEFAULT_PLANETS) and planet_spacing_ok(zs_try, zp_try, DEFAULT_PLANETS):
                    candidates.append({
                        "Sun": zs_try, "Planet": zp_try, "Ring": zr_try,
                        "Ratio": f"1:{ratio_try:.3f}",
                        "Module 1.0 OD": f"{(zr_try + 2.5) * 1.0:.1f} mm",
                        "Module 1.5 OD": f"{(zr_try + 2.5) * 1.5:.1f} mm",
                        "Module 2.0 OD": f"{(zr_try + 2.5) * 2.0:.1f} mm"
                    })
    
    if candidates:
        st.dataframe(pd.DataFrame(candidates), hide_index=True, use_container_width=True)

# ---------------- TAB 3: GEOMETRY ----------------
with tabs[2]:
    st.subheader("📐 Complete Gear Geometry")
    
    geo_data = []
    for gear_name, geom in [("Sun", geom_sun), ("Planet", geom_planet), ("Ring", geom_ring)]:
        geo_data.append({
            "Gear": gear_name,
            "Teeth (z)": geom["z"],
            "Module (mm)": geom["module"],
            "Pitch Dia (mm)": f"{geom['pitch_d']:.3f}",
            "Base Dia (mm)": f"{geom['base_d']:.3f}",
            "Tip Dia (mm)": f"{geom['tip_d']:.3f}",
            "Root Dia (mm)": f"{geom['root_d']:.3f}",
            "Addendum (mm)": f"{geom['addendum']:.3f}",
            "Dedendum (mm)": f"{geom['dedendum']:.3f}",
            "Circular Pitch (mm)": f"{geom['circular_pitch']:.3f}",
            "Face Width (mm)": f"{geom['face_width']:.2f}"
        })
    
    st.dataframe(pd.DataFrame(geo_data), hide_index=True, use_container_width=True)
    
    # Center distances
    st.subheader("Center Distances")
    a_sun_planet = (geom_sun["pitch_d"] + geom_planet["pitch_d"]) / 2.0
    a_ring_planet = (geom_ring["pitch_d"] - geom_planet["pitch_d"]) / 2.0
    
    st.markdown(f"""
    - **Sun–Planet Center Distance:** {a_sun_planet:.3f} mm
    - **Ring–Planet Center Distance:** {a_ring_planet:.3f} mm
    - **Verification:** {a_sun_planet:.3f} = {a_ring_planet:.3f} → {'✓' if abs(a_sun_planet - a_ring_planet) < 0.01 else '✗'}
    """)

# ---------------- TAB 4: KINEMATICS ----------------
with tabs[3]:
    st.subheader("🔄 Kinematics")
    
    output_speed = nin / ratio_actual
    n_planet_rel = abs(nin - output_speed) * zs / zp
    
    kin_data = {
        "Member": ["Sun (Input)", "Ring (Fixed)", "Carrier (Output)", 
                   "Planet (Relative to Carrier)", "Planet (Absolute)"],
        "Speed (rpm)": [
            f"{nin:.1f}",
            "0.0",
            f"{output_speed:.2f}",
            f"{n_planet_rel:.2f}",
            f"{output_speed + n_planet_rel:.2f}"
        ],
        "Direction": ["Same as input", "Stationary", "Reduced speed", 
                      "Opposite to carrier", "Compound motion"]
    }
    
    st.dataframe(pd.DataFrame(kin_data), hide_index=True, use_container_width=True)
    
    st.markdown(f"""
    **Speed Relationships:**
    - **Willis Equation:** (n_s - n_c) / (n_r - n_c) = -z_r / z_s = -{zr}/{zs} = {-zr/zs:.3f}
    - **Planet Spin:** n_p = -n_c × (z_s/z_p) + (n_s - n_c) × (z_s/z_p) = **{n_planet_rel:.2f} rpm**
    - **Output Speed:** n_out = n_in / i = {nin} / {ratio_actual:.3f} = **{output_speed:.2f} rpm**
    """)

# ---------------- TAB 5: FORCES & STRESS ----------------
with tabs[4]:
    st.subheader("🧲 Mesh Forces")
    
    force_data = {
        "Force": [
            "Tangential Ft (Sun-Planet)", 
            "Tangential Ft (Ring-Planet)",
            "Radial Fr (Sun-Planet)",
            "Radial Fr (Ring-Planet)",
            "Normal Fn (Sun-Planet)",
            "Normal Fn (Ring-Planet)",
            "Resultant Planet Pin Load"
        ],
        "Value (N)": [
            f"{loads['Ft_SP']:.2f}",
            f"{loads['Ft_RP']:.2f}",
            f"{loads['Fr_SP']:.2f}",
            f"{loads['Fr_RP']:.2f}",
            f"{loads['Fn_SP']:.2f}",
            f"{loads['Fn_RP']:.2f}",
            f"{stress['F_pin']:.2f}"
        ]
    }
    
    st.dataframe(pd.DataFrame(force_data), hide_index=True, use_container_width=True)
    
    st.subheader("🧮 Gear Stresses")
    
    stress_data = {
        "Check": [
            "Bending σF — Sun/Planet",
            "Bending σF — Ring/Planet",
            "Combined Planet Root σF",
            "Contact σH — Sun/Planet",
            "Contact σH — Ring/Planet"
        ],
        "Actual (MPa)": [
            f"{stress['sigmaF_SP']:.1f}",
            f"{stress['sigmaF_RP']:.1f}",
            f"{stress['sigmaF_planet']:.1f}",
            f"{stress['sigmaH_SP']:.1f}",
            f"{stress['sigmaH_RP']:.1f}"
        ],
        "Allowable (MPa)": [
            f"{mat['sigmaF']:.0f}",
            f"{mat['sigmaF']:.0f}",
            f"{mat['sigmaF']:.0f}",
            f"{mat['sigmaH']:.0f}",
            f"{mat['sigmaH']:.0f}"
        ],
        "Safety Factor": [
            f"{stress['SF_bending_sun']:.2f}",
            f"{stress['SF_bending_planet']:.2f}",
            f"{stress['SF_bending_planet']:.2f}",
            f"{stress['SF_contact_sun']:.2f}",
            f"{stress['SF_contact_ring']:.2f}"
        ],
        "Status": [
            "✅ PASS" if stress['SF_bending_sun'] >= 1 else "❌ FAIL",
            "✅ PASS" if stress['SF_bending_planet'] >= 1 else "❌ FAIL",
            "✅ PASS" if stress['SF_bending_planet'] >= 1 else "❌ FAIL",
            "✅ PASS" if stress['SF_contact_sun'] >= 1 else "❌ FAIL",
            "✅ PASS" if stress['SF_contact_ring'] >= 1 else "❌ FAIL"
        ]
    }
    
    st.dataframe(pd.DataFrame(stress_data), hide_index=True, use_container_width=True)

# ---------------- TAB 6: SHAFTS & PINS ----------------
with tabs[5]:
    st.subheader("🔩 Shaft & Pin Sizing (ASME Code)")
    
    shaft_data = {
        "Component": ["Input Shaft", "Output Shaft", "Planet Pin"],
        "Design Torque/Load": [
            f"{Tin_design:.2f} N·m",
            f"{design_out:.2f} N·m",
            f"{pin['F_design']:.2f} N"
        ],
        "Equivalent Torque (N·m)": [
            f"{Te_in:.2f}",
            f"{Te_out:.2f}",
            "—"
        ],
        "From Bending": [
            "—", "—",
            f"{pin['d_bend']:.2f} mm"
        ],
        "From Shear": [
            "—", "—",
            f"{pin['d_shear']:.2f} mm"
        ],
        "Required Diameter": [
            f"{din:.2f} mm",
            f"{dout:.2f} mm",
            f"{pin['d_pin']:.2f} mm"
        ],
        "Bearing Pressure": [
            "—", "—",
            f"{pin['p_bearing']:.2f} MPa"
        ]
    }
    
    st.dataframe(pd.DataFrame(shaft_data), hide_index=True, use_container_width=True)
    
    st.subheader("Keyway Sizing")
    key_data = {
        "Component": ["Input Shaft", "Output Shaft"],
        "Key Width (mm)": [
            f"{comps['input_shaft']['key_width']:.1f}",
            f"{comps['output_shaft']['key_width']:.1f}"
        ],
        "Key Depth (mm)": [
            f"{comps['input_shaft']['key_depth']:.1f}",
            f"{comps['output_shaft']['key_depth']:.1f}"
        ],
        "Shoulder Diameter (mm)": [
            f"{comps['input_shaft']['shoulder_diameter']:.1f}",
            f"{comps['output_shaft']['shoulder_diameter']:.1f}"
        ]
    }
    st.dataframe(pd.DataFrame(key_data), hide_index=True, use_container_width=True)

# ---------------- TAB 7: COMPONENT DIMENSIONS ----------------
with tabs[6]:
    st.subheader("🧱 Complete Component Dimensions")
    
    # Convert dimensions to DataFrame using the helper function
    dim_df = components_to_dataframe(comps)
    st.dataframe(dim_df, hide_index=True, use_container_width=True)
    
    # Detailed breakdown by component
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sun Gear")
        st.json(comps["sun_gear"], expanded=False)
        
        st.subheader("Planet Gear")
        st.json(comps["planet_gear"], expanded=False)
        
        st.subheader("Ring Gear")
        st.json(comps["ring_gear"], expanded=False)
    
    with col2:
        st.subheader("Carrier")
        st.json(comps["carrier"], expanded=False)
        
        st.subheader("Planet Pin")
        st.json(comps["planet_pin"], expanded=False)
        
        st.subheader("Housing")
        st.json(comps["housing"], expanded=False)

# ---------------- TAB 8: DEFLECTIONS ----------------
with tabs[7]:
    st.subheader("📏 Deflection Analysis")
    
    defl_data = {
        "Component": [
            "Sun Gear Radial Deflection",
            "Ring Gear Radial Deflection",
            "Carrier Deflection",
            "Input Shaft Torsional Angle",
            "Output Shaft Torsional Angle",
            "Tooth Deflection at Contact"
        ],
        "Value": [
            f"{deflections['sun_deflection']:.4f} mm",
            f"{deflections['ring_deflection']:.4f} mm",
            f"{deflections['carrier_deflection']:.4f} mm",
            f"{deflections['shaft_torsional_angle_input']:.2f}°",
            f"{deflections['shaft_torsional_angle_output']:.2f}°",
            f"{deflections['tooth_deflection']:.4f} mm"
        ],
        "Limit": [
            "< 0.10 mm (gear deflection limit)",
            "< 0.10 mm",
            "< 0.15 mm",
            "< 1.0° per metre",
            "< 1.0° per metre",
            "< 0.05 mm"
        ],
        "Status": [
            "✅" if deflections['sun_deflection'] < 0.10 else "⚠️",
            "✅" if deflections['ring_deflection'] < 0.10 else "⚠️",
            "✅" if deflections['carrier_deflection'] < 0.15 else "⚠️",
            "✅" if deflections['shaft_torsional_angle_input'] < 1.0 else "⚠️",
            "✅" if deflections['shaft_torsional_angle_output'] < 1.0 else "⚠️",
            "✅" if deflections['tooth_deflection'] < 0.05 else "⚠️"
        ]
    }
    
    st.dataframe(pd.DataFrame(defl_data), hide_index=True, use_container_width=True)

# ---------------- TAB 9: BEARINGS ----------------
with tabs[8]:
    st.subheader("🛞 Bearing Selection & Life")
    
    # Main bearing
    st.markdown("### Main Shaft Bearing (Input/Output)")
    main_bearing_data = {
        "Parameter": [
            "Radial Load on Bearing",
            "Equivalent Load (× SF)",
            "Selected Bearing",
            "Dynamic Capacity C",
            "L10 Life (million rev)",
            "L10 Life (hours)",
            "Operating Speed"
        ],
        "Value": [
            f"{main_bearing_radial / service_factor:.2f} N",
            f"{main_bearing_radial:.2f} N",
            "SKF 6204 (⌀20×47×14)",
            "12,800 N",
            f"{main_bearing_life['L10_Mrev']:.0f}",
            f"{main_bearing_life['L10_h']:.0f}",
            f"{nin} rpm"
        ]
    }
    st.dataframe(pd.DataFrame(main_bearing_data), hide_index=True, use_container_width=True)
    
    # Planet bearing
    st.markdown("### Planet Bearing")
    planet_bearing_data = {
        "Parameter": [
            "Radial Load on Bearing",
            "Equivalent Load (× SF)",
            "Selected Bearing",
            "Dynamic Capacity C",
            "L10 Life (million rev)",
            "L10 Life (hours)",
            "Operating Speed (relative)"
        ],
        "Value": [
            f"{stress['F_pin']:.2f} N",
            f"{stress['F_pin'] * service_factor:.2f} N",
            "INA NKI12/16 (⌀12×24×16)",
            "6,500 N",
            f"{planet_bearing_life['L10_Mrev']:.0f}",
            f"{planet_bearing_life['L10_h']:.0f}",
            f"{n_planet_rel:.0f} rpm"
        ]
    }
    st.dataframe(pd.DataFrame(planet_bearing_data), hide_index=True, use_container_width=True)

# ---------------- TAB 10: THERMAL ----------------
with tabs[9]:
    st.subheader("🔥 Thermal Analysis")
    
    thermal_data = {
        "Parameter": [
            "Power Loss (Friction)",
            "Housing Surface Area",
            "Heat Transfer Coefficient",
            "Temperature Rise",
            "Steady-State Temperature",
            "Maximum Allowable (mineral oil)",
            "Status"
        ],
        "Value": [
            f"{thermal['power_loss_W']:.2f} W",
            f"{thermal['housing_area_m2']:.4f} m²",
            "15.0 W/m²·K (natural convection)",
            f"{thermal['delta_T']:.1f} °C",
            f"{thermal['steady_state_T']:.1f} °C",
            "90.0 °C",
            "✅ PASS" if thermal['thermal_ok'] else "⚠️ NEEDS COOLING"
        ]
    }
    st.dataframe(pd.DataFrame(thermal_data), hide_index=True, use_container_width=True)
    
    if not thermal['thermal_ok']:
        st.warning("⚠️ Thermal limit exceeded. Consider:")
        st.markdown("""
        - Increase housing size (surface area)
        - Add cooling fins
        - Use synthetic oil with higher temperature rating
        - Reduce continuous power
        """)

# ---------------- TAB 11: DESIGN CHECKS ----------------
with tabs[10]:
    st.subheader("✅ Complete Design Check Dashboard")
    
    check_df = pd.DataFrame(
        [(name, "✅ PASS" if ok else "❌ FAIL") for name, ok in checks],
        columns=["Design Check", "Status"]
    )
    
    st.dataframe(check_df, hide_index=True, use_container_width=True)
    
    # Overall status
    st.divider()
    if overall_ok:
        st.markdown("## 🎉 **DESIGN PASSES ALL CHECKS**")
        st.balloons()
    else:
        st.markdown("## ⚠️ **DESIGN REQUIRES REVIEW**")
        failing = [name for name, ok in checks if not ok]
        st.warning(f"Failed checks: {', '.join(failing)}")

# ---------------- TAB 12: CHARTS & PLOTS ----------------
with tabs[11]:
    st.subheader("📊 Design Visualizations")
    
    # Gear schematic
    st.markdown("### Gearbox Schematic")
    fig = create_gearbox_schematic(geom_sun, geom_planet, geom_ring, 
                                   DEFAULT_PLANETS, pin["d_pin"])
    st.pyplot(fig)
    
    # Stress comparison chart
    st.markdown("### Stress Safety Factors")
    sf_data = pd.DataFrame({
        "Check": ["Bending (Sun)", "Bending (Planet)", 
                   "Contact (Sun)", "Contact (Ring)"],
        "Safety Factor": [
            stress["SF_bending_sun"],
            stress["SF_bending_planet"],
            stress["SF_contact_sun"],
            stress["SF_contact_ring"]
        ]
    })
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    bars = ax2.bar(sf_data["Check"], sf_data["Safety Factor"], 
                   color=['#28a745' if v >= 1 else '#dc3545' for v in sf_data["Safety Factor"]])
    ax2.axhline(1.0, color='red', linestyle='--', label='SF = 1.0 (minimum)')
    ax2.axhline(1.5, color='orange', linestyle='--', label='SF = 1.5 (recommended)')
    ax2.set_ylabel("Safety Factor")
    ax2.set_title("Gear Safety Factors")
    ax2.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig2)
    
    # Force distribution
    st.markdown("### Force Distribution")
    force_labels = ["Ft (Sun-Planet)", "Ft (Ring-Planet)", 
                     "Fr (Sun-Planet)", "Fr (Ring-Planet)"]
    force_values = [loads["Ft_SP"], loads["Ft_RP"], loads["Fr_SP"], loads["Fr_RP"]]
    
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    wedges, texts, autotexts = ax3.pie(
        force_values, labels=force_labels, autopct='%1.1f%%',
        colors=['#D9531E', '#EDB120', '#5C88C5', '#66CC99']
    )
    ax3.set_title("Mesh Force Distribution")
    st.pyplot(fig3)

# ================================================================
# EXPORT SECTION
# ================================================================
st.divider()
st.subheader("📋 Export Design Data")

# Combine all data into a comprehensive export
export_data = {
    "Configuration": {
        "Type": "Single-Stage Planetary",
        "Ratio": ratio_actual,
        "Input Torque (N·m)": tin,
        "Input Speed (rpm)": nin,
        "Motor Power (kW)": motor_power_manual,
        "Design Torque (N·m)": design_out,
        "Module (mm)": module,
        "Face Width (mm)": b,
        "Number of Planets": DEFAULT_PLANETS
    },
    "Teeth": {
        "Sun": zs,
        "Planet": zp,
        "Ring": zr
    },
    "Diameters (mm)": {
        "Sun Pitch": geom_sun["pitch_d"],
        "Planet Pitch": geom_planet["pitch_d"],
        "Ring Pitch": geom_ring["pitch_d"],
        "Sun Tip": geom_sun["tip_d"],
        "Planet Tip": geom_planet["tip_d"],
        "Ring Tip": geom_ring["tip_d"],
        "Sun Root": geom_sun["root_d"],
        "Planet Root": geom_planet["root_d"],
        "Ring Root": geom_ring["root_d"]
    },
    "Stresses (MPa)": {
        "Bending Sun": stress["sigmaF_SP"],
        "Bending Planet": stress["sigmaF_planet"],
        "Contact Sun": stress["sigmaH_SP"],
        "Contact Ring": stress["sigmaH_RP"]
    },
    "Forces (N)": {
        "Tangential SP": loads["Ft_SP"],
        "Tangential RP": loads["Ft_RP"],
        "Radial SP": loads["Fr_SP"],
        "Radial RP": loads["Fr_RP"],
        "Normal SP": loads["Fn_SP"],
        "Normal RP": loads["Fn_RP"],
        "Pin Load": stress["F_pin"]
    },
    "Component Dimensions (mm)": {
        "Input Shaft Dia": din,
        "Output Shaft Dia": dout,
        "Planet Pin Dia": pin["d_pin"],
        "Ring OD": comps["ring_gear"]["outer_d"],
        "Carrier OD": comps["carrier"]["plate_od"],
        "Housing OD": comps["housing"]["outer_d"]
    }
}

# Convert to DataFrame for export
export_rows = []
for category, items in export_data.items():
    for param, value in items.items():
        export_rows.append({
            "Category": category,
            "Parameter": param,
            "Value": value
        })

export_df = pd.DataFrame(export_rows)
st.download_button(
    "⬇️ Download Complete Design Data (CSV)",
    export_df.to_csv(index=False),
    "planetary_gearbox_design.csv",
    "text/csv",
    key="download_full"
)

# Summary text export
summary_text = f"""PLANETARY GEARBOX DESIGN SUMMARY
================================
Configuration: Single-Stage Planetary (Ring Fixed)
Ratio: 1:{ratio_actual:.3f} (Target 1:{ratio_target:.1f})

OPERATING CONDITIONS
- Input Torque: {tin:.2f} N·m
- Input Speed: {nin:.0f} rpm
- Motor Power: {motor_power_manual:.3f} kW
- Output Speed: {nin/ratio_actual:.1f} rpm
- Design Output Torque: {design_out:.1f} N·m

GEAR TOOTHING
- Sun: {zs} teeth
- Planet: {zp} teeth × {DEFAULT_PLANETS}
- Ring: {zr} teeth
- Module: {module:.2f} mm
- Face Width: {b:.1f} mm

DIAMETERS (mm)
- Sun Pitch/Base: {geom_sun['pitch_d']:.2f}/{geom_sun['base_d']:.2f}
- Planet Pitch/Base: {geom_planet['pitch_d']:.2f}/{geom_planet['base_d']:.2f}
- Ring Pitch/Base: {geom_ring['pitch_d']:.2f}/{geom_ring['base_d']:.2f}

FORCES (N)
- Tangential SP: {loads['Ft_SP']:.2f}
- Tangential RP: {loads['Ft_RP']:.2f}
- Normal SP: {loads['Fn_SP']:.2f}
- Normal RP: {loads['Fn_RP']:.2f}
- Planet Pin Load: {stress['F_pin']:.2f}

STRESSES (MPa)
- Bending SP: {stress['sigmaF_SP']:.1f} (allow {mat['sigmaF']:.0f})
- Bending RP: {stress['sigmaF_RP']:.1f}
- Combined Planet: {stress['sigmaF_planet']:.1f}
- Contact SP: {stress['sigmaH_SP']:.1f} (allow {mat['sigmaH']:.0f})
- Contact RP: {stress['sigmaH_RP']:.1f}

COMPONENT SIZES
- Input Shaft: ⌀{din:.2f} mm
- Output Shaft: ⌀{dout:.2f} mm
- Planet Pin: ⌀{pin['d_pin']:.2f} mm
- Ring Gear OD: {comps['ring_gear']['outer_d']:.1f} mm
- Carrier OD: {comps['carrier']['plate_od']:.1f} mm
- Housing OD: {comps['housing']['outer_d']:.1f} mm

STATUS: {'PASS ✅' if overall_ok else 'REVIEW REQUIRED ⚠️'}
"""

st.download_button(
    "⬇️ Download Design Summary (TXT)",
    summary_text,
    "planetary_gearbox_summary.txt",
    "text/plain",
    key="download_summary"
)

# Footer
st.markdown("---")
st.caption("""
**⚠️ DISCLAIMER:** This is a preliminary design/screening tool using simplified 
ISO-6336-lite, ASME shaft code, and Lundberg-Palmgren bearing life equations. 
For production/certified design, verify all results against full standards:
- ISO 6336 (Gear Rating)
- AGMA 2001-B88
- ISO 281 (Bearing Life)
- ASME B106.1M (Shaft Design)
Final manufacturing drawings should include: tolerances, surface finish, 
heat treatment specifications, and quality control requirements.
""")
