"""
================================================================================
 PLANETARY GEARBOX — COMPLETE DESIGN & STRESS CALCULATOR WITH CAD-QUALITY 3D
================================================================================
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================================================================
# 1. FIXED PROJECT CONSTANTS
# ================================================================
TARGET_RATIO   = 9.0
MAX_OD_MM      = 200.0
N_PLANETS      = 3
EFFICIENCY     = 0.97
PRESSURE_ANGLE = 20.0
HELIX_ANGLE    = 0.0
MODULE_LIST    = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
DESIGN_OUT_TQ  = 75.0

RATED_T_IN_NM  = 5.0
RATED_N_IN_RPM = 1500.0
MOTOR_POWER_W  = (2.0 * math.pi * RATED_N_IN_RPM / 60.0) * \
                 (DESIGN_OUT_TQ / (TARGET_RATIO * EFFICIENCY))

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

ASME_FACTORS = {
    'Gradually applied / steady load':        (1.5, 1.0),
    'Minor shocks (typical machine drive)':    (1.5, 1.2),
    'Heavy shocks / frequent starts':          (2.0, 1.5),
}

# ================================================================
# 2. TOOTH-COUNT SYNTHESIS
# ================================================================
def find_teeth_combo(fixed_case, target_ratio, n_planets):
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
            else:
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
    alpha_n = math.radians(alpha_n_deg)
    beta = math.radians(beta_deg)
    alpha_t = math.atan(math.tan(alpha_n) / math.cos(beta))

    def ext_gear(z):
        d = z * m / math.cos(beta)
        return {
            'z': z, 'd_pitch': d, 'd_base': d * math.cos(alpha_t),
            'd_tip': d + 2 * m, 'd_root': d - 2.5 * m,
        }

    def int_gear(z):
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
    return {
        'n_sun': n_sun, 'n_ring': n_ring, 'n_carrier': n_carrier,
        'n_planet_spin_rel_carrier': n_planet_spin_rel,
        'output_speed': output_speed,
    }

# ================================================================
# 5. GEAR TOOTH FORCE & STRESS ANALYSIS
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

    Ft_SP = (TS / (np_planets * rbS)) * Kp
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
    sigmaF_planet = math.sqrt(sigmaF_SP**2 + sigmaF_RP**2
                               - 2 * sigmaF_SP * sigmaF_RP * math.cos(theta))

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

    return {
        'Ft_SP': Ft_SP, 'Ft_RP': Ft_RP, 'Fr_SP': Fr_SP, 'Fr_RP': Fr_RP,
        'Fn_SP': Fn_SP, 'Fn_RP': Fn_RP, 'F_pin': F_pin,
        'sigmaF_SP': sigmaF_SP, 'sigmaF_RP': sigmaF_RP,
        'sigmaF_planet': sigmaF_planet, 'sigmaH_SP': sigmaH_SP,
        'sigmaH_RP': sigmaH_RP, 'sigmaH_ring': sigmaH_ring,
        'sigmaF_ring': sigmaF_ring,
    }

# ================================================================
# 6. SHAFT SIZING
# ================================================================
def shaft_diameter_asme(T_nmm, M_nmm, tau_allow_mpa, Kb, Kt, Kw=1.0):
    Te = math.sqrt((Kb * M_nmm)**2 + (Kt * Kw * T_nmm)**2)
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

    return {
        'M_max_Nmm': M_max, 'V_support_N': V_support,
        'd_bend_mm': d_bend, 'd_shear_mm': d_shear, 'd_pin_mm': d_pin,
        'bearing_pressure_mpa': bearing_pressure, 'pressure_ok': pressure_ok,
    }

# ================================================================
# 8. ROLLING BEARING LIFE
# ================================================================
def bearing_L10_life(C_dyn_N, P_equiv_N, n_rpm, bearing_type='Ball'):
    p = 3.0 if bearing_type == 'Ball' else 10.0 / 3.0
    if P_equiv_N <= 0 or n_rpm <= 0:
        return {'L10_Mrev': float('inf'), 'L10_h': float('inf'), 'p': p}
    L10_Mrev = (C_dyn_N / P_equiv_N) ** p
    L10_h = (L10_Mrev * 1.0e6) / (60.0 * n_rpm)
    return {'L10_Mrev': L10_Mrev, 'L10_h': L10_h, 'p': p}

# ================================================================
# 9. ADVANCED 3D CAD-LIKE VISUALIZATION
# ================================================================

def generate_involute_tooth_profile(module, z, phase_angle=0):
    """Generate accurate involute gear tooth profile."""
    # Involute parameters
    alpha_p = math.radians(PRESSURE_ANGLE)  # Pressure angle
    r_base = z * module / 2 * math.cos(alpha_p)  # Base circle radius
    r_pitch = z * module / 2  # Pitch circle radius
    r_tip = r_pitch + module  # Tip radius
    r_root = r_pitch - 1.25 * module  # Root radius
    
    # Generate involute curve points
    n_points = 20  # Points per tooth flank
    theta_max = math.sqrt((r_tip**2 - r_base**2) / r_base**2)  # Max involute angle
    
    # Create involute points
    involute_points = []
    for i in range(n_points):
        theta = theta_max * i / (n_points - 1)
        x_inv = r_base * (math.cos(theta) + theta * math.sin(theta))
        y_inv = r_base * (math.sin(theta) - theta * math.cos(theta))
        involute_points.append((x_inv, y_inv))
    
    # Create full tooth profile
    tooth_pts = []
    half_tooth_angle = math.pi / z  # Angle per half tooth
    
    # Add involute flanks and tip arc
    for i, (x, y) in enumerate(involute_points):
        angle = math.atan2(y, x) - half_tooth_angle
        tooth_pts.append((r_root * math.cos(angle), r_root * math.sin(angle)))
    
    # Add tip arc
    for i in range(5):
        angle = (i / 4) * half_tooth_angle * 2 - half_tooth_angle
        tooth_pts.append((r_tip * math.cos(angle), r_tip * math.sin(angle)))
    
    # Add other flank
    for i in range(n_points - 1, -1, -1):
        x, y = involute_points[i]
        angle = math.atan2(y, x) + half_tooth_angle
        tooth_pts.append((r_root * math.cos(angle), r_root * math.sin(angle)))
    
    # Add root arc
    tooth_pts.append(tooth_pts[0])
    
    return np.array(tooth_pts)


def create_cad_quality_gear(teeth_count, module, face_width, is_internal=False, 
                           color='#708090', name='Gear', center_x=0, center_y=0, 
                           rotation_angle=0):
    """Create CAD-quality gear with accurate tooth profiles."""
    r_pitch = teeth_count * module / 2
    r_tip = r_pitch + module if not is_internal else r_pitch - module
    r_root = r_pitch - 1.25 * module if not is_internal else r_pitch + 1.25 * module
    
    # Generate base tooth profile for one tooth
    alpha_p = math.radians(PRESSURE_ANGLE)
    r_base = r_pitch * math.cos(alpha_p)
    
    # Generate points for full gear
    all_points = []
    pts_per_tooth = 20
    total_points = teeth_count * pts_per_tooth
    
    # Create involute profile
    for tooth_idx in range(teeth_count):
        base_angle = tooth_idx * 2 * math.pi / teeth_count + rotation_angle
        
        # Generate tooth profile points
        tooth_pts = []
        # Flank 1 (involute)
        for i in range(pts_per_tooth // 2):
            theta = math.sqrt((r_tip**2 - r_base**2) / r_base**2) * i / (pts_per_tooth // 2 - 1)
            x_inv = r_base * (math.cos(theta) + theta * math.sin(theta))
            y_inv = r_base * (math.sin(theta) - theta * math.cos(theta))
            angle = math.atan2(y_inv, x_inv) - math.pi / teeth_count / 2
            tooth_pts.append((r_root * math.cos(angle + base_angle), 
                             r_root * math.sin(angle + base_angle)))
        
        # Tip arc
        for i in range(3):
            angle = (i / 2) * math.pi / teeth_count / 2 - math.pi / teeth_count / 2
            tooth_pts.append((r_tip * math.cos(angle + base_angle),
                             r_tip * math.sin(angle + base_angle)))
        
        # Flank 2 (involute)
        for i in range(pts_per_tooth // 2 - 1, -1, -1):
            theta = math.sqrt((r_tip**2 - r_base**2) / r_base**2) * i / (pts_per_tooth // 2 - 1)
            x_inv = r_base * (math.cos(theta) + theta * math.sin(theta))
            y_inv = r_base * (math.sin(theta) - theta * math.cos(theta))
            angle = math.atan2(y_inv, x_inv) + math.pi / teeth_count / 2
            tooth_pts.append((r_root * math.cos(angle + base_angle),
                             r_root * math.sin(angle + base_angle)))
        
        all_points.extend(tooth_pts)
    
    # Close the profile
    all_points.append(all_points[0])
    
    # Convert to numpy array
    profile = np.array(all_points)
    
    # Create 3D extrusion
    n_pts = len(profile)
    z_top = np.full(n_pts, face_width / 2)
    z_bottom = np.full(n_pts, -face_width / 2)
    
    # Create vertices
    x = profile[:, 0] + center_x
    y = profile[:, 1] + center_y
    
    # Create 3D mesh
    vertices = np.column_stack([
        np.concatenate([x, x]),
        np.concatenate([y, y]),
        np.concatenate([z_top, z_bottom])
    ])
    
    # Create faces
    i_faces = []
    j_faces = []
    k_faces = []
    
    # Side faces
    for idx in range(n_pts - 1):
        next_idx = idx + 1
        i_faces.extend([idx, idx, next_idx])
        j_faces.extend([next_idx, idx + n_pts, next_idx + n_pts])
        k_faces.extend([idx + n_pts, next_idx + n_pts, idx + n_pts])
    
    # Create mesh
    gear_mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i_faces,
        j=j_faces,
        k=k_faces,
        color=color,
        name=name,
        opacity=0.85,
        flatshading=False,
        lighting=dict(
            ambient=0.4,
            diffuse=0.7,
            specular=0.3,
            roughness=0.4,
            fresnel=0.2
        ),
        hovertext=f"Z={teeth_count}, m={module}mm, d={2*r_pitch:.1f}mm",
        showscale=False,
        # Add edge lines for CAD look
        contour=dict(show=True, color='rgba(0,0,0,0.2)', width=0.5)
    )
    
    return gear_mesh


def create_cad_quality_sun_gear(S, m, face_width, d_sun):
    """Create CAD-quality sun gear."""
    r_pitch = d_sun / 2
    r_tip = r_pitch + m
    r_root = r_pitch - 1.25 * m
    
    # Generate gear mesh
    gear_mesh = create_cad_quality_gear(S, m, face_width, False, '#D9531E', 'Sun Gear')
    
    # Add central bore with keyway
    bore_radius = m * 3
    key_width = m * 2
    key_height = m * 1.5
    
    # Create bore with keyway
    theta = np.linspace(0, 2 * np.pi, 60)
    z = np.linspace(-face_width/2, face_width/2, 20)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    x_bore = bore_radius * np.cos(theta_grid)
    y_bore = bore_radius * np.sin(theta_grid)
    
    # Create keyway
    key_theta = np.linspace(-key_width/(2*bore_radius), key_width/(2*bore_radius), 20)
    key_z = np.linspace(-face_width/2, face_width/2, 10)
    key_theta_grid, key_z_grid = np.meshgrid(key_theta, key_z)
    x_key = bore_radius * np.cos(key_theta_grid)
    y_key = bore_radius * np.sin(key_theta_grid)
    z_key = key_z_grid
    
    bore_mesh = go.Surface(
        x=np.concatenate([x_bore.flatten(), x_key.flatten()]),
        y=np.concatenate([y_bore.flatten(), y_key.flatten()]),
        z=np.concatenate([z_grid.flatten(), z_key.flatten()]),
        colorscale=[[0, '#3B3B3B'], [1, '#5B5B5B']],
        showscale=False,
        opacity=0.7,
        name='Bore with Keyway'
    )
    
    return [gear_mesh, bore_mesh]


def create_cad_quality_planet_gear(P, m, face_width, d_planet):
    """Create CAD-quality planet gear."""
    r_pitch = d_planet / 2
    r_tip = r_pitch + m
    r_root = r_pitch - 1.25 * m
    
    # Generate gear mesh
    gear_mesh = create_cad_quality_gear(P, m, face_width, False, '#EDB120', 'Planet Gear')
    
    # Add pin bore
    pin_radius = m * 1.5
    theta = np.linspace(0, 2 * np.pi, 40)
    z = np.linspace(-face_width/2, face_width/2, 15)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_bore = pin_radius * np.cos(theta_grid)
    y_bore = pin_radius * np.sin(theta_grid)
    
    bore_mesh = go.Surface(
        x=x_bore, y=y_bore, z=z_grid,
        colorscale=[[0, '#555555'], [1, '#777777']],
        showscale=False,
        opacity=0.85,
        name='Pin Bore'
    )
    
    return [gear_mesh, bore_mesh]


def create_cad_quality_ring_gear(R, m, face_width, d_ring):
    """Create CAD-quality internal ring gear."""
    r_pitch = d_ring / 2
    r_tip = r_pitch - m
    r_root = r_pitch + 1.25 * m
    
    # Generate internal gear mesh
    gear_mesh = create_cad_quality_gear(R, m, face_width, True, '#7f7f7f', 'Ring Gear')
    
    # Add outer housing
    r_outer = r_root + m * 4
    theta = np.linspace(0, 2 * np.pi, 60)
    z = np.linspace(-face_width/2 - m, face_width/2 + m, 10)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_outer = r_outer * np.cos(theta_grid)
    y_outer = r_outer * np.sin(theta_grid)
    
    outer_mesh = go.Surface(
        x=x_outer, y=y_outer, z=z_grid,
        colorscale=[[0, '#6a6a6a'], [1, '#8a8a8a']],
        showscale=False,
        opacity=0.5,
        name='Ring Housing'
    )
    
    return [gear_mesh, outer_mesh]


def create_cad_quality_carrier(n_planets, r_carrier, d_pin, d_output):
    """Create CAD-quality carrier."""
    fig_objs = []
    
    # Carrier disc
    r_disc = r_carrier + d_pin/2 * 3
    thickness = 5
    
    # Create disc with holes
    theta = np.linspace(0, 2 * np.pi, 80)
    z = np.linspace(-thickness/2, thickness/2, 10)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_disc = r_disc * np.cos(theta_grid)
    y_disc = r_disc * np.sin(theta_grid)
    
    disc_mesh = go.Surface(
        x=x_disc, y=y_disc, z=z_grid,
        colorscale=[[0, '#4C72B0'], [1, '#6A92C7']],
        showscale=False,
        opacity=0.75,
        name='Carrier Disc'
    )
    fig_objs.append(disc_mesh)
    
    # Output shaft hub
    theta_hub = np.linspace(0, 2 * np.pi, 50)
    z_hub = np.linspace(-thickness, thickness, 15)
    theta_hub_grid, z_hub_grid = np.meshgrid(theta_hub, z_hub)
    x_hub = (d_output/2 + 3) * np.cos(theta_hub_grid)
    y_hub = (d_output/2 + 3) * np.sin(theta_hub_grid)
    
    hub_mesh = go.Surface(
        x=x_hub, y=y_hub, z=z_hub_grid,
        colorscale=[[0, '#3A5F8A'], [1, '#5A7FA8']],
        showscale=False,
        opacity=0.85,
        name='Output Hub'
    )
    fig_objs.append(hub_mesh)
    
    # Planet pin posts with chamfered edges
    pin_r = d_pin / 2
    for k in range(n_planets):
        ang = k * 2 * np.pi / n_planets
        px = r_carrier * math.cos(ang)
        py = r_carrier * math.sin(ang)
        
        # Create pin post with taper
        theta_pin = np.linspace(0, 2 * np.pi, 30)
        z_pin = np.linspace(-thickness*1.5, thickness*1.5, 20)
        theta_pin_grid, z_pin_grid = np.meshgrid(theta_pin, z_pin)
        
        # Tapered pin
        taper = 1 - 0.05 * np.abs(z_pin_grid) / thickness
        x_pin = px + pin_r * taper * np.cos(theta_pin_grid)
        y_pin = py + pin_r * taper * np.sin(theta_pin_grid)
        
        pin_mesh = go.Surface(
            x=x_pin, y=y_pin, z=z_pin_grid,
            colorscale=[[0, '#555555'], [1, '#888888']],
            showscale=False,
            opacity=0.95,
            name=f'Planet Pin {k+1}'
        )
        fig_objs.append(pin_mesh)
        
        # Add retaining ring groove
        groove_z = thickness * 1.2
        theta_groove = np.linspace(0, 2 * np.pi, 30)
        x_groove = px + (pin_r + 0.5) * np.cos(theta_groove)
        y_groove = py + (pin_r + 0.5) * np.sin(theta_groove)
        z_groove = np.full_like(theta_groove, groove_z)
        
        groove_mesh = go.Scatter3d(
            x=x_groove, y=y_groove, z=z_groove,
            mode='lines',
            line=dict(color='#AA3333', width=2),
            name=f'Pin {k+1} Groove'
        )
        fig_objs.append(groove_mesh)
    
    # Central bore
    theta_bore = np.linspace(0, 2 * np.pi, 40)
    z_bore = np.linspace(-thickness, thickness, 10)
    theta_bore_grid, z_bore_grid = np.meshgrid(theta_bore, z_bore)
    x_bore = (d_output/2) * np.cos(theta_bore_grid)
    y_bore = (d_output/2) * np.sin(theta_bore_grid)
    
    bore_mesh = go.Surface(
        x=x_bore, y=y_bore, z=z_bore_grid,
        colorscale=[[0, 'white'], [1, 'white']],
        showscale=False,
        opacity=0.3,
        name='Central Bore'
    )
    fig_objs.append(bore_mesh)
    
    return fig_objs


def create_cad_quality_shaft(diameter, length, label='Shaft', color='#8C8C8C', 
                            keyway=False, bearing_seats=False):
    """Create CAD-quality shaft with features."""
    # Main shaft body
    theta = np.linspace(0, 2 * np.pi, 40)
    z = np.linspace(0, length, 20)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = (diameter/2) * np.cos(theta_grid)
    y = (diameter/2) * np.sin(theta_grid)
    
    shaft_mesh = go.Surface(
        x=x, y=y, z=z_grid,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        opacity=0.9,
        name=label
    )
    
    fig_objs = [shaft_mesh]
    
    if keyway:
        # Add keyway
        key_width = diameter * 0.25
        key_depth = diameter * 0.1
        key_length = length * 0.3
        
        key_z_start = length * 0.35
        key_z_end = key_z_start + key_length
        
        # Create keyway as box
        x_key = np.linspace(-key_width/2, key_width/2, 10)
        z_key = np.linspace(key_z_start, key_z_end, 10)
        x_key_grid, z_key_grid = np.meshgrid(x_key, z_key)
        y_key = np.full_like(x_key_grid, diameter/2 - key_depth)
        
        # Add keyway visualization
        key_mesh = go.Mesh3d(
            x=x_key_grid.flatten(),
            y=y_key.flatten(),
            z=z_key_grid.flatten(),
            color='#FFFFFF',
            opacity=0.5,
            name='Keyway'
        )
        fig_objs.append(key_mesh)
    
    if bearing_seats:
        # Add bearing seats
        seat_radius = diameter/2 + 1
        seat_length = length * 0.2
        seat_start = length - seat_length
        
        theta_seat = np.linspace(0, 2 * np.pi, 40)
        z_seat = np.linspace(seat_start, length, 10)
        theta_seat_grid, z_seat_grid = np.meshgrid(theta_seat, z_seat)
        x_seat = seat_radius * np.cos(theta_seat_grid)
        y_seat = seat_radius * np.sin(theta_seat_grid)
        
        seat_mesh = go.Surface(
            x=x_seat, y=y_seat, z=z_seat_grid,
            colorscale=[[0, '#B0B0B0'], [1, '#D0D0D0']],
            showscale=False,
            opacity=0.8,
            name='Bearing Seat'
        )
        fig_objs.append(seat_mesh)
    
    return fig_objs


def create_cad_quality_bearing(outer_diameter, inner_diameter, width, label='Bearing'):
    """Create CAD-quality rolling bearing."""
    theta = np.linspace(0, 2 * np.pi, 60)
    z = np.linspace(0, width, 10)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    # Outer ring
    r_outer = outer_diameter / 2
    x_outer = r_outer * np.cos(theta_grid)
    y_outer = r_outer * np.sin(theta_grid)
    
    outer_mesh = go.Surface(
        x=x_outer, y=y_outer, z=z_grid,
        colorscale=[[0, '#C0C0C0'], [1, '#E0E0E0']],
        showscale=False,
        opacity=0.85,
        name=f'{label} Outer Ring'
    )
    
    # Inner ring
    r_inner = inner_diameter / 2
    x_inner = r_inner * np.cos(theta_grid)
    y_inner = r_inner * np.sin(theta_grid)
    
    inner_mesh = go.Surface(
        x=x_inner, y=y_inner, z=z_grid,
        colorscale=[[0, '#A0A0A0'], [1, '#C0C0C0']],
        showscale=False,
        opacity=0.85,
        name=f'{label} Inner Ring'
    )
    
    # Rolling elements
    ball_radius = (outer_diameter - inner_diameter) / 4
    n_balls = 12
    ball_objs = []
    
    for i in range(n_balls):
        ang = i * 2 * np.pi / n_balls
        ball_x = (r_outer + r_inner) / 2 * math.cos(ang)
        ball_y = (r_outer + r_inner) / 2 * math.sin(ang)
        ball_z = width / 2
        
        # Create sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        ball_x_grid = ball_x + ball_radius * np.outer(np.cos(u), np.sin(v))
        ball_y_grid = ball_y + ball_radius * np.outer(np.sin(u), np.sin(v))
        ball_z_grid = ball_z + ball_radius * np.outer(np.ones_like(u), np.cos(v))
        
        ball_mesh = go.Surface(
            x=ball_x_grid, y=ball_y_grid, z=ball_z_grid,
            colorscale=[[0, '#E8E8E8'], [1, '#FFFFFF']],
            showscale=False,
            opacity=0.9,
            name=f'Ball {i+1}'
        )
        ball_objs.append(ball_mesh)
    
    return [outer_mesh, inner_mesh] + ball_objs


def create_advanced_assembly_view(S, P, R, m, n_planets, geom, pin_dia, d_shaft_in, d_shaft_out):
    """Create advanced CAD-quality assembly view."""
    face_width = 16 * m
    d_sun = geom['sun']['d_pitch']
    d_planet = geom['planet']['d_pitch']
    d_ring = geom['ring']['d_pitch']
    r_carrier = (S + P) * m / 2
    
    fig = go.Figure()
    
    # Add sun gear
    sun_objs = create_cad_quality_sun_gear(S, m, face_width, d_sun)
    for obj in sun_objs:
        fig.add_trace(obj)
    
    # Add planet gears
    for k in range(n_planets):
        ang = k * 2 * np.pi / n_planets
        px = r_carrier * math.cos(ang)
        py = r_carrier * math.sin(ang)
        
        planet_objs = create_cad_quality_planet_gear(P, m, face_width, d_planet)
        for obj in planet_objs:
            # Offset to planet position
            if hasattr(obj, 'x') and hasattr(obj, 'y'):
                obj.x = np.array(obj.x) + px
                obj.y = np.array(obj.y) + py
            fig.add_trace(obj)
        
        # Add planet pins
        pin_objs = create_cad_quality_shaft(pin_dia, face_width + 10, f'Planet Pin {k+1}', 
                                           '#3B3B3B', keyway=False)
        for obj in pin_objs:
            if hasattr(obj, 'x') and hasattr(obj, 'y'):
                obj.x = np.array(obj.x) + px
                obj.y = np.array(obj.y) + py
                obj.z = np.array(obj.z) - (face_width + 10) / 2
            fig.add_trace(obj)
    
    # Add ring gear
    ring_objs = create_cad_quality_ring_gear(R, m, face_width, d_ring)
    for obj in ring_objs:
        fig.add_trace(obj)
    
    # Add carrier
    carrier_objs = create_cad_quality_carrier(n_planets, r_carrier, pin_dia, d_shaft_out)
    for obj in carrier_objs:
        fig.add_trace(obj)
    
    # Add input shaft
    input_shaft_objs = create_cad_quality_shaft(d_shaft_in, face_width + 30, 'Input Shaft', 
                                               '#8C8C8C', keyway=True, bearing_seats=True)
    for obj in input_shaft_objs:
        obj.z = np.array(obj.z) - (face_width + 30) / 2
        fig.add_trace(obj)
    
    # Add output shaft
    output_shaft_objs = create_cad_quality_shaft(d_shaft_out, face_width + 40, 'Output Shaft', 
                                                '#8C8C8C', keyway=True, bearing_seats=True)
    for obj in output_shaft_objs:
        obj.z = np.array(obj.z) - (face_width + 40) / 2
        fig.add_trace(obj)
    
    # Add bearings
    # Input shaft bearing
    bearing_objs = create_cad_quality_bearing(d_shaft_in * 2.5, d_shaft_in, 10, 'Input Bearing')
    for obj in bearing_objs:
        obj.z = np.array(obj.z) - face_width / 2 - 5
        fig.add_trace(obj)
    
    # Output shaft bearing
    bearing_objs = create_cad_quality_bearing(d_shaft_out * 2.5, d_shaft_out, 10, 'Output Bearing')
    for obj in bearing_objs:
        obj.z = np.array(obj.z) + face_width / 2 + 5
        fig.add_trace(obj)
    
    # Update layout
    fig.update_layout(
        title=f'Complete Planetary Gearbox Assembly (S:{S} | P:{P} | R:{R} | m:{m:.2f}mm)',
        scene=dict(
            xaxis=dict(title='X (mm)', showbackground=True, backgroundcolor='#F0F0F0'),
            yaxis=dict(title='Y (mm)', showbackground=True, backgroundcolor='#F0F0F0'),
            zaxis=dict(title='Z (mm)', showbackground=True, backgroundcolor='#F0F0F0'),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            traceorder='normal',
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    return fig


# ================================================================
# 10. STREAMLIT INTERFACE
# ================================================================
st.set_page_config(page_title="Planetary Gearbox Calculator", layout="wide")
st.title("⚙️ Planetary Gearbox — Professional Design & CAD Visualization")
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
    mat_data = dict(MATERIAL_PROPS[selected_mat])
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

# ---- Module selection ----
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

# ---- Shaft sizing ----
T_in_design_Nmm = (DESIGN_OUT_TQ / (ratio_actual * EFFICIENCY)) * 1000.0
T_out_design_Nmm = DESIGN_OUT_TQ * 1000.0

d_shaft_in, Te_in = shaft_diameter_asme(T_in_design_Nmm, 0.0, tau, Kb, Kt, kw)
d_shaft_out, Te_out = shaft_diameter_asme(T_out_design_Nmm, 0.0, tau, Kb, Kt, kw)

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
# Main visualization area
st.header("📊 Design Summary")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Achieved Ratio", f"{ratio_actual:.3f}")
with col2:
    st.metric("Teeth (S/P/R)", f"{S}/{P}/{R}")
with col3:
    st.metric("Module", f"{m_use:.2f} mm")
with col4:
    st.metric("Output Speed", f"{output_speed:.1f} rpm")

st.markdown("---")

# 3D CAD Visualization Section
st.header("🧊 3D CAD Visualization")

# Create tabs for different views
view_tabs = st.tabs(["🏗️ Full Assembly", "⚙️ Individual Components"])

with view_tabs[0]:
    st.subheader("Complete Planetary Gearbox Assembly")
    st.caption("Interactive 3D CAD model - rotate, zoom, and explore every detail")
    
    # Create advanced assembly view
    assembly_fig = create_advanced_assembly_view(
        S, P, R, m_use, N_PLANETS, geom, pin_res['d_pin_mm'], d_shaft_in, d_shaft_out
    )
    st.plotly_chart(assembly_fig, width='stretch')
    
    # Add controls info
    st.info("🖱️ **Controls:** Left-click + drag to rotate | Scroll to zoom | Right-click + drag to pan")

with view_tabs[1]:
    st.subheader("Individual Component Views")
    
    # Sun Gear
    st.write("### ☀️ Sun Gear")
    fig_sun = go.Figure()
    sun_objs = create_cad_quality_sun_gear(S, m_use, face_width, d_sun)
    for obj in sun_objs:
        fig_sun.add_trace(obj)
    fig_sun.update_layout(
        title=f'Sun Gear - Z={S}, m={m_use:.2f}mm, OD={geom["sun"]["d_tip"]:.1f}mm',
        scene=dict(
            xaxis=dict(title='X (mm)'),
            yaxis=dict(title='Y (mm)'),
            zaxis=dict(title='Z (mm)'),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
        ),
        height=500,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig_sun, width='stretch')
    
    # Planet Gear
    st.write("### 🔶 Planet Gear")
    fig_planet = go.Figure()
    planet_objs = create_cad_quality_planet_gear(P, m_use, face_width, d_planet if 'd_planet' in dir() else geom['planet']['d_pitch'])
    for obj in planet_objs:
        fig_planet.add_trace(obj)
    fig_planet.update_layout(
        title=f'Planet Gear - Z={P}, m={m_use:.2f}mm, OD={geom["planet"]["d_tip"]:.1f}mm',
        scene=dict(
            xaxis=dict(title='X (mm)'),
            yaxis=dict(title='Y (mm)'),
            zaxis=dict(title='Z (mm)'),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
        ),
        height=500,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig_planet, width='stretch')
    
    # Ring Gear
    st.write("### 🔵 Ring Gear")
    fig_ring = go.Figure()
    ring_objs = create_cad_quality_ring_gear(R, m_use, face_width, d_ring)
    for obj in ring_objs:
        fig_ring.add_trace(obj)
    fig_ring.update_layout(
        title=f'Ring Gear (Internal) - Z={R}, m={m_use:.2f}mm, ID={geom["ring"]["d_tip"]:.1f}mm',
        scene=dict(
            xaxis=dict(title='X (mm)'),
            yaxis=dict(title='Y (mm)'),
            zaxis=dict(title='Z (mm)'),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
        ),
        height=500,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig_ring, width='stretch')
    
    # Carrier
    st.write("### 🔧 Carrier")
    r_carrier = (S + P) * m_use / 2
    fig_carrier = go.Figure()
    carrier_objs = create_cad_quality_carrier(N_PLANETS, r_carrier, pin_res['d_pin_mm'], d_shaft_out)
    for obj in carrier_objs:
        fig_carrier.add_trace(obj)
    fig_carrier.update_layout(
        title=f'Carrier - {N_PLANETS} Planet Positions',
        scene=dict(
            xaxis=dict(title='X (mm)'),
            yaxis=dict(title='Y (mm)'),
            zaxis=dict(title='Z (mm)'),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
        ),
        height=500,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig_carrier, width='stretch')
    
    # Shafts
    st.write("### 🛠️ Shafts")
    fig_shafts = go.Figure()
    
    # Input shaft
    input_shaft_objs = create_cad_quality_shaft(d_shaft_in, 80, 'Input Shaft', '#8C8C8C', keyway=True, bearing_seats=True)
    for obj in input_shaft_objs:
        if hasattr(obj, 'z') and isinstance(obj.z, (list, np.ndarray)):
            obj.z = np.array(obj.z) - 40
        fig_shafts.add_trace(obj)
    
    # Output shaft
    output_shaft_objs = create_cad_quality_shaft(d_shaft_out, 100, 'Output Shaft', '#8C8C8C', keyway=True, bearing_seats=True)
    for obj in output_shaft_objs:
        if hasattr(obj, 'z') and isinstance(obj.z, (list, np.ndarray)):
            obj.z = np.array(obj.z) - 50
        fig_shafts.add_trace(obj)
    
    fig_shafts.update_layout(
        title='Input & Output Shafts with Keyways and Bearing Seats',
        scene=dict(
            xaxis=dict(title='X (mm)'),
            yaxis=dict(title='Y (mm)'),
            zaxis=dict(title='Z (mm)'),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
        ),
        height=500,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig_shafts, width='stretch')
    
    # Planet Pin
    st.write("### 📌 Planet Pin")
    fig_pin = go.Figure()
    pin_objs = create_cad_quality_shaft(pin_res['d_pin_mm'], pin_span_mm, 'Planet Pin', '#3B3B3B')
    for obj in pin_objs:
        fig_pin.add_trace(obj)
    fig_pin.update_layout(
        title=f'Planet Pin - ⌀{pin_res["d_pin_mm"]:.1f}mm × {pin_span_mm:.0f}mm',
        scene=dict(
            xaxis=dict(title='X (mm)'),
            yaxis=dict(title='Y (mm)'),
            zaxis=dict(title='Z (mm)'),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
        ),
        height=500,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig_pin, width='stretch')

# ================================================================
# STRESS ANALYSIS SECTION
# ================================================================
st.markdown("---")
st.header("📊 Stress Analysis Results")

# Create columns for stress results
col_stress1, col_stress2 = st.columns(2)

with col_stress1:
    st.subheader("Mesh Forces")
    force_df = pd.DataFrame({
        'Force Type': ['Tangential (Sun-Planet)', 'Tangential (Ring-Planet)',
                       'Radial (Sun-Planet)', 'Radial (Ring-Planet)',
                       'Normal (Sun-Planet)', 'Normal (Ring-Planet)',
                       'Resultant Pin Load'],
        'Value (N)': [f"{stress_res['Ft_SP']:.1f}", f"{stress_res['Ft_RP']:.1f}",
                      f"{stress_res['Fr_SP']:.1f}", f"{stress_res['Fr_RP']:.1f}",
                      f"{stress_res['Fn_SP']:.1f}", f"{stress_res['Fn_RP']:.1f}",
                      f"{stress_res['F_pin']:.1f}"]
    })
    st.dataframe(force_df, width='stretch', hide_index=True)

with col_stress2:
    st.subheader("Safety Factors")
    safety_df = pd.DataFrame([
        {'Check': 'Bending - Sun/Planet', 'Safety Factor': f"{sfF_SP:.2f}", 
         'Status': '✅ PASS' if sfF_SP >= 1 else '❌ FAIL'},
        {'Check': 'Bending - Ring/Planet', 'Safety Factor': f"{sfF_RP:.2f}",
         'Status': '✅ PASS' if sfF_RP >= 1 else '❌ FAIL'},
        {'Check': 'Contact - Sun/Planet', 'Safety Factor': f"{sfH_SP:.2f}",
         'Status': '✅ PASS' if sfH_SP >= 1 else '❌ FAIL'},
        {'Check': 'Contact - Ring/Planet', 'Safety Factor': f"{sfH_RP:.2f}",
         'Status': '✅ PASS' if sfH_RP >= 1 else '❌ FAIL'},
        {'Check': 'Combined Planet Bending', 'Safety Factor': f"{mat_data['sigmaF_lim']/stress_res['sigmaF_planet']:.2f}",
         'Status': '✅ PASS' if mat_data['sigmaF_lim']/stress_res['sigmaF_planet'] >= 1 else '❌ FAIL'},
    ])
    st.dataframe(safety_df, width='stretch', hide_index=True)

# ================================================================
# DESIGN CHECK DASHBOARD
# ================================================================
st.markdown("---")
st.header("📋 Design Check Dashboard")

# Overall status
overall_ok = (od_fits and assembly_ok and clearance_ok and bearing_pass
              and pin_res['pressure_ok'] and sfF_SP >= 1 and sfF_RP >= 1
              and sfH_SP >= 1 and sfH_RP >= 1)

# Create status metrics
status_cols = st.columns(5)
with status_cols[0]:
    st.metric("Outer Diameter", "✅" if od_fits else "❌", 
              f"{est_od:.0f}mm ≤ {MAX_OD_MM:.0f}mm")
with status_cols[1]:
    st.metric("Assembly", "✅" if assembly_ok else "❌", 
              f"(S+R)%{N_PLANETS}=0")
with status_cols[2]:
    st.metric("Clearance", "✅" if clearance_ok else "❌", 
              "Planet spacing OK")
with status_cols[3]:
    st.metric("Bearing Load", "✅" if bearing_pass else "❌", 
              f"{f_design:.0f}N ≤ {cdyn:.0f}N")
with status_cols[4]:
    st.metric("Overall Design", "✅ PASS" if overall_ok else "⚠️ CHECK", 
              "All criteria verified")

# ================================================================
# DOWNLOAD SUMMARY
# ================================================================
st.markdown("---")
with st.expander("📋 Export Design Summary"):
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
    
    # Download button
    st.download_button(
        label="📥 Download Design Summary",
        data=summary_text,
        file_name="planetary_gearbox_design.txt",
        mime="text/plain"
    )

st.markdown("---")
st.caption("⚙️ Professional Planetary Gearbox Calculator with CAD-Quality 3D Visualization | "
           "Simplified sizing tool (ISO 6336-lite / ASME shaft code / Lundberg-Palmgren bearing life)")
