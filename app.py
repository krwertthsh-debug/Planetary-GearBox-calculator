import math
from dataclasses import dataclass, asdict
import pandas as pd
import streamlit as st

# ================================================================
# PLANETARY GEARBOX DESIGNER v2
# Single-stage planetary, intended configuration: Ring fixed,
# Sun input, Carrier output, target reduction 9:1.
# IMPORTANT: preliminary sizing/learning aid; verify with full ISO/AGMA,
# shaft fatigue, bearing catalogue, housing stiffness, thermal, etc.
# ================================================================

PI = math.pi
DEFAULT_RATIO = 9.0
DEFAULT_MAX_OD = 200.0
DEFAULT_PLANETS = 3
DEFAULT_ETA = 0.97
DEFAULT_ALPHA = 20.0
DEFAULT_BETA = 0.0
MODULES = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]

MATERIALS = {
    "17CrNiMo6 / 18CrNiMo7-6 case carburized": dict(E=210000.0, nu=0.30, tau=240.0, sigmaF=430.0, sigmaH=1500.0, sigma_allow_bend=380.0),
    "20MnCr5 / 16MnCr5 case carburized": dict(E=210000.0, nu=0.30, tau=140.0, sigmaF=380.0, sigmaH=1350.0, sigma_allow_bend=320.0),
    "EN24 / 4340 hardened": dict(E=206000.0, nu=0.30, tau=150.0, sigmaF=310.0, sigmaH=1150.0, sigma_allow_bend=260.0),
    "Custom": dict(E=200000.0, nu=0.30, tau=180.0, sigmaF=300.0, sigmaH=1200.0, sigma_allow_bend=250.0),
}

# ------------------------- Geometry ------------------------------
def exact_ring_teeth(zs: int, zp: int) -> int:
    return zs + 2 * zp


def ratio_ring_fixed(zs: int, zr: int) -> float:
    return (zs + zr) / zs


def assembly_ok(zs: int, zr: int, nplanets: int) -> bool:
    return (zs + zr) % nplanets == 0


def planet_spacing_ok(zs: int, zp: int, nplanets: int, clearance_mm: float = 1.0) -> bool:
    # Conservative pitch-circle based check. A CAD tooth-tip interference check
    # should replace this for production use.
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
                # External housing OD allowance is deliberately explicit.
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

# ------------------------- Loads/stress --------------------------
def kinematics_ring_fixed(nin, ratio, zs, zp):
    ncarrier = nin / ratio
    nplanet_rel = abs(nin - ncarrier) * zs / zp
    # Absolute planet spin from Willis relation for context.
    return dict(n_sun=nin, n_ring=0.0, n_carrier=ncarrier,
                n_planet_rel_carrier=nplanet_rel, n_out=ncarrier)


def mesh_loads(Tin_Nm, zs, zr, nplanets, m, alpha_deg, beta_deg, Kp=1.10):
    a = math.radians(alpha_deg)
    rb_s = (zs*m/(2*math.cos(math.radians(beta_deg)))) * math.cos(a)
    rb_r = (zr*m/(2*math.cos(math.radians(beta_deg)))) * math.cos(a)
    Ft_sp = Tin_Nm*1000.0 / (nplanets * rb_s) * Kp
    # Ring-planet tangential force based on pitch radius ratio.
    Ft_rp = Ft_sp * (rb_s/rb_r)
    Fr_sp = Ft_sp * math.tan(a)
    Fr_rp = Ft_rp * math.tan(a)
    Fn_sp = Ft_sp / math.cos(a)
    Fn_rp = Ft_rp / math.cos(a)
    # For a 3-planet symmetrical stage, the planet-pin reaction is approximately
    # the vector result of the two mesh normals; theta can be adjusted in UI.
    return dict(Ft_SP=Ft_sp, Ft_RP=Ft_rp, Fr_SP=Fr_sp, Fr_RP=Fr_rp,
                Fn_SP=Fn_sp, Fn_RP=Fn_rp, rbS=rb_s, rbR=rb_r)


def gear_stress(loads, m, b, zs, zp, zr, material, alpha_deg=20.0,
                KA=1.25, KV=1.15, KFbeta=1.20, KFalpha=1.00,
                KHbeta=1.25, KHalpha=1.00, Kp=1.10, theta_deg=120.0):
    # "ISO-lite" screening equations consistent with the user's existing model.
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

# ------------------------- Shafts/pins ---------------------------
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
    p = Fd/(d*support_span*0.55)  # effective bush/roller contact width estimate
    return dict(F_design=Fd, Mmax=Mmax, V=V, d_bend=db, d_shear=ds,
                d_pin=d, p_bearing=p, pressure_ok=p <= bearing_p)

# ------------------------- Component sizing ----------------------
def component_dimensions(zs, zp, zr, m, b, d_sun, d_planet, d_ring,
                         d_in_shaft, d_out_shaft, d_pin, max_od=200.0,
                         pin_span=30.0, design_out=75.0):
    # Preliminary manufacturability / structural rules, deliberately exposed.
    carrier_pitch_radius = (d_sun + d_planet)/2.0
    carrier_arm_radius = carrier_pitch_radius
    carrier_plate_thk = max(6.0, 0.30*b, 0.20*d_pin)
    carrier_od = min(max_od-4.0, 2*(carrier_pitch_radius + 1.6*d_pin))
    carrier_bore = d_out_shaft + 2.0
    carrier_hub_od = max(d_out_shaft + 8.0, d_out_shaft*1.5)
    carrier_hub_len = max(1.5*d_out_shaft, 1.5*b)

    sun_bore = d_in_shaft + 2.0
    sun_hub_od = max(sun_bore + 6.0, d_in_shaft*1.6)
    sun_hub_len = max(1.2*d_in_shaft, b)

    planet_bore = d_pin + 2.0
    planet_hub_od = max(planet_bore + 4.0, d_pin*1.45)
    planet_face_width = b

    ring_tooth_tip_d = d_ring - 2*m
    ring_tooth_root_d = d_ring + 2.5*m
    ring_radial_rim = max(0.18*d_ring, 8.0)
    ring_outer_d = ring_tooth_root_d + 2*ring_radial_rim
    ring_inner_d = ring_tooth_root_d  # internal root region
    ring_wall = (ring_outer_d - ring_inner_d)/2
    ring_face_width = b + 2.0

    housing_clearance = 2.0
    envelope_od = ring_outer_d + 2*housing_clearance

    return {
        'carrier': {
            'plate_thickness': carrier_plate_thk, 'plate_od': carrier_od,
            'output_bore': carrier_bore, 'hub_od': carrier_hub_od,
            'hub_length': carrier_hub_len, 'planet_pin_circle_d': 2*carrier_pitch_radius,
            'arm_center_radius': carrier_arm_radius,
        },
        'sun': {'gear_face_width': b, 'bore_d': sun_bore, 'hub_od': sun_hub_od, 'hub_length': sun_hub_len},
        'planet': {'gear_face_width': planet_face_width, 'bore_d': planet_bore, 'hub_od': planet_hub_od, 'hub_length': max(b, pin_span*0.75)},
        'ring': {'inner_tooth_tip_d': ring_tooth_tip_d, 'inner_tooth_root_d': ring_tooth_root_d,
                 'outer_d': ring_outer_d, 'wall_thickness': ring_wall, 'face_width': ring_face_width},
        'housing': {'outer_allowance': housing_clearance, 'envelope_od': envelope_od},
    }


def row_section(component, dims):
    rows=[]
    for k,v in dims.items():
        rows.append({'Component':component,'Dimension':k,'Value':round(v,3),'Unit':'mm'})
    return rows

# ------------------------- UI ------------------------------------
st.set_page_config(page_title="Planetary Gearbox Designer v2", layout="wide")
st.title("⚙️ Planetary Gearbox Designer — 1:9 Single Stage")
st.caption("Parametric preliminary design tool. Target: 1:9 reduction • 65–75 N·m design output • ≤200 mm OD")

with st.sidebar:
    st.header("1. Operating inputs")
    tin = st.number_input("Input torque Tᵢ (N·m)", 0.1, 1000.0, 5.0, 0.1)
    nin = st.number_input("Input speed nᵢ (rpm)", 10.0, 20000.0, 1500.0, 50.0)
    motor_power_kw = st.number_input("Motor power (kW)", 0.05, 20.0, 0.785, 0.01)
    eta = st.number_input("Stage efficiency η", 0.80, 0.99, DEFAULT_ETA, 0.005)
    design_out = st.slider("Design output torque (N·m)", 65.0, 75.0, 75.0, 1.0)

    st.header("2. Gear inputs")
    ratio_target = st.number_input("Target ratio", 1.0, 20.0, 9.0, 0.1)
    nplanets = st.selectbox("Number of planets", [3,4,5], index=0)
    alpha = st.number_input("Pressure angle α (deg)", 14.5, 25.0, 20.0, 0.5)
    beta = st.number_input("Helix angle β (deg)", 0.0, 30.0, 0.0, 1.0)
    max_od = st.number_input("Maximum gearbox OD (mm)", 50.0, 500.0, 200.0, 1.0)

    st.subheader("Manual tooth-count / module")
    manual = st.checkbox("Use manual S / P / module", value=True)
    if manual:
        zs = st.number_input("Sun teeth Zs", 17, 120, 20, 1)
        zp = st.number_input("Planet teeth Zp", 17, 150, 70, 1)
        module = st.number_input("Module m (mm)", 0.5, 10.0, 1.0, 0.05)
    else:
        cand = suggest_tooth_sets(ratio_target, nplanets, max_od)
        if cand.empty:
            st.error("No exact candidate found. Increase OD or change planet count/module range.")
            st.stop()
        opt = cand.sort_values(['module','zs','zp']).iloc[0]
        zs, zp, module = int(opt.zs), int(opt.zp), float(opt.module)
        st.info(f"Auto suggestion: Zs={zs}, Zp={zp}, Zr={int(opt.zr)}, m={module:.2f} mm")

    st.header("3. Materials / factors")
    mat_name = st.selectbox("Material", list(MATERIALS.keys()))
    mat = dict(MATERIALS[mat_name])
    if mat_name == "Custom":
        mat['E'] = st.number_input("E (MPa)", 50000.0, 300000.0, 200000.0)
        mat['nu'] = st.number_input("ν", 0.20, 0.40, 0.30, 0.01)
        mat['tau'] = st.number_input("Allowable shaft/pin shear (MPa)", 20.0, 500.0, 180.0)
        mat['sigmaF'] = st.number_input("Allowable tooth bending (MPa)", 50.0, 1000.0, 300.0)
        mat['sigmaH'] = st.number_input("Allowable tooth contact (MPa)", 200.0, 2000.0, 1200.0)
        mat['sigma_allow_bend'] = st.number_input("Allowable pin bending (MPa)", 50.0, 800.0, 250.0)
    Kb = st.number_input("ASME Kb", 1.0, 3.0, 1.5, 0.1)
    Kt = st.number_input("ASME Kt", 1.0, 2.0, 1.2, 0.1)
    Kw = st.number_input("Keyway factor Kw", 1.0, 2.0, 1.3, 0.05)
    service = st.number_input("Bearing service factor", 1.0, 3.0, 1.5, 0.1)
    pin_span = st.number_input("Planet pin support span (mm)", 10.0, 100.0, 30.0, 1.0)
    bearing_p = st.number_input("Allowable pin bearing pressure (MPa)", 5.0, 80.0, 25.0, 1.0)
    theta = st.number_input("Sun/ring mesh force angle θ (deg)", 60.0, 180.0, 120.0, 1.0)

# ------------------------- Core calculation ----------------------
zr = exact_ring_teeth(int(zs), int(zp))
ratio_actual = ratio_ring_fixed(int(zs), int(zr))
b_factor = st.sidebar.number_input("Face width / module ratio b/m", 6.0, 20.0, 12.0, 0.5)
b = b_factor*module

sun = gear_geometry(int(zs), module, alpha, False, beta)
planet = gear_geometry(int(zp), module, alpha, False, beta)
ring = gear_geometry(int(zr), module, alpha, True, beta)

nin_power = 2*PI*nin/60.0
tout_nominal = tin*ratio_actual*eta
output_speed = nin/ratio_actual
Tin_design = design_out/(ratio_actual*eta)
P_from_torque = Tin_design*nin_power/1000.0

loads = mesh_loads(Tin_design, int(zs), int(zr), nplanets, module, alpha, beta)
stress = gear_stress(loads, module, b, int(zs), int(zp), int(zr), mat, alpha, theta_deg=theta)

# Shaft loads: no external bending unless user later adds it.
din, Te_in = shaft_diameter(Tin_design, 0.0, mat['tau'], Kb, Kt, Kw)
dout, Te_out = shaft_diameter(design_out, 0.0, mat['tau'], Kb, Kt, Kw)
pin = pin_design(stress['F_pin'], pin_span, mat['sigma_allow_bend'], mat['tau'], bearing_p)

comp = component_dimensions(int(zs), int(zp), int(zr), module, b,
                             sun['pitch_d'], planet['pitch_d'], ring['pitch_d'],
                             din, dout, pin['d_pin'], max_od, pin_span, design_out)

assembly = assembly_ok(int(zs), int(zr), nplanets)
clearance = planet_spacing_ok(int(zs), int(zp), nplanets)
ratio_ok = abs(ratio_actual-ratio_target) < 1e-9
od_ok = comp['housing']['envelope_od'] <= max_od
sfF = mat['sigmaF'] / max(stress['sigmaF_planet'], 1e-9)
sfH_sp = mat['sigmaH'] / max(stress['sigmaH_SP'], 1e-9)
sfH_rp = mat['sigmaH'] / max(stress['sigmaH_RP'], 1e-9)

checks = [
    ("Ratio", ratio_ok), ("Planet assembly", assembly), ("Planet clearance", clearance),
    ("OD envelope", od_ok), ("Gear bending SF", sfF >= 1.0),
    ("Sun-planet contact SF", sfH_sp >= 1.0), ("Ring-planet contact SF", sfH_rp >= 1.0),
    ("Planet pin pressure", pin['pressure_ok']),
]
overall = all(x[1] for x in checks)

# ------------------------- Dashboard ------------------------------
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Ratio", f"1:{ratio_actual:.3f}")
c2.metric("Output speed", f"{output_speed:.1f} rpm")
c3.metric("Nominal output torque", f"{tout_nominal:.1f} N·m")
c4.metric("Design torque", f"{design_out:.1f} N·m")
c5.metric("Design status", "PASS ✅" if overall else "CHECK ⚠️")

if abs(motor_power_kw*1000 - 2*PI*nin/60*tin) / max(2*PI*nin/60*tin,1.0) > 0.20:
    st.warning("Motor-power input and input torque/speed imply noticeably different power. Review the operating point.")

st.info(f"Selected teeth: Zs={zs}, Zp={zp}, Zr={zr}. Exact ring-fixed reduction equation: i=(Zs+Zr)/Zs, with Zr=Zs+2Zp."
        f"  Input design torque for the 75/65–75 N·m output design basis = {Tin_design:.2f} N·m.")

TABS = st.tabs([
    "📊 Results", "⚙️ Tooth Synthesis", "📐 Gear Geometry", "🔄 Kinematics",
    "🧲 Forces", "🧮 Gear Stress", "🔩 Shafts", "📌 Planet Pin", "🧱 Components",
    "🛞 Bearings / Life", "✅ Checks & Export"
])

with TABS[0]:
    st.subheader("Overall design results")
    summary = pd.DataFrame([
        ["Input torque", tin, "N·m"], ["Input speed", nin, "rpm"],
        ["Motor power input", motor_power_kw, "kW"], ["Stage efficiency", eta, "—"],
        ["Actual ratio", ratio_actual, "—"], ["Output speed", output_speed, "rpm"],
        ["Nominal output torque", tout_nominal, "N·m"], ["Design output torque", design_out, "N·m"],
        ["Input design torque", Tin_design, "N·m"], ["Module", module, "mm"], ["Face width", b, "mm"],
        ["Ring envelope OD", comp['housing']['envelope_od'], "mm"],
        ["Input shaft required diameter", din, "mm"], ["Output shaft required diameter", dout, "mm"],
        ["Planet pin required diameter", pin['d_pin'], "mm"],
        ["Gear bending safety factor", sfF, "—"], ["SP contact safety factor", sfH_sp, "—"],
        ["RP contact safety factor", sfH_rp, "—"],
    ], columns=["Parameter","Value","Unit"])
    st.dataframe(summary, hide_index=True, use_container_width=True)

with TABS[1]:
    st.subheader("User-controlled tooth synthesis")
    tooth = pd.DataFrame([
        ["Sun", zs, "Input"], ["Planet", zp, "Meshing gear"], ["Ring", zr, "Fixed"],
        ["S + 2P = R", f"{zs}+2({zp}) = {zs+2*zp}", "Required geometry"],
        ["Ratio", ratio_actual, "(S+R)/S"],
        ["Assembly", "PASS" if assembly else "FAIL", "(S+R) mod N = 0"],
        ["Planet clearance", "PASS" if clearance else "CHECK", "Pitch-circle screening check"],
        ["Module", module, "mm"],
    ], columns=["Item","Value","Meaning"])
    st.dataframe(tooth, hide_index=True, use_container_width=True)
    cand = suggest_tooth_sets(ratio_target, nplanets, max_od)
    if not cand.empty:
        st.markdown("**Other exact 1:9 candidate tooth sets that fit the current OD search:**")
        st.dataframe(cand.head(25), hide_index=True, use_container_width=True)

with TABS[2]:
    st.subheader("Pitch / base / tip / root geometry")
    geo_df = pd.DataFrame([
        ["Sun",zs,sun['pitch_d'],sun['base_d'],sun['tip_d'],sun['root_d'],b],
        ["Planet",zp,planet['pitch_d'],planet['base_d'],planet['tip_d'],planet['root_d'],b],
        ["Ring (internal)",zr,ring['pitch_d'],ring['base_d'],ring['tip_d'],ring['root_d'],comp['ring']['face_width']],
    ], columns=["Gear","Teeth","Pitch dia (mm)","Base dia (mm)","Tip dia (mm)","Root dia (mm)","Face width (mm)"])
    st.dataframe(geo_df, hide_index=True, use_container_width=True)
    st.write(f"Sun–planet centre distance = {(sun['pitch_d']+planet['pitch_d'])/2:.3f} mm")
    st.write(f"Ring–planet centre distance = {(ring['pitch_d']-planet['pitch_d'])/2:.3f} mm")
    st.write(f"Circular pitch = {PI*module:.3f} mm")

with TABS[3]:
    kin = kinematics_ring_fixed(nin, ratio_actual, int(zs), int(zp))
    st.dataframe(pd.DataFrame([
        ["Sun",kin['n_sun'],"rpm"],["Ring",0.0,"rpm"],["Carrier / output",kin['n_carrier'],"rpm"],
        ["Planet spin relative to carrier",kin['n_planet_rel_carrier'],"rpm"]
    ], columns=["Member","Speed","Unit"]), hide_index=True, use_container_width=True)

with TABS[4]:
    st.dataframe(pd.DataFrame([
        ["Ft SP",loads['Ft_SP'],"N"],["Fr SP",loads['Fr_SP'],"N"],["Fn SP",loads['Fn_SP'],"N"],
        ["Ft RP",loads['Ft_RP'],"N"],["Fr RP",loads['Fr_RP'],"N"],["Fn RP",loads['Fn_RP'],"N"],
        ["Resultant planet-pin load",stress['F_pin'],"N"]
    ], columns=["Force","Value","Unit"]), hide_index=True, use_container_width=True)

with TABS[5]:
    stress_df = pd.DataFrame([
        ["Sun/Planet bending",stress['sigmaF_SP'],mat['sigmaF'],mat['sigmaF']/max(stress['sigmaF_SP'],1e-9)],
        ["Ring/Planet bending",stress['sigmaF_RP'],mat['sigmaF'],mat['sigmaF']/max(stress['sigmaF_RP'],1e-9)],
        ["Combined planet bending",stress['sigmaF_planet'],mat['sigmaF'],sfF],
        ["Sun/Planet contact",stress['sigmaH_SP'],mat['sigmaH'],sfH_sp],
        ["Ring/Planet contact",stress['sigmaH_RP'],mat['sigmaH'],sfH_rp],
    ], columns=["Check","Actual MPa","Allowable MPa","Safety factor"])
    st.dataframe(stress_df, hide_index=True, use_container_width=True)
    st.warning("These are screening equations based on the simplified model from your original app, not a full ISO 6336 rating calculation.")

with TABS[6]:
    st.dataframe(pd.DataFrame([
        ["Input shaft",Tin_design,0.0,Te_in,din],["Output shaft",design_out,0.0,Te_out,dout]
    ], columns=["Shaft","Design torque N·m","Bending moment N·m","Equivalent torque N·m","Required diameter mm"]), hide_index=True, use_container_width=True)
    st.caption("External bending is set to zero in this version. Add an overhung-load model before using the shaft diameter for manufacture.")

with TABS[7]:
    st.dataframe(pd.DataFrame([
        ["Design pin load",pin['F_design'],"N"],["Max bending moment",pin['Mmax'],"N·mm"],
        ["Support shear",pin['V'],"N"],["d from bending",pin['d_bend'],"mm"],
        ["d from shear",pin['d_shear'],"mm"],["Selected pin diameter",pin['d_pin'],"mm"],
        ["Bearing pressure",pin['p_bearing'],"MPa"],["Pressure check","PASS" if pin['pressure_ok'] else "FAIL","—"]
    ], columns=["Parameter","Value","Unit"]), hide_index=True, use_container_width=True)

with TABS[8]:
    rows=[]
    for cname, dims in comp.items():
        if isinstance(dims, dict):
            for k,v in dims.items():
                rows.append([cname,k,v,"mm"])
    st.dataframe(pd.DataFrame(rows, columns=["Component","Dimension","Value","Unit"]), hide_index=True, use_container_width=True)
    st.markdown("### Recommended interpretation")
    st.write("All dimensions here are preliminary sizing rules. They provide a parametric starting point for CAD, not a finished manufacturing drawing.")

with TABS[9]:
    # Basic L10 screening, catalogue capacity is user-specified here so no fake bearing is selected.
    C_main = st.number_input("Main bearing dynamic rating C (N)", 1000.0, 200000.0, 12000.0, 500.0, key="Cmain")
    C_planet = st.number_input("Planet bearing dynamic rating C (N)", 1000.0, 100000.0, 6000.0, 250.0, key="Cplanet")
    Pmain = math.hypot(loads['Fr_SP'], loads['Ft_SP']) * service
    Pplanet = stress['F_pin'] * service
    Lmain = (C_main/Pmain)**3 if Pmain>0 else float('inf')
    Lplanet = (C_planet/Pplanet)**(10/3) if Pplanet>0 else float('inf')
    Lmain_h = Lmain*1e6/(60*nin)
    Lplanet_h = Lplanet*1e6/(60*kinematics_ring_fixed(nin,ratio_actual,int(zs),int(zp))['n_planet_rel_carrier'])
    st.dataframe(pd.DataFrame([
        ["Main bearing",Pmain,C_main,Lmain,Lmain_h,"million rev / h"],
        ["Planet bearing",Pplanet,C_planet,Lplanet,Lplanet_h,"million rev / h"],
    ], columns=["Bearing","Equivalent load N","C N","L10 million rev","L10 hours","Unit"]), hide_index=True, use_container_width=True)
    st.caption("Use actual bearing catalogue ratings, lubrication, temperature, contamination and speed limits for final selection.")

with TABS[10]:
    checkdf = pd.DataFrame([[name,"PASS" if ok else "FAIL/CHECK"] for name,ok in checks], columns=["Design check","Status"])
    st.dataframe(checkdf, hide_index=True, use_container_width=True)
    export_df = summary.copy()
    st.download_button("⬇️ Download results CSV", export_df.to_csv(index=False), "planetary_gearbox_results.csv", "text/csv")
    st.markdown("### Formula map")
    formulae = {
        "Ring teeth": "Zr = Zs + 2 Zp",
        "Ring-fixed ratio": "i = (Zs + Zr)/Zs",
        "Pitch diameter": "d = Z·m/cosβ",
        "Base diameter": "db = d·cos(αt)",
        "External tip/root": "da=d+2m ; df=d−2.5m",
        "Internal ring tip/root": "da=d−2m ; df=d+2.5m",
        "Tangential force": "Ft ≈ T/(Nplanet·rb) × Kp",
        "Normal force": "Fn = Ft/cosα",
        "Radial force": "Fr = Ft·tanα",
        "ASME equivalent torque": "Te=√((KbM)^2+(KtKwT)^2)",
        "Solid shaft diameter": "d=(16Te/(πτallow))^(1/3)",
        "Planet pin bending diameter": "d=(32M/(πσallow))^(1/3)",
        "L10 life": "L10=(C/P)^p; p=3 ball, 10/3 roller",
    }
    st.table(pd.DataFrame(formulae.items(), columns=["Calculation","Formula"]))

st.markdown("---")
st.caption("Source basis: your original Streamlit calculator. It already defines the same major calculation blocks and explicitly describes the model as simplified/representative rather than certified. Final production design should be checked against the full ISO 6336 / AGMA / ISO 281 methods and manufacturer data.")
