def solve_planetary(target_ratio, planets, max_od, module=2.0):
    # R = 1 + Nr / Ns = 9  => Nr / Ns = 8 => Nr = 8 * Ns
    # Condition 1: Nr = Ns + 2*Np => 8*Ns = Ns + 2*Np => 7*Ns = 2*Np => Np = 3.5 * Ns
    # Condition 2 (Assembly): (Nr + Ns) % planets == 0 => (8*Ns + Ns) % 3 == 0 => 9*Ns % 3 == 0 (Always true for planets=3 since 9 is divisible by 3)
    # Condition 3 (Max OD): OD_ring = m * (Nr + 2) <= max_od
    # Let's search over Ns values
    valid_configs = []
    for Ns in range(12, 50): # Ns >= 12 to avoid undercutting
        Nr = 8 * Ns
        Np = 3.5 * Ns
        if Np != int(Np):
            continue
        Np = int(Np)
       
        # Check if planets physically clear each other:
        # Distance between planet centers: 2 * CD * sin(180/planets)
        # CD = m * (Ns + Np) / 2
        # Planet OD approx m * (Np + 2)
        # Clear if 2 * CD * sin(180/planets) > m * (Np + 2)
        import math
        CD = (Ns + Np) / 2.0 # in units of module
        dist_centers = 2 * CD * math.sin(math.radians(180 / planets))
        planet_od_units = Np + 2
        if dist_centers <= planet_od_units:
            continue
           
        # Max OD constraint dictates module max
        # OD_ring = m * (Nr + 2) <= 200 => m_max = 200 / (Nr + 2)
        m_max = max_od / (Nr + 2)
       
        # Let's find standard modules: e.g., 1, 1.25, 1.5, 2, 2.5, 3, 4 etc.
        valid_configs.append({
            'Ns': Ns,
            'Nr': Nr,
            'Np': Np,
            'm_max': m_max
        })
    return valid_configs

configs = solve_planetary(9, 3, 200)
print(configs[:3])



