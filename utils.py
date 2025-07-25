import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, RegularGridInterpolator

###################### CONSTANTES PHYSIQUES #######################

R_Terre = 6371000.0   # rayon moyen Terre en m
g0 = 9.81             # gravité au sol en m/s²
gamma = 1.4           # rapport des capacités calorifiques pour l'air sec
R = 287.05            # constante des gaz parfaits (J/kg·K)
omega = 7.292115e-5   # vitesse de rotation de la Terre (rad/s)

######################## DENSITE DE L'AIR #########################

"""
Modèle atmosphérique standard basé sur les équations empiriques
de la NASA Glenn Research Center (GRC), disponible ici :
https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html

Ce modèle est divisé en 3 zones :
1. Troposphère      (0 - 11 000 m)
2. Stratosphère basse (11 000 - 25 000 m)
3. Stratosphère haute (25 000 m +)

Toutes les fonctions utilisent :
- h : altitude en mètres (m)
- T : température en degrés Celsius (°C)
- p : pression en kiloPascal (kPa)
- ρ : densité en kg/m³
"""

def temperature(h):
    """
    Calcule la température (°C) en fonction de l'altitude (m) selon le modèle NASA GRC.
    """
    if h < 0:
        h = 0

    if h <= 11000:
        return 15.04 - 0.00649 * h
    elif h <= 25000:
        return -56.46
    else:
        return -131.21 + 0.00299 * h

def pressure(h):
    """
    Calcule la pression (kPa) en fonction de l'altitude (m) selon le modèle NASA GRC.
    """
    if h < 0:
        h = 0

    if h <= 11000:
        T = temperature(h)
        return 101.29 * ((T + 273.1) / 288.08) ** 5.256
    elif h <= 25000:
        return 22.65 * np.exp(1.73 - 0.000157 * h)
    else:
        T = temperature(h)
        return 2.488 * ((T + 273.1) / 216.6) ** -11.388

def atmospheric_density(h):
    """
    Calcule la densité de l'air (kg/m³) en fonction de l'altitude (m)
    selon le modèle atmosphérique standard NASA GRC.
    
    Utilise l'équation d'état des gaz parfaits :
        ρ = p / (R * T)
    avec :
        - p en Pa  (donc conversion depuis kPa)
        - T en K   (conversion depuis °C)
        - R spécifique de l'air sec = 0.2869 kJ/kg·K
    """
    h = max(h, 0) # notre simulateur (solve_ivp) pouvant parfois donner des valeurs négatives.
    T = temperature(h)
    p = pressure(h)
    return p / (0.2869 * (T + 273.1))  # pression en kPa, T en K

######################### DONNEES DE VENT ##########################

def load_wind_profile(csv_path):
    """
    Charge un profil vertical de vent à partir d'un fichier CSV contenant les données radiosondées.
    
    Le fichier doit contenir les colonnes suivantes :
    - 'HGHT (m)'        : altitude en mètres
    - 'DRCT (deg)'      : direction du vent en degrés (d'où vient le vent)
    - 'SPED (m/s)'      : vitesse du vent en m/s

    Retourne :
    - wind_east(z)   : vitesse du vent vers l'est en m/s (fonction interpolée)
    - wind_north(z)  : vitesse du vent vers le nord en m/s (fonction interpolée)
    - wind_vert(z)   : vitesse verticale du vent (zéro par défaut, fonction interpolée)
    """
    df = pd.read_csv(csv_path)

    # Conversion en radians
    direction_rad = np.radians(df["DRCT (deg)"])
    speed = df["SPED (m/s)"]
    altitude = df["HGHT (m)"]

    # Calcul des composantes vers l'Est et le Nord (vent soufflant depuis une direction)
    wind_east = -speed * np.sin(direction_rad)    # vers l'Est
    wind_north = -speed * np.cos(direction_rad)   # vers le Nord
    wind_vertical = np.zeros_like(altitude)       # vent vertical nul par défaut

    # Création des fonctions interpolées
    wind_east_fn = interp1d(altitude, wind_east, bounds_error=False, fill_value="extrapolate")
    wind_north_fn = interp1d(altitude, wind_north, bounds_error=False, fill_value="extrapolate")
    wind_vertical_fn = interp1d(altitude, wind_vertical, bounds_error=False, fill_value=0.0)

    return wind_east_fn, wind_north_fn, wind_vertical_fn

###################### SIMULATIONS DIVERSES #######################

def speed_of_sound(h):
    """
    Calcule la vitesse du son (m/s) à l'altitude h en mètres,
    selon la température locale (modèle NASA GRC).
    """

    T = temperature(h) + 273.15  # Convertir en Kelvin
    return np.sqrt(gamma * R * T)

def gravity(h):
    """
    Calcul de la gravité en fonction de l'altitude h (m).
    """

    return g0 * (R_Terre / (R_Terre + h))**2

def coriolis_acceleration(v, latitude_deg):
    """
    Calcule l'accélération de Coriolis sur un vecteur vitesse v (3D),
    à une latitude donnée (en degrés).
    v : np.array([vx, vy, vz])
    """

    phi = np.deg2rad(latitude_deg)

    # Vecteur rotation Terre en coordonnées locales (Est, Nord, Haut)
    # Convention : axe z vertical vers le haut, x vers l'Est, y vers le Nord
    Omega_vec = omega * np.array([0, np.cos(phi), np.sin(phi)])  

    return -2 * np.cross(Omega_vec, v)

def magnus_acceleration(v_proj, v_rel, spin_rate, radius, C_M, rho, mass):
    """
    Calcule l'accélération due à l'effet Magnus sur un projectile.
    
    Paramètres :
    - v_proj : vitesse du projectile (sol) (np.array, 3D)
    - v_rel  : vitesse relative à l'air (np.array, 3D)
    - spin_rate : rotation en tours par minute (rpm)
    - radius : rayon du projectile (m)
    - C_M : coefficient de Magnus (sans unité)
    - rho : densité de l'air locale (kg/m³)
    - mass : masse actuelle du projectile (kg)
    
    Retour :
    - accélération Magnus (np.array, 3D)
    """
    v_proj_mag = np.linalg.norm(v_proj)
    if v_proj_mag == 0:
        return np.zeros(3)
    
    omega_magnitude = 2 * np.pi * spin_rate / 60  # rad/s
    omega = omega_magnitude * v_proj / v_proj_mag  # vecteur rotation aligné au vecteur vitesse
    
    F_magnus = C_M * rho * radius * np.cross(omega, v_rel)
    a_magnus = F_magnus / mass
    return a_magnus

def get_mass(t, burn_time, initial_mass, final_mass):
    """
    Retourne la masse instantanée et la poussée vectorielle (dans l'axe de la vitesse).
    
    - t : temps actuel
    - burn_time : durée de la combustion
    - initial_mass : masse au départ
    - final_mass : masse après combustion
    - thrust : intensité de la poussée (N)
    - use_thrust : booléen d’activation
    """
    if t > burn_time:
        return final_mass

    m = initial_mass - (initial_mass - final_mass) * (t / burn_time)
    return m  # Poussée brute (le vecteur sera orienté dans simulator.py)
    
def base_cd_mach_shape(mach, forme="ogive_tangent"):
    """
    Retourne un coefficient de traînée (Cd) en fonction du Mach et de la forme de l'objet.
    Les valeurs sont approximées à partir de données aérodynamiques classiques.
    """

    profils = {
        "ogive_tangent":   (0.10, 0.30, 0.18),
        "ogive_secant":    (0.12, 0.35, 0.20),
        "conical":         (0.15, 0.40, 0.25),
        "hemispherical":   (0.30, 0.60, 0.40),
        "blunt":           (0.50, 0.90, 0.60),
    }

    cd_sub, cd_pic, cd_sup = profils.get(forme, (0.2, 0.4, 0.3))

    if mach < 0.8:
        return cd_sub
    elif mach < 1.2:
        # Transition linéaire vers le pic
        return cd_sub + (cd_pic - cd_sub) * ((mach - 0.8) / 0.4)
    elif mach < 3.0:
        # Descente linéaire vers le Cd supersonique
        return cd_pic + (cd_sup - cd_pic) * ((mach - 1.2) / (3.0 - 1.2))
    else:
        return cd_sup

def drag_correction_angle(alpha_deg):
    k = 0.01  # ajustable selon le projectile
    return k * (alpha_deg ** 2)

def total_cd(shape, mach, alpha_deg):
    cd0 = base_cd_mach_shape(mach, shape)
    cd_alpha = drag_correction_angle(alpha_deg)
    return cd0 + cd_alpha

def angle_of_attack(v_proj, orientation):
    """
    Calcule l'angle d'incidence (alpha) entre la vitesse et l'orientation du missile
    """

    if np.linalg.norm(v_proj) == 0 or np.linalg.norm(orientation) == 0:
        return 0.0
    cos_angle = np.dot(v_proj, orientation) / (np.linalg.norm(v_proj) * np.linalg.norm(orientation))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    alpha_rad = np.arccos(cos_angle)
    return np.degrees(alpha_rad)

def load_thrust_mass_profile(csv_path, empty_mass):
    """
    Charge un fichier CSV contenant les colonnes : t, f, m
    Retourne : (thrust_fn, mass_fn, burn_time)
    """
    df = pd.read_csv(csv_path)
    burn_time = df['t'].max()

    # Poussée : 0 en dehors du burn time
    thrust_fn = interp1d(df['t'], df['f'], bounds_error=False, fill_value=0.0)

    # Masse totale = masse vide + carburant restant
    mass_total = empty_mass + df['m']
    mass_fn = interp1d(df['t'], mass_total, bounds_error=False, fill_value=(mass_total.iloc[0], empty_mass))

    return thrust_fn, mass_fn, burn_time

def load_aero_coeffs(csv_path):
    """
    Charge un fichier CSV contenant les colonnes : mach, alpha, cd et cl
    Retourne : (cd_fn, cl_fn)
    """

    df = pd.read_csv(csv_path)

    if not {'mach', 'alpha', 'cd', 'cl'}.issubset(df.columns):
        raise ValueError("Le fichier CSV doit contenir les colonnes : mach, alpha, cd, cl")

    # Tri unique des valeurs
    mach_vals = np.sort(df['mach'].unique())
    alpha_vals = np.sort(df['alpha'].unique())

    # Création des grilles de CD et CL
    cd_grid = np.full((len(mach_vals), len(alpha_vals)), np.nan)
    cl_grid = np.full_like(cd_grid, np.nan)

    for i, mach in enumerate(mach_vals):
        for j, alpha in enumerate(alpha_vals):
            subset = df[(df['mach'] == mach) & (df['alpha'] == alpha)]
            if not subset.empty:
                cd_grid[i, j] = subset['cd'].values[0]
                cl_grid[i, j] = subset['cl'].values[0]

    # Interpolateurs 2D (mach, alpha) → cd / cl
    cd_interp = RegularGridInterpolator((mach_vals, alpha_vals), cd_grid, bounds_error=False, fill_value=None)
    cl_interp = RegularGridInterpolator((mach_vals, alpha_vals), cl_grid, bounds_error=False, fill_value=None)

    def cd_fn(mach, alpha):
        return cd_interp([[mach, alpha]])[0]

    def cl_fn(mach, alpha):
        return cl_interp([[mach, alpha]])[0]

    return cd_fn, cl_fn
