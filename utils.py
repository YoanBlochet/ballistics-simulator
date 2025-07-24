import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

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

def dynamic_CD(mach):
    """
    Interpolation linéaire de C_D en fonction du Mach.
    """
    # Table simplifiée du CD en fonction de Mach
    mach_table = [0.0, 0.8, 1.0, 1.2, 2.0, 3.0, 5.0]
    cd_table   = [0.3, 0.3, 0.9, 0.6, 0.5, 0.4, 0.35]

    CD_interp = interp1d(mach_table, cd_table, kind='linear', fill_value="extrapolate")

    return float(CD_interp(mach))

