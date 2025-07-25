##############################################
#         PARAMÈTRES INITIAUX DE TIR         #
##############################################

elev = 90.0                             # Angle d'élévation du tir (°)
azim = 90.0                             # Azimut du tir (°) (0° = Est, 90° = Nord)
latitude_deg = 44.83                    # Latitude initiale du tir (°)
v0 = 0.0                                # Vitesse initiale (m/s) — utile si pas de poussée

##############################################
#           TRAINÉE AÉRODYNAMIQUE            #
##############################################

enable_drag = True                      # Activer la traînée aérodynamique
A_front = 0.044                         # Surface frontale (m²)
CD = 0.5                                # Coefficient de traînée (CD) — utilisé si enable_dynamic_CD = False
enable_dynamic_CD = True                # Faire varier CD selon Mach
nose_shape = "ogive_tangent"            # Forme du nez : "conical", "ogive_tangent", "ogive_secant", "blunt", "hemispherical"
k_drag_cor = 0.01                       # Coefficient de correction de trainée : CD = CD_dynamic + CD_drag_corr
                                        # CD_drag_corr = k_drag_cor * (angle_attaque_en_degrés) ** 2

##############################################
#           PORTANCE AÉRODYNAMIQUE           #
##############################################

enable_lift = True                      # Activer la portance aérodynamique
A_port = 0.80                           # Surface portante (m²)
CL = 0.3                                # Coefficient de portance (CL) — utilisé si use_aero_data = False  

use_aero_data = True                    # Utiliser données aérodynamiques dynamiques (PREND LE DESSUS SUR nose_shape, CD et CL)
csv_path_aero = "data/aero_data.csv"

##############################################
#         COMBUSTION / PROPULSION             #
##############################################

use_thrust = True                       # Active la propulsion par poussée
initial_mass = 4.161                    # Masse initiale (kg)
final_mass = 1.840                      # Masse finale (kg) après combustion
thrust_duration = 1.9                   # Durée de combustion (s)
thrust = 2400                           # Poussée constante (N) (si pas de profil)

use_thrust_profile = True               # Utiliser un profil de poussée via CSV
csv_path_thrust = "data/thrust_data.csv"

##############################################
#            MODÈLES PHYSIQUES               #
##############################################

enable_variable_gravity = True          # Gravité variable selon altitude

enable_coriolis = True                  # Activer l'effet Coriolis
enable_latitude_variation = True        # Actualiser la latitude dynamiquement avec le vol

enable_magnus = True                    # Activer l'effet Magnus (si spin_rate > 0)
spin_rate = 0                           # Taux de rotation (tours/min) — pour effet Magnus
C_M = 0                                 # Coefficient de Magnus (typiquement entre 0.2 et 0.6)
radius = 0.0375                         # Rayon du projectile (m)

##############################################
#            DONNÉES MÉTÉO — VENT            #
##############################################

use_wind_profile = True                 # Utiliser un profil de vent réel (via CSV)
csv_path = "data/wind_data.csv"         # Chemin vers fichier CSV radiosondage

# Vent constant alternatif (si use_wind_profile = False)
wind_speed = 14.0       # Vitesse du vent (m/s)
wind_azim_deg = 90.0    # Azimut vent (°) — 0°=Est, 90°=Nord
wind_elev_deg = 0.0     # Élévation du vent (°)

##############################################
#           AFFICHAGE DES RÉSULTATS           #
##############################################

show_3d_trajectory = True               # Affiche trajectoire 3D colorée par Mach
show_2d_graphs = True                   # Affiche courbes (Altitude, Vitesse, Mach, etc.)
show_realtime_animation = True          # Affiche animation 3D temps réel

export_simulation_data = False          # Exporte données simulation
export_trajectory = False               # Exporte trajectoire
export_data_plots = False               # Exporte graphiques
export_anim = False                     # Exporte animation

##############################################
#         CONFIGURATION DU SOLVEUR          #
##############################################

t_simu = 100                            # Durée totale de la simulation (en secondes)

# Profil actif (cf PROFILS SOLVEUR PRÉDÉFINIS ci-dessous)
ACTIVE_PROFILE = 'DEFAULT'              # 'HIGH_PRECISION', 'FAST_ROUGH', 'ULTRA_STABLE', ou 'DEFAULT'

# Debug et monitoring
enable_solver_debug = True              # Affiche les infos de résolution

##############################################
#       PROFILS SOLVEUR PRÉDÉFINIS          #
##############################################

# DEFAULT - Configuration adaptative du solveur numérique
solver_config = {
    
    # Seuils de détection
    'lift_threshold': 1e-6,          # Seuil CL pour détecter portance significative
    'min_altitude': 0.0,             # Altitude minimale pour continuer simulation
    
    # PHASE DE POUSSÉE (critique avec portance)
    'thrust_phase': {
        'method': 'RK45',            # RK45 généralement suffisamment stable
        'max_step': 0.002,           # Pas très fin (2ms) durant la poussée
        'rtol': 1e-3,                # Tolérance relative relâchée
        'atol': 1e-4,                # Tolérance absolue
        'first_step': 1e-5,          # Premier pas minuscule (10µs)
        'burn_extension': 0.1,       # Extension après fin poussée (s)
        'enable_fallback': True      # Active le mode de secours (si échec)
    },
    
    # PHASE VOL LIBRE (moins critique)
    'coast_phase': {
        'method': 'DOP853',          # Solveur haute précision Dormand-Prince
        'max_step': 0.05,            # Pas plus large (50ms)
        'rtol': 1e-4,                # Précision standard
        'atol': 1e-6
    },
    
    # SIMULATION SIMPLE (pas de poussée ni portance)
    'simple': {
        'method': 'DOP853',          # Haute précision pour trajectoires simples
        'max_step': 0.1,             # Pas large (100ms)
        'rtol': 1e-4,
        'atol': 1e-6
    },
    
    # SIMULATION COMPLEXE SANS POUSSÉE
    'complex_no_thrust': {
        'method': 'RK45',            # Compromis stabilité/précision
        'max_step': 0.01,            # Pas moyen (10ms)
        'rtol': 1e-3,
        'atol': 1e-4
    },
    
    # MODE DE SECOURS (si échec phase poussée)
    'fallback': {
        'method': 'RK23',            # Solveur simple et robuste
        'max_step': 0.001,           # Pas très fin (1ms)
        'rtol': 5e-3,                # Tolérance très relâchée
        'atol': 1e-3
    },
    
    # MODE D'URGENCE (dernier recours)
    'enable_emergency_mode': True,
    'emergency': {
        'method': 'RK23',            # Le plus simple
        'max_step': 0.0005,          # Pas ultra-fin (0.5ms)
        'rtol': 1e-2,                # Très peu précis mais stable
        'atol': 1e-2
    }
}

SOLVER_PROFILES = {
    
    'HIGH_PRECISION': {
        'thrust_phase': {
            'method': 'DOP853', 'max_step': 0.001, 'rtol': 1e-5, 'atol': 1e-7,
            'first_step': 1e-6, 'burn_extension': 0.05, 'enable_fallback': True
        },
        'coast_phase': {
            'method': 'DOP853', 'max_step': 0.01, 'rtol': 1e-5, 'atol': 1e-7
        }
    },
    
    'FAST_ROUGH': {
        'thrust_phase': {
            'method': 'RK23', 'max_step': 0.01, 'rtol': 1e-2, 'atol': 1e-3,
            'first_step': 1e-4, 'burn_extension': 0.2, 'enable_fallback': True
        },
        'coast_phase': {
            'method': 'RK45', 'max_step': 0.1, 'rtol': 1e-3, 'atol': 1e-4
        }
    },
    
    'ULTRA_STABLE': {
        'thrust_phase': {
            'method': 'RK23', 'max_step': 0.0005, 'rtol': 1e-3, 'atol': 1e-4,
            'first_step': 1e-6, 'burn_extension': 0.5, 'enable_fallback': True
        },
        'coast_phase': {
            'method': 'RK45', 'max_step': 0.01, 'rtol': 1e-3, 'atol': 1e-4
        }
    }
}