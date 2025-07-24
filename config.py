##############################################
#           CONFIGURATION GLOBALE            #
##############################################

t_simu = 300                     # Durée totale de la simulation (en secondes)
precision = 0.5                  # Pas de simulation maximal (s)

##############################################
#         PARAMÈTRES INITIAUX DE TIR         #
##############################################

elev = 45.0                      # Angle d'élévation du tir (°)
azim = 90.0                      # Azimut du tir (°) (0° = Est, 90° = Nord)
latitude_deg = 44.83             # Latitude initiale du tir (°)
v0 = 0.0                         # Vitesse initiale (m/s) — utile si pas de poussée

##############################################
#             PARAMÈTRES MISSILE             #
##############################################

m = 50                           # Masse constante [kg] (si use_thrust = False)
CD = 0.35                        # Coefficient de traînée (CD) — utilisé si enable_dynamic_CD = False
A = 0.29                         # Surface frontale (m²)
radius = 0.29                    # Rayon du projectile (m)
spin_rate = 0                    # Taux de rotation (tours/min) — pour effet Magnus
C_M = 0                          # Coefficient de Magnus (typiquement entre 0.2 et 0.6)

##############################################
#         COMBUSTION / PROPULSION            #
##############################################

use_thrust = True                # Active la propulsion par poussée
initial_mass = 1321              # Masse au départ (kg)
final_mass = 610                 # Masse à la fin de la poussée (kg)
burn_time = 28                   # Durée de combustion (s)
thrust = 100000.0                # Poussée en Newtons (N)

##############################################
#            MODÈLES PHYSIQUES               #
##############################################

enable_drag = True               # Active la traînée aérodynamique
enable_dynamic_CD = True         # Fait varier le CD selon le Mach
enable_magnus = True             # Active l'effet Magnus (si spin_rate > 0)
enable_coriolis = True           # Active l'effet Coriolis
enable_variable_gravity = True   # Gravité dépendante de l'altitude
enable_latitude_variation = True # Latitude actualisée dynamiquement avec le vol

##############################################
#            DONNÉES MÉTÉO — VENT            #
##############################################

use_wind_profile = True          # Utiliser un profil de vent réel (via CSV)
csv_path = "wind_data.csv"       # Chemin vers le fichier CSV du radiosondage

# -- Vent constant alternatif (si use_wind_profile = False) --
wind_speed = 14.0                # Vitesse du vent (m/s)
wind_azim_deg = 90.0             # Azimut du vent (°) — 0° = Est, 90° = Nord
wind_elev_deg = 0.0              # Élévation du vent (°)

##############################################
#           AFFICHAGE DES RESULTATS          #
##############################################

show_3d_trajectory = True           # Affiche la trajectoire 3D colorée par Mach
show_2d_graphs = True               # Affiche les courbes (Altitude, Vitesse, Mach, etc.)
show_realtime_animation = True      # Affiche l'animation 3D en temps réel

