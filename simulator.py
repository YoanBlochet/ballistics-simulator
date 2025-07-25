import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from scipy.interpolate import interp1d
import csv
from utils import *
from config import *

if use_thrust_profile:
    thrust_fn, mass_fn, burn_time = load_thrust_mass_profile(csv_path_thrust, final_mass)

if use_aero_data :
    cd_fn, cl_fn = load_aero_coeffs(csv_path_aero)

if use_wind_profile:
    wind_east_fn, wind_north_fn, wind_vertical_fn = load_wind_profile(csv_path)
else:
    azim_rad = np.deg2rad(wind_azim_deg)
    elev_rad = np.deg2rad(wind_elev_deg)

    wind_constant = wind_speed * np.array([
        np.cos(elev_rad) * np.cos(azim_rad),  # x (Est)
        np.cos(elev_rad) * np.sin(azim_rad),  # y (Nord)
        np.sin(elev_rad)                      # z (Vertical)
    ])

# Direction initiale unitaire issue de l'élévation et l'azimut
dir_init = np.array([
    np.cos(np.deg2rad(elev)) * np.cos(np.deg2rad(azim)),  # x (Est)
    np.cos(np.deg2rad(elev)) * np.sin(np.deg2rad(azim)),  # y (Nord)
    np.sin(np.deg2rad(elev))                              # z (Vertical)
])
dir_init /= np.linalg.norm(dir_init)

############################# FONCTIONS #############################

def projectile_dynamics(t, state):
    global burn_time, thrust_fn, mass_fn, cd_fn, cl_fn
    x, y, z, vx, vy, vz = state
    v_proj = np.array([vx, vy, vz])  # vitesse sol

    # Masse variable et Poussée
    if use_thrust:
        if use_thrust_profile:
            m_current = mass_fn(t)
            thrust_current = thrust_fn(t)
        else:
            burn_time = thrust_duration
            m_current = get_mass(t, burn_time, initial_mass, final_mass)
            thrust_current = thrust
    else:
        m_current = initial_mass
        thrust_current = 0

    if use_thrust and t <= burn_time:
        direction = dir_init
        a_thrust = thrust_current * dir_init / m_current
    else:
        direction = v_proj / np.linalg.norm(v_proj)
        a_thrust = np.zeros(3)

    # Vent
    if use_wind_profile:
        wind = np.array([
            wind_east_fn(z),
            wind_north_fn(z),
            wind_vertical_fn(z)
        ])
    else:
        wind = wind_constant

    # Vitesse relative
    v_rel = v_proj - wind
    v = np.linalg.norm(v_rel)
    rho = atmospheric_density(z)
    alpha_deg = angle_of_attack(v_proj, direction)

    # Calcul dynamique du Mach
    mach = v / speed_of_sound(z)

    # Gravité
    g_local = gravity(z) if enable_variable_gravity else g0
    a_gravity = np.array([0, 0, -g_local])

    # Traînée avec CD variable
    if enable_drag and v > 0:
        if use_aero_data:
            CD_current = cd_fn(mach, alpha_deg)
        elif enable_dynamic_CD:
            CD_current = total_cd(nose_shape, mach, alpha_deg)
        else:
            CD_current = CD
        a_drag = 0.5 * rho * CD_current * A_front * v**2 / m_current
        a_drag_vec = -a_drag * v_rel / v
    else:
        a_drag_vec = np.zeros(3)

    # Portance
    if enable_lift and v > 0:
        alpha_rad = np.deg2rad(alpha_deg)

        if use_aero_data:
            CL_current = cl_fn(mach, alpha_deg)
        else:
            CL_current = CL

        # Appliquer portance seulement si angle et CL significatifs
        if abs(alpha_rad) > 1e-4 and abs(CL_current) > 1e-6:
            lift_axis = np.cross(v_rel, direction)
            norm_axis = np.linalg.norm(lift_axis)

            if norm_axis > 1e-6:
                lift_axis /= norm_axis
                lift_dir = np.cross(lift_axis, v_rel)
                norm_lift = np.linalg.norm(lift_dir)

                if norm_lift > 1e-6:
                    lift_dir /= norm_lift
                    a_lift = 0.5 * rho * CL_current * A_port * v**2 / m_current
                    a_lift_vec = np.clip(a_lift, 0, 100 * g0) * lift_dir
                else:
                    a_lift_vec = np.zeros(3)
            else:
                a_lift_vec = np.zeros(3)
        else:
            a_lift_vec = np.zeros(3)
    else:
        a_lift_vec = np.zeros(3)

    # Magnus
    v_proj_mag = np.linalg.norm(v_proj)

    if enable_magnus and v_proj_mag > 0:
        a_magnus = magnus_acceleration(v_proj, v_rel, spin_rate, radius, C_M, rho, m_current)
    else:
        a_magnus = np.zeros(3)

    # Coriolis
    latitude = latitude_deg + (y / R_Terre) * (180 / np.pi) if enable_latitude_variation else latitude_deg

    if enable_coriolis:
        a_coriolis = coriolis_acceleration(v_proj, latitude)
    else:
        a_coriolis = np.zeros(3)

    # Accélération totale
    a_total = a_drag_vec + a_lift_vec + a_gravity + a_coriolis + a_magnus + a_thrust

    #print(f"t={t:.2f}, m_current={m_current:.3f}, thrust_current={thrust_current:.3f}")
    
    return [vx, vy, vz, *a_total]


# --- événement pour détecter l'impact (z=0) --- #
def hit_ground(t, state):
    if t < 0.25:  # Ne détecte pas l'impact avant 1/4 seconde (pour laisser le temps au booster de délivrer sa poussée)
        return 1
    return state[2]  # z
hit_ground.terminal = True
hit_ground.direction = -1


############################## SOLVEUR ##############################

# Conversion des angles
e = np.deg2rad(elev)
a = np.deg2rad(azim)
vx0 = v0 * np.cos(e) * np.cos(a)
vy0 = v0 * np.cos(e) * np.sin(a)
vz0 = v0 * np.sin(e)

state0 = [0, 0, 0, vx0, vy0, vz0]
t_span = (0, t_simu)

# Application du profil sélectionné
if ACTIVE_PROFILE != 'DEFAULT' and ACTIVE_PROFILE in SOLVER_PROFILES:
    profile = SOLVER_PROFILES[ACTIVE_PROFILE]
    solver_config.update(profile)
    print(f"Profil solveur actif: {ACTIVE_PROFILE}")

def robust_solver():
    """
    Solveur robuste configuré via config.py
    """
    
    burn_duration = burn_time if use_thrust_profile else thrust_duration
    
    def phase_sensitive_solve():
        # Détection automatique du type de simulation
        is_complex_sim = (enable_lift and abs(CL) > solver_config['lift_threshold']) or \
                        (enable_magnus and spin_rate > 0) or \
                        (use_thrust and burn_duration > 0)
        
        if is_complex_sim:
            if enable_solver_debug:
                print(f"Simulation complexe détectée (portance={enable_lift}, Magnus={enable_magnus}, poussée={use_thrust})")
            
            # Phase 1: Poussée (si applicable)
            if use_thrust and burn_duration > 0:
                thrust_config = solver_config['thrust_phase']
                
                if enable_solver_debug:
                    print(f"Phase poussée: durée={burn_duration:.3f}s, méthode={thrust_config['method']}")
                
                sol1 = solve_ivp(
                    projectile_dynamics,
                    (0, min(burn_duration + thrust_config['burn_extension'], t_simu)),
                    state0,
                    method=thrust_config['method'],
                    max_step=thrust_config['max_step'],
                    rtol=thrust_config['rtol'],
                    atol=thrust_config['atol'],
                    first_step=thrust_config['first_step'],
                    events=hit_ground
                )
                
                # Tentative de récupération si échec
                if not sol1.success and thrust_config['enable_fallback']:
                    if enable_solver_debug:
                        print("Échec phase poussée, basculement vers paramètres de secours...")
                    
                    fallback_config = solver_config['fallback']
                    sol1 = solve_ivp(
                        projectile_dynamics,
                        (0, min(burn_duration + thrust_config['burn_extension'], t_simu)),
                        state0,
                        method=fallback_config['method'],
                        max_step=fallback_config['max_step'],
                        rtol=fallback_config['rtol'],
                        atol=fallback_config['atol'],
                        events=hit_ground
                    )
                
                # Phase 2: Vol libre
                if sol1.success and sol1.t[-1] < t_simu and sol1.y[2, -1] > solver_config['min_altitude']:
                    coast_config = solver_config['coast_phase']
                    
                    if enable_solver_debug:
                        print(f"Phase vol libre: méthode={coast_config['method']}")
                    
                    sol2 = solve_ivp(
                        projectile_dynamics,
                        (sol1.t[-1], t_simu),
                        sol1.y[:, -1],
                        method=coast_config['method'],
                        max_step=coast_config['max_step'],
                        rtol=coast_config['rtol'],
                        atol=coast_config['atol'],
                        events=hit_ground
                    )
                    
                    if sol2.success:
                        return combine_solutions(sol1, sol2)
                
                return sol1
            
            else:
                # Simulation complexe sans poussée
                complex_config = solver_config['complex_no_thrust']
                return solve_ivp(
                    projectile_dynamics, t_span, state0,
                    method=complex_config['method'],
                    max_step=complex_config['max_step'],
                    rtol=complex_config['rtol'],
                    atol=complex_config['atol'],
                    events=hit_ground
                )
        
        else:
            # Simulation simple
            simple_config = solver_config['simple']
            if enable_solver_debug:
                print(f"Simulation simple: méthode={simple_config['method']}")
            
            return solve_ivp(
                projectile_dynamics, t_span, state0,
                method=simple_config['method'],
                max_step=simple_config['max_step'],
                rtol=simple_config['rtol'],
                atol=simple_config['atol'],
                events=hit_ground
            )
    
    # Résolution avec gestion d'erreurs
    try:
        sol = phase_sensitive_solve()
        
        if not sol.success and solver_config['enable_emergency_mode']:
            if enable_solver_debug:
                print("Activation du mode d'urgence...")
            
            emergency_config = solver_config['emergency']
            sol = solve_ivp(
                projectile_dynamics, t_span, state0,
                method=emergency_config['method'],
                max_step=emergency_config['max_step'],
                rtol=emergency_config['rtol'],
                atol=emergency_config['atol'],
                events=hit_ground
            )
        
        if enable_solver_debug:
            status = "réussie" if sol.success else "ÉCHOUÉE"
            print(f"Résolution {status} avec {len(sol.t) if hasattr(sol, 't') else 0} points")
        
        return sol
        
    except Exception as e:
        if enable_solver_debug:
            print(f"Erreur durant résolution: {e}")
        
        if solver_config['enable_emergency_mode']:
            emergency_config = solver_config['emergency']
            return solve_ivp(
                projectile_dynamics, t_span, state0,
                method=emergency_config['method'],
                max_step=emergency_config['max_step'],
                rtol=emergency_config['rtol'],
                atol=emergency_config['atol'],
                events=hit_ground
            )
        else:
            raise e

def combine_solutions(sol1, sol2):
    """Combine deux solutions de solve_ivp"""
    from types import SimpleNamespace
    
    t_combined = np.concatenate([sol1.t, sol2.t[1:]])
    y_combined = np.concatenate([sol1.y, sol2.y[:, 1:]], axis=1)
    
    combined = SimpleNamespace()
    combined.t = t_combined
    combined.y = y_combined
    combined.success = True
    combined.t_events = sol2.t_events if sol2.t_events[0].size > 0 else sol1.t_events
    
    return combined

# Utilisation du solveur configuré
sol = robust_solver()

x, y, z = sol.y[0], sol.y[1], sol.y[2]
vx, vy, vz = sol.y[3], sol.y[4], sol.y[5]

############################# AFFICHAGE ##############################

ground_distance = np.sqrt(x[-1]**2 + y[-1]**2)
flight_time = sol.t_events[0][0] if sol.t_events[0].size > 0 else sol.t[-1]
altitude_max = np.max(z)
impact_angle_rad = np.arctan2(vz[-1], np.sqrt(vx[-1]**2 + vy[-1]**2))
impact_angle_deg = np.rad2deg(impact_angle_rad)

idx_burn_end = np.searchsorted(sol.t, burn_time)
alt_burn_end = z[idx_burn_end]
v_burn_end = np.sqrt(vx[idx_burn_end]**2 + vy[idx_burn_end]**2 + vz[idx_burn_end]**2)
v_total = np.sqrt(vx**2 + vy**2 + vz**2)
v_max = np.max(v_total)
v_impact = np.sqrt(vx[-1]**2 + vy[-1]**2 + vz[-1]**2)

print(f"Temps de vol : {flight_time:.2f} s")
print(f"Altitude maximale : {altitude_max:.2f} m")
print(f"Distance sol parcourue : {ground_distance:.2f} m")
print(f"Angle d'impact : {impact_angle_deg:.2f}°")
print(f"Altitude à la fin de poussée : {alt_burn_end:.2f} m")
print(f"Vitesse à la fin de poussée : {v_burn_end:.2f} m/s")
print(f"Vitesse maximale atteinte : {v_max:.2f} m/s")
print(f"Vitesse à l’impact : {v_impact:.2f} m/s")

############################# GRAPHIQUES #############################

v = np.sqrt(vx**2 + vy**2 + vz**2)
v_horiz = np.sqrt(vx**2 + vy**2)
mach = v / np.array([speed_of_sound(alt) for alt in z])

if export_simulation_data:
    # Export des résultats dans un CSV
    export_csv_path = "exports/simulation_data.csv"
    with open(export_csv_path, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["t (s)", "x (m)", "y (m)", "z (m)", "vx (m/s)", "vy (m/s)", "vz (m/s)", "v (m/s)", "Mach"])
        for i in range(len(sol.t)):
            writer.writerow([
                sol.t[i], x[i], y[i], z[i],
                vx[i], vy[i], vz[i],
                v[i], mach[i]
            ])
    print(f"Données exportées dans : {export_csv_path}")

# Trajectoire 3D colorée par Mach
if show_3d_trajectory or export_trajectory:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=mach, cmap=cm.plasma, s=1)
    fig.colorbar(sc, label='Mach')
    ax.set_xlabel("x (m) - vers Est")
    ax.set_ylabel("y (m) - vers Nord")
    ax.set_zlabel("z (m) - Altitude")
    ax.set_title("Trajectoire 3D colorée par Mach")
    ax.set_zlim(0, np.max(z) * 1.1)

    idx_apogee = np.argmax(z)
    ax.scatter(x[idx_apogee], y[idx_apogee], z[idx_apogee],
            color='blue', s=50, label='Apogée')

    mach_supersonic_indices = np.where(mach >= 1.0)[0]
    if mach_supersonic_indices.size > 0:
        idx_mach1 = mach_supersonic_indices[0]
        ax.scatter(x[idx_mach1], y[idx_mach1], z[idx_mach1],
                color='white', edgecolors='black', marker='o', s=100, label='Mach 1')

    if thrust_duration < sol.t[-1]:
        idx_burn_end = np.searchsorted(sol.t, thrust_duration)
        ax.scatter(x[idx_burn_end], y[idx_burn_end], z[idx_burn_end],
                color='black', marker='x', s=80, label='Fin de poussée')

    idx_vmax = np.argmax(v)
    ax.scatter(x[idx_vmax], y[idx_vmax], z[idx_vmax],
            color='magenta', s=50, label='Vitesse max')

    ax.legend()

    if export_trajectory:
        fig.savefig("exports/trajectoire_3d.png")
        print("Trajectoire 3D sauvegardée : exports/trajectoire_3d.png")

# Graphiques 2D

if show_2d_graphs or export_data_plots:
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    ax1.plot(sol.t, z, color='blue')
    ax1.set_ylabel("Altitude (m)")
    ax1.set_title("Altitude, Vitesse, Vitesse horizontale et Mach en fonction du temps")
    ax1.grid()

    ax2.plot(sol.t, v, color='red')
    ax2.set_ylabel("Vitesse (m/s)")
    ax2.grid()

    ax3.plot(sol.t, v_horiz, color='green')
    ax3.set_ylabel("Vitesse horiz. (m/s)")
    ax3.grid()

    ax4.plot(sol.t, mach, color='purple')
    ax4.set_xlabel("Temps (s)")
    ax4.set_ylabel("Nombre de Mach")
    ax4.grid()

    if export_data_plots:
        fig.savefig("exports/graphes_2d.png")
        print("Graphiques 2D sauvegardés : exports/graphes_2d.png")

# Animation 3D avec paramètres optimisés pour la fluidité

# Paramètres d'animation optimisés
fps = 25  # Réduit de 30 à 25 pour moins de charge
duration = sol.t[-1]
n_frames = int(fps * duration)

# Pré-calcul des données interpolées (optimisation majeure)
print("Pré-calcul des données d'animation...")
time_frames = np.linspace(0, duration, n_frames)
interp_x = interp1d(sol.t, x, kind='linear')
interp_y = interp1d(sol.t, y, kind='linear')
interp_z = interp1d(sol.t, z, kind='linear')
interp_v = interp1d(sol.t, v, kind='linear')
interp_mach = interp1d(sol.t, mach, kind='linear')

# Pré-calcul de toutes les positions
x_frames = interp_x(time_frames)
y_frames = interp_y(time_frames)
z_frames = interp_z(time_frames)
v_frames = interp_v(time_frames)
mach_frames = interp_mach(time_frames)

if show_realtime_animation or export_anim:
    # Configuration optimisée de la figure
    plt.ioff()  # Mode non-interactif pour éviter les rafraîchissements inutiles
    fig_anim = plt.figure(figsize=(10, 8))
    fig_anim.patch.set_facecolor('black')  # Fond noir pour de meilleures performances
    
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    ax_anim.set_facecolor('black')
    
    # Limites avec marges réduites
    x_margin = (np.max(x) - np.min(x)) * 0.05
    y_margin = (np.max(y) - np.min(y)) * 0.05
    z_margin = np.max(z) * 0.05
    
    ax_anim.set_xlim(np.min(x) - x_margin, np.max(x) + x_margin)
    ax_anim.set_ylim(np.min(y) - y_margin, np.max(y) + y_margin)
    ax_anim.set_zlim(0, np.max(z) + z_margin)
    
    ax_anim.set_xlabel("x (m) - Est", color='white')
    ax_anim.set_ylabel("y (m) - Nord", color='white')
    ax_anim.set_zlabel("z (m) - Altitude", color='white')
    ax_anim.set_title("Animation 3D du tir", color='white', fontsize=14)
    
    # Style de grille optimisé
    ax_anim.grid(True, alpha=0.3)
    ax_anim.tick_params(colors='white')
    
    # Trajectoire complète avec moins de points pour optimiser
    step = max(1, len(x) // 1000)  # Réduction du nombre de points affichés
    traj_scatter = ax_anim.scatter(x[::step], y[::step], z[::step], 
                                  c=mach[::step], cmap=cm.plasma, s=3, alpha=0.7)
    
    # Colorbar optimisée
    cbar = fig_anim.colorbar(traj_scatter, ax=ax_anim, label='Mach', shrink=0.8)
    cbar.ax.tick_params(colors='white')
    cbar.set_label('Mach', color='white')
    
    # Points caractéristiques (taille réduite pour optimiser)
    idx_apogee = np.argmax(z)
    ax_anim.scatter(x[idx_apogee], y[idx_apogee], z[idx_apogee],
                    color='cyan', s=40, label='Apogée', edgecolors='white', linewidth=1)
    
    mach_supersonic_indices = np.where(mach >= 1.0)[0]
    if mach_supersonic_indices.size > 0:
        idx_mach1 = mach_supersonic_indices[0]
        ax_anim.scatter(x[idx_mach1], y[idx_mach1], z[idx_mach1],
                        color='lime', s=40, label='Mach 1', edgecolors='white', linewidth=1)
    
    if thrust_duration < sol.t[-1]:
        idx_burn_end = np.searchsorted(sol.t, thrust_duration)
        ax_anim.scatter(x[idx_burn_end], y[idx_burn_end], z[idx_burn_end],
                        color='orange', s=40, label='Fin poussée', edgecolors='white', linewidth=1)
    
    idx_vmax = np.argmax(v)
    ax_anim.scatter(x[idx_vmax], y[idx_vmax], z[idx_vmax],
                    color='magenta', s=40, label='Vitesse max', edgecolors='white', linewidth=1)
    
    # Légende optimisée
    legend = ax_anim.legend(loc='upper right', fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('black')
    legend.get_frame().set_alpha(0.8)
    for text in legend.get_texts():
        text.set_color('white')
    
    # Point missile avec traînée
    trail_length = min(50, n_frames // 20)  # Longueur de la traînée adaptative
    missile_trail_x = []
    missile_trail_y = []
    missile_trail_z = []
    
    # Initialisation du missile et de sa traînée
    missile_point, = ax_anim.plot([], [], [], marker='o', color='red', 
                                 markersize=10, markeredgecolor='white', 
                                 markeredgewidth=2, label='Missile')
    
    trail_line, = ax_anim.plot([], [], [], color='red', alpha=0.6, linewidth=2)
    
    # Texte d'information optimisé
    text_box = ax_anim.text2D(0.02, 0.98, "", transform=ax_anim.transAxes, 
                             fontsize=11, verticalalignment='top', color='white',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='black', 
                                     alpha=0.8, edgecolor='white'))
    
    def update_animation(frame_idx):
        """Fonction d'animation optimisée"""
        if frame_idx >= len(time_frames):
            return missile_point, trail_line, text_box
        
        # Données pré-calculées (pas d'interpolation en temps réel)
        x_t = x_frames[frame_idx]
        y_t = y_frames[frame_idx]
        z_t = z_frames[frame_idx]
        v_t = v_frames[frame_idx]
        mach_t = mach_frames[frame_idx]
        t_real = time_frames[frame_idx]
        
        # Mise à jour du point missile
        missile_point.set_data([x_t], [y_t])
        missile_point.set_3d_properties([z_t])
        
        # Gestion de la traînée
        missile_trail_x.append(x_t)
        missile_trail_y.append(y_t)
        missile_trail_z.append(z_t)
        
        # Limitation de la longueur de la traînée
        if len(missile_trail_x) > trail_length:
            missile_trail_x.pop(0)
            missile_trail_y.pop(0)
            missile_trail_z.pop(0)
        
        # Mise à jour de la traînée
        if len(missile_trail_x) > 1:
            trail_line.set_data(missile_trail_x, missile_trail_y)
            trail_line.set_3d_properties(missile_trail_z)
        
        # Mise à jour du texte (moins fréquente pour optimiser)
        if frame_idx % 3 == 0:  # Mise à jour tous les 3 frames
            # Calcul de l'altitude en km si > 1000m
            alt_display = f"{z_t/1000:.2f} km" if z_t > 1000 else f"{z_t:.0f} m"
            
            # Status du vol
            status = ""
            if t_real < thrust_duration:
                status = ">>> PROPULSION <<<"
            else:
                # Calcul de la vitesse verticale pour déterminer montée/descente
                if frame_idx > 0 and frame_idx < len(z_frames):
                    # Vitesse verticale approximée
                    dz_dt = (z_frames[frame_idx] - z_frames[frame_idx-1]) / (time_frames[1] - time_frames[0])
                    if dz_dt > 1.0:  # Montée si vitesse verticale > 1 m/s
                        status = "^^^ MONTEE ^^^"
                    elif dz_dt < -1.0:  # Descente si vitesse verticale < -1 m/s
                        status = "vvv DESCENTE vvv"
                    else:
                        status = "--- APOGEE ---"  # Proche de l'apogée
                else:
                    status = "^^^ MONTEE ^^^"
            
            text_box.set_text(
                f"{status}\n"
                f"Temps : {t_real:.1f} s\n"
                f"Altitude : {alt_display}\n"
                f"Vitesse : {v_t:.0f} m/s\n"
                f"Mach : {mach_t:.2f}\n"
                f"Frame : {frame_idx}/{n_frames}"
            )
        
        return missile_point, trail_line, text_box
    
    print(f"Démarrage de l'animation ({n_frames} frames à {fps} fps)...")
    
    # Animation avec paramètres optimisés
    ani = animation.FuncAnimation(
        fig_anim, 
        update_animation, 
        frames=n_frames,
        interval=1000/fps,
        blit=True,  # Optimisation majeure : blitting activé
        repeat=True,
        cache_frame_data=False  # Évite la mise en cache excessive
    )
     
    if export_anim:
        ani.save("exports/animation.gif", writer='pillow', fps=fps)
        print("Animation 3D exportée : exports/animation_3d.gif")

print("Animation 3D optimisée terminée !")

plt.tight_layout()
#plt.ion()
plt.show()