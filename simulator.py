import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from scipy.interpolate import interp1d
from utils import atmospheric_density, load_wind_profile, speed_of_sound, coriolis_acceleration, gravity, magnus_acceleration, get_mass, dynamic_CD
from config import *

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
    x, y, z, vx, vy, vz = state
    v_proj = np.array([vx, vy, vz])  # vitesse sol

    # Masse variable et Poussée
    if use_thrust:
        m_current = get_mass(t, burn_time, initial_mass, final_mass)
    else:
        m_current = m

    if use_thrust and t <= burn_time:

        direction = dir_init

        a_thrust = thrust * direction / m_current
    else:
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

    # Calcul dynamique du Mach
    mach = v / speed_of_sound(z)

    # Gravité
    g_local = gravity(z) if enable_variable_gravity else 9.81
    a_gravity = np.array([0, 0, -g_local])

    # Traînée avec CD variable
    if enable_drag and v > 0:
        CD_current = dynamic_CD(mach) if enable_dynamic_CD else CD
        a_drag = 0.5 * rho * CD_current * A * v**2 / m_current
        a_drag_vec = -a_drag * v_rel / v
    else:
        a_drag_vec = np.zeros(3)

    # Magnus
    v_proj_mag = np.linalg.norm(v_proj)

    if enable_magnus and v_proj_mag > 0:
        a_magnus = magnus_acceleration(v_proj, v_rel, spin_rate, radius, C_M, rho, m_current)
    else:
        a_magnus = np.zeros(3)

    # Coriolis
    latitude = latitude_deg + (y / 6371000.0) * (180 / np.pi) if enable_latitude_variation else latitude_deg

    if enable_coriolis:
        a_coriolis = coriolis_acceleration(v_proj, latitude)
    else:
        a_coriolis = np.zeros(3)

    # Accélération totale
    a_total = a_drag_vec + a_gravity + a_coriolis + a_magnus + a_thrust
    
    return [vx, vy, vz, *a_total]


# --- événement pour détecter l'impact (z=0) --- #
def hit_ground(t, state):
    return state[2]
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

sol = solve_ivp(projectile_dynamics, t_span, state0, events=hit_ground, max_step=precision, rtol=1e-6, atol=1e-9)

x, y, z = sol.y[0], sol.y[1], sol.y[2]
vx, vy, vz = sol.y[3], sol.y[4], sol.y[5]

############################# AFFICHAGE ##############################

ground_distance = np.sqrt(x[-1]**2 + y[-1]**2)
flight_time = sol.t_events[0][0] if sol.t_events[0].size > 0 else sol.t[-1]
altitude_max = np.max(z)
impact_angle_rad = np.arctan2(vz[-1], np.sqrt(vx[-1]**2 + vy[-1]**2))
impact_angle_deg = np.rad2deg(impact_angle_rad)

print(f"Temps de vol : {flight_time:.2f} s")
print(f"Altitude maximale : {altitude_max:.2f} m")
print(f"Distance sol parcourue : {ground_distance:.2f} m")
print(f"Angle d'impact : {impact_angle_deg:.2f}°")

############################# GRAPHIQUES #############################

v = np.sqrt(vx**2 + vy**2 + vz**2)
v_horiz = np.sqrt(vx**2 + vy**2)
mach = v / np.array([speed_of_sound(alt) for alt in z])

# Trajectoire 3D colorée par Mach
if show_3d_trajectory:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=mach, cmap=cm.plasma, s=1)
    fig.colorbar(sc, label='Mach')
    ax.set_xlabel("x (m) - vers Est")
    ax.set_ylabel("y (m) - vers Nord")
    ax.set_zlabel("z (m) - Altitude")
    ax.set_title("Trajectoire 3D colorée par Mach")
    ax.set_zlim(0, np.max(z) * 1.1)

    if burn_time < sol.t[-1]:
        idx_burn_end = np.searchsorted(sol.t, burn_time)
        ax.scatter(x[idx_burn_end], y[idx_burn_end], z[idx_burn_end],
                   color='black', marker='x', s=80, label='Fin de poussée')

    mach_supersonic_indices = np.where(mach >= 1.0)[0]
    if mach_supersonic_indices.size > 0:
        idx_mach1 = mach_supersonic_indices[0]
        ax.scatter(x[idx_mach1], y[idx_mach1], z[idx_mach1],
                   color='white', edgecolors='black', marker='o', s=100, label='Mach 1')

    ax.legend()

# Graphiques 2D

if show_2d_graphs:
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


# Animation 3D avec paramètres en temps réel

# Paramètres d'animation
fps = 30
duration = sol.t[-1]  # durée du vol
n_frames = int(fps * duration)

# Interpolateurs pour fluidité
interp_x = interp1d(sol.t, x, kind='linear')
interp_y = interp1d(sol.t, y, kind='linear')
interp_z = interp1d(sol.t, z, kind='linear')
interp_v = interp1d(sol.t, v, kind='linear')
interp_mach = interp1d(sol.t, mach, kind='linear')

# Fenêtre 3D
if show_realtime_animation:
    fig_anim = plt.figure(figsize=(8, 6))
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    ax_anim.set_xlim(np.min(x), np.max(x))
    ax_anim.set_ylim(np.min(y), np.max(y))
    ax_anim.set_zlim(0, np.max(z) * 1.1)
    ax_anim.set_xlabel("x (m) - Est")
    ax_anim.set_ylabel("y (m) - Nord")
    ax_anim.set_zlabel("z (m) - Altitude")
    ax_anim.set_title("Animation 3D du tir")

    missile_point, = ax_anim.plot([], [], [], marker='o', color='red', markersize=6, label='Missile')
    text_box = ax_anim.text2D(0.05, 0.95, "", transform=ax_anim.transAxes, fontsize=10, verticalalignment='top')

    def update(frame_idx):
        t_real = frame_idx / fps
        if t_real > duration:
            return missile_point, text_box

        x_t = interp_x(t_real)
        y_t = interp_y(t_real)
        z_t = interp_z(t_real)
        v_t = interp_v(t_real)
        mach_t = interp_mach(t_real)

        missile_point.set_data([x_t], [y_t])
        missile_point.set_3d_properties([z_t])

        text_box.set_text(
            f"Temps : {t_real:.1f} s\n"
            f"Altitude : {z_t:.0f} m\n"
            f"Vitesse : {v_t:.1f} m/s\n"
            f"Mach : {mach_t:.2f}"
        )

        return missile_point, text_box

    ani = animation.FuncAnimation(fig_anim, update, frames=n_frames, interval=1000 / fps, blit=False)

plt.tight_layout()
plt.show()
