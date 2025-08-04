# 🚀 Simulateur Balistique Atmosphérique – Python

Projet personnel développé en tant qu'élève ingénieur en 1ère année à l’[ISAE-ENSMA](https://www.ensma.fr)
Ce simulateur modélise la trajectoire d’un missile/projetile dans un environnement atmosphérique 3D, avec prise en compte fine de nombreux phénomènes physiques. Il se veut particulièrement modulable et personnalisable.

Ce projet vise à démontrer des compétences en modélisation physique, calcul scientifique, architecture logicielle Python et visualisation.
Je précise que n'étant qu'en 1ère année, la prise en compte dynamique de la portance et de la trainée est partiellement simplifiée.

---

## 🧠 Fonctionnalités principales

* 🌍 Gravité variable avec l’altitude
* ☁️ Modélisation avancée du vent (profil réel ou vent constant)
* 💨 Trainée aérodynamique dynamique (selon Mach, angle d’attaque, profil nez)
* ✈️ Portance aérodynamique avec modèle ou profil CSV dynamique
* 🔁 Effet Magnus (si rotation active)
* 🌐 Effet Coriolis + variation dynamique de latitude
* 🔥 Propulsion avec combustion (masse variable ou profils de varariation masse/pousssée CSV)
* ⚙️ Solveur numérique adaptatif avec profils prédéfinis (RK45, DOP853, etc.)
* 🖨️ Export des données (CSV, PNG, GIF)
* 📊 Visualisations 2D/3D dynamiques + animation temps réel

---

## 📂 Organisation du projet

```
ballistic-simulator/
│         
├── simulator.py             # Point d’entrée de la simulation
├── utils.py                 # Fonctions physiques et outils
└── config.py                # Fichier de configuration
│
├── data/                    # Données d'entrée utilisateur
│   ├── wind_data.csv        # Profil atmosphérique  (radiosondage)
│   ├── aero_data.csv        # Profil aérodynamique du projectile (soufflerie)
│   └── thrust_data.csv      # Profil de poussée temporel (essais moteur)
│
├── exports/                 # Dossiers de sortie (.csv, .png, .gif)
│
├── README.md                # Le fichier que vous lisez 😆
├── LICENSE                  # Projet sous licence MIT
└── requirements.txt         # Dépendances Python
```

---

## ⚙️ Configuration détaillée (`config.py`)

### 🎯 Paramètres de tir

| Paramètre      | Description                  |
| -------------- | ---------------------------- |
| `elev`         | Angle d’élévation du tir (°) |
| `azim`         | Azimut du tir (°)            |
| `v0`           | Vitesse initiale (m/s)       |
| `latitude_deg` | Latitude initiale du tir (°) |

---

### 💨 Traînée aérodynamique (CD)

| Paramètre           | Description                              |
| ------------------- | ---------------------------------------- |
| `enable_drag`       | Active la traînée                        |
| `A_front`           | Surface frontale (m²)                    |
| `enable_dynamic_CD` | Active le modèle dynamique CD(Mach, AoA) |
| `CD`                | Valeur fixe si dynamique désactivé       |
| `nose_shape`        | Forme du nez (affecte CD dynamique)      |
| `k_drag_cor`        | Correction liée à l’angle d’attaque      |

---

### ✈️ Portance (CL)

| Paramètre       | Description                          |
| --------------- | ------------------------------------ |
| `enable_lift`   | Active la portance                   |
| `A_port`        | Surface portante (m²)                |
| `CL`            | Coefficient de portance (fixe)       |
| `use_aero_data` | Chargement profil CD/CL dynamiques   |
| `csv_path_aero` | Chemin vers le fichier CSV de profil |

---

### 🚀 Propulsion & Masse variable

| Paramètre            | Description                      |
| -------------------- | -------------------------------- |
| `use_thrust`         | Active la poussée                |
| `initial_mass`       | Masse au lancement               |
| `final_mass`         | Masse finale (post-combustion)   |
| `thrust_duration`    | Durée de combustion              |
| `thrust`             | Poussée (N) si non dynamique     |
| `use_thrust_profile` | Active profil de poussée         |
| `csv_path_thrust`    | Fichier CSV du profil de poussée |

---

### ☁️ Vent atmosphérique

| Paramètre          | Description                    |
| ------------------ | ------------------------------ |
| `use_wind_profile` | Utilise profil de vent (CSV)   |
| `csv_path`         | Fichier de données météo       |
| `wind_speed`       | Vent constant alternatif (m/s) |
| `wind_azim_deg`    | Direction du vent (°)          |
| `wind_elev_deg`    | Angle d’élévation du vent (°)  |

---

### 🧪 Autres modèles physiques

| Paramètre                   | Description                         |
| --------------------------- | ----------------------------------- |
| `enable_variable_gravity`   | Gravité variable selon altitude     |
| `enable_coriolis`           | Effet Coriolis                      |
| `enable_latitude_variation` | Latitude dynamique (avec mouvement) |
| `enable_magnus`             | Effet Magnus                        |
| `spin_rate`                 | Taux de rotation (rpm)              |
| `C_M`                       | Coefficient Magnus                  |
| `radius`                    | Rayon du projectile                 |

---

### 🖼️ Affichage & export

| Paramètre                 | Description            |
| ------------------------- | ---------------------- |
| `show_3d_trajectory`      | Affiche trajectoire 3D |
| `show_2d_graphs`          | Affiche graphes        |
| `show_realtime_animation` | Animation interactive  |
| `export_simulation_data`  | Export CSV brut        |
| `export_trajectory`       | Export CSV trajectoire |
| `export_data_plots`       | Export PNG graphes     |
| `export_anim`             | Export GIF animation   |

---

### 🧮 Configuration du solveur

| Paramètre             | Description                                          |
| --------------------- | ---------------------------------------------------- |
| `ACTIVE_PROFILE`      | Profil de simulation (`DEFAULT`, `FAST_ROUGH`, etc.) |
| `solver_config`       | Paramètres internes dynamiques                       |
| `enable_solver_debug` | Affichage console résolution                         |

---

## ✅ Profils du solveur

| Nom              | Objectif                                  | Méthodes utilisées |
| ---------------- | ----------------------------------------- | ------------------ |
| `DEFAULT`        | Adaptatif (balance perf/précision)        | RK45 + DOP853      |
| `HIGH_PRECISION` | Précision maximale                        | DOP853             |
| `FAST_ROUGH`     | Résolution rapide approximative           | RK23               |
| `ULTRA_STABLE`   | Résolution robuste (situations critiques) | RK23               |
| `FALLBACK`       | Mode de secours (si instabilité)          | RK23               |
| `EMERGENCY`      | Ultime stabilité                          | RK23               |

---

## ▶️ Lancement de la simulation

```bash
python simulator.py
```

Les graphes, animations et données sont affichées / exportées selon `config.py`.

---

## 📊 Exemple de sortie console

```
Simulation complexe détectée (portance=True, Magnus=True, poussée=True)
Phase poussée: durée=2.108s, méthode=RK45
Phase vol libre: méthode=DOP853
Résolution réussie avec 10576 points
Temps de vol : 42.58 s
Altitude maximale : 1998.85 m
Distance sol parcourue : 179.90 m
Angle d'impact : -86.48°
Altitude à la fin de poussée : 929.05 m
Vitesse à la fin de poussée : 564.91 m/s
Vitesse maximale atteinte : 654.78 m/s
Vitesse à l’impact : 84.53 m/s
Pré-calcul des données d'animation...
Démarrage de l'animation (1064 frames à 25 fps)...
Animation 3D optimisée terminée !
```

---

## 🧪 Dépendances (`requirements.txt`)

```txt
numpy
scipy
matplotlib
pandas
csv
```

---

## 🎓 Auteur

Projet personnel réalisé par **Yoan Blochet**,
Élève ingénieur à l’[ISAE-ENSMA](https://www.ensma.fr), promotion 2028.
Conçu dans une optique de **démonstration technique**, de **modélisation physique avancée**, et de **valorisation en recrutement ingénieur / stage**.

---

## 🔒 Licence

Ce projet est sous licence MIT – Voir le fichier [LICENSE](LICENSE).

---
