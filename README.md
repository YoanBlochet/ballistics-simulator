# 🚀 Simulation Balistique Avancée

Ce simulateur Python modélise la trajectoire balistique d’un projectile/missile dans un environnement atmosphérique réaliste.  
Il prend en compte un grand nombre de phénomènes physiques et permet une personnalisation complète via un fichier de configuration.

---

## 🧠 Fonctionnalités principales

- 🌍 Gravité variable avec l'altitude
- 🌬️ Vent configurable (profil réel CSV ou vent constant)
- 💨 Traînée aérodynamique avec CD dynamique (selon Mach)
- 🎯 Effet Magnus (si rotation active)
- 🌐 Effet Coriolis selon la latitude
- 🔥 Poussée avec combustion (masse variable)
- 📊 Visualisation 2D et 3D (trajectoire, vitesse, Mach, etc.)
- 🎥 Animation temps réel interactive

---

## ⚙️ Configuration complète (`config.py`)

L'utilisateur peut activer/désactiver chaque phénomène physique ou graphique.

### 🎯 Paramètres de tir
| Paramètre        | Description                        |
|------------------|------------------------------------|
| `elev`           | Angle d’élévation (en degrés)      |
| `azim`           | Azimut du tir (0° Est, 90° Nord)   |
| `v0`             | Vitesse initiale (utile sans poussée) |
| `latitude_deg`   | Latitude initiale (pour Coriolis)  |

### 🚀 Missile & Propulsion
| Paramètre         | Description                              |
|-------------------|------------------------------------------|
| `use_thrust`      | Active la poussée avec combustion        |
| `initial_mass`    | Masse de départ                          |
| `final_mass`      | Masse après combustion                   |
| `burn_time`       | Durée de la poussée                      |
| `thrust`          | Intensité de la poussée (N)              |
| `CD`              | Coefficient de traînée (fixe ou variable)|
| `enable_dynamic_CD`| CD dépendant du Mach                    |
| `radius`          | Rayon du projectile (effet Magnus)       |
| `spin_rate`       | Vitesse de rotation (rpm)                |
| `C_M`             | Coefficient de Magnus                    |

### 🌬️ Vent
| Paramètre           | Description                           |
|---------------------|---------------------------------------|
| `use_wind_profile`  | Active le profil de vent CSV          |
| `csv_path`          | Fichier CSV radiosondé                |
| `wind_speed`        | Vitesse du vent constant              |
| `wind_azim_deg`     | Direction du vent (°)                 |
| `wind_elev_deg`     | Angle d’élévation du vent (°)         |

### 🔬 Modèles physiques
- `enable_drag`
- `enable_magnus`
- `enable_coriolis`
- `enable_variable_gravity`
- `enable_latitude_variation`

### 📺 Affichage
| Option                    | Description                                 |
|---------------------------|---------------------------------------------|
| `show_3d_trajectory`      | Affiche la trajectoire 3D colorée par Mach  |
| `show_2d_graphs`          | Affiche les graphes (altitude, vitesse...)  |
| `show_realtime_animation` | Animation 3D temps réel                     |

---

## 🧪 Installation

```bash
git clone https://github.com/YoanBlochet/ballistic-simulator.git
cd ballistic-simulator
pip install -r requirements.txt
```

---

## ▶️ Utilisation

```bash
python simulator.py
```

Visualisations et logs s’affichent à la fin de la simulation.

---

## 📁 Organisation du projet

| Fichier / Dossier | Rôle                                                                 |
|-------------------|----------------------------------------------------------------------|
| `simulator.py`    | Script principal (simulation, affichage)                             |
| `utils.py`        | Fonctions physiques (atmosphère, forces, interpolation...)           |
| `config.py`       | Paramètres globaux de simulation                                     |
| `wind_data.csv`   | Données radiosondées (externe, optionnel)                            |
| `requirements.txt`| Dépendances Python                                                   |

---

## ✅ Exemple de sortie console

```
Temps de vol : 28.75 s
Altitude maximale : 9821.30 m
Distance sol parcourue : 22312.50 m
Angle d'impact : -34.20°
```

---

## 📊 Exemples de visualisations

- ✅ **Trajectoire 3D** : colorée selon Mach, avec repères (fin poussée, Mach 1)
- ✅ **Graphiques 2D** : altitude, vitesse, vitesse horizontale, Mach vs temps
- ✅ **Animation** : mise à jour temps réel de l'état du missile (altitude, vitesse, Mach)

---

## 🔒 Licence

Distribué sous licence MIT. Voir [LICENSE](LICENSE).

---

## 🙋‍♂️ Auteur

Projet Personnel développé par un élève ingénieur à l’[ISAE-ENSMA](https://www.ensma.fr/) en début de 1ère année, dans un objectif de démonstration technique et d’application physique avancée.

