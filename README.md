# 🚀 Simulation Balistique Avancée

Ce simulateur Python modélise la trajectoire balistique d’un projectile/missile dans un environnement atmosphérique réaliste.  
Il prend en compte un grand nombre de phénomènes physiques et permet une personnalisation complète via un fichier de configuration.

---

## 🧠 Fonctionnalités principales

- 🌍 Gravité et densité de l'air variables avec l'altitude
- 🌬️ Vent configurable (profil réel CSV ou vent constant)
- 💨 Traînée aérodynamique avec CD dynamique (selon Mach) ou statique
- 🎯 Effet Magnus (si rotation active)
- 🌐 Effet Coriolis selon la latitude
- 🔥 Poussée avec combustion (masse variable) si activée
- 📊 Visualisation 2D et 3D (trajectoire, vitesse, Mach, etc.)
- 🎥 Animation 3D en temps réel interactive

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
| `initial_mass`    | Masse de départ                          |
| `final_mass`      | Masse après combustion                   |
| `burn_time`       | Durée de la poussée                      |
| `thrust`          | Intensité de la poussée (N)              |
| `CD`              | Coefficient de traînée (si fixé)         |
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
| Option                      | Description                                                            |
| --------------------------- | ---------------------------------------------------------------------- |
| `enable_drag`               | Active la traînée aérodynamique en fonction de la vitesse              |
| `enable_magnus`             | Active l’effet Magnus dû à la rotation du projectile                   |
| `enable_coriolis`           | Prend en compte l’effet Coriolis lié à la rotation terrestre           |
| `enable_variable_gravity`   | Modélise une gravité variant avec l'altitude                           |
| `enable_latitude_variation` | Met à jour dynamiquement la latitude (utile avec Coriolis)             |
| `enable_dynamic_CD`         | Rend le coefficient de traînée dépendant du Mach (plutôt que constant) |
| `use_thrust`                | Active la propulsion (sinon le projectile est en chute libre)          |

### 📺 Affichage
| Option                    | Description                                 |
|---------------------------|---------------------------------------------|
| `show_3d_trajectory`      | Affiche la trajectoire 3D colorée par Mach  |
| `show_2d_graphs`          | Affiche les graphes (altitude, vitesse...)  |
| `show_realtime_animation` | Animation 3D temps réel                     |

---

## 🧪 Installation

```bash
git clone https://github.com/YoanBlochet/ballistics-simulator.git
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

