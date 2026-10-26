# Scénarios de simulation

Ce document décrit les réseaux routiers et la demande de trafic utilisés dans toutes les
expériences, ainsi que la procédure qui permet de les régénérer à l'identique.

Tous les fichiers SUMO sont **générés par script** à partir des fichiers de configuration
`configs/scenarios/*.yaml`. Ils sont **versionnés dans le dépôt** pour que toute l'équipe utilise
exactement les mêmes scénarios, quelle que soit sa version de SUMO. Les fichiers de routes sont
compressés (`.rou.xml.gz`, lus directement par SUMO), ce qui ramène l'ensemble d'environ 190 Mo à
environ 12 Mo.

**Version de SUMO utilisée pour la génération : Eclipse SUMO 1.28.0** (également enregistrée
dans chaque `manifest.yaml`). Régénérer avec une autre version peut modifier les fichiers ;
la différence apparaîtra alors dans git.

```bash
networks/build_all.sh                               # construit puis valide tous les scénarios
python -m trafficmarl scenarios build grid3x3       # un seul scénario
python -m trafficmarl scenarios validate all        # rapport : results/scenario_validation/
```

Chaque scénario construit contient :

- le réseau (`<nom>.net.xml`), avec un **plan fixe** par feu (ligne de base « plan fixe ») ;
- `tls_actuated.add.xml` : la variante **actionnée** des plans (ligne de base « actionnée »).
  Charger ce fichier comme fichier additionnel rend ces plans actifs. Les phases vertes ont une
  durée de 5 à 50 s, valeurs par défaut de SUMO ;
- `routes/<profil>_<split>_<i>.rou.xml.gz` : les fichiers de routes ;
- `<nom>.sumocfg` : configuration pour ouvrir le scénario dans `sumo-gui` ;
- `manifest.yaml` : liste de tous ces fichiers, des identifiants des feux et de la version de
  SUMO. Le reste du code lit uniquement ce manifeste.

## 1. Vue d'ensemble

| Scénario | Carrefours à feux | Réseau | Rôle |
|---|---|---|---|
| `single` | 1 | carrefour isolé à 4 branches | vérification que l'apprentissage fonctionne |
| `corridor` | 4 | artère est-ouest, 4 carrefours | coordination en « onde verte » |
| `grid3x3` | 9 | grille 3 × 3 | scénario principal de développement |
| `grid5x5` | 25 | grille 5 × 5 (type Manhattan) | passage à l'échelle |
| `lust` | 13 | centre-ville de Luxembourg (LuST) | scénario urbain réaliste |

## 2. Réseaux synthétiques (`single`, `corridor`, `grid3x3`, `grid5x5`)

Générés avec `netgenerate` (paramètres dans `configs/scenarios/_synthetic_defaults.yaml`) :

- îlots de **200 m** entre carrefours, branches d'accès de 200 m en bordure ;
- **2 voies** par approche + **une voie dédiée au tourne-à-gauche** de 100 m ;
- vitesse limite **50 km/h** ;
- feux **à plan fixe** générés par SUMO, cycle de **90 s**, avec **4 phases vertes**
  (Nord-Sud tout droit/droite, Nord-Sud gauche protégé, Est-Ouest tout droit/droite,
  Est-Ouest gauche protégé) séparées par des phases jaunes de 3 s.

La voie de tourne-à-gauche dédiée a été choisie pour obtenir l'espace d'actions standard à
4 phases utilisé dans la littérature (SUMO-RL, RESCO). Sans elle, SUMO ne génère que 2 phases
avec tourne-à-gauche permissif, ce qui laisse très peu de marge de décision aux agents.

Ce plan fixe généré par SUMO constitue la ligne de base « programme SUMO par défaut ».

## 3. Demande de trafic synthétique

Implémentation : `src/trafficmarl/scenarios/demand.py`.

- **Arrivées** : processus de Poisson indépendant sur chaque entrée en bordure du réseau. Le
  débit peut varier dans le temps (processus non homogène simulé par amincissement,
  Lewis & Shedler 1979).
- **Itinéraires** : à chaque carrefour, le véhicule va tout droit avec une probabilité de 0,70,
  tourne à droite avec 0,15 et à gauche avec 0,15, jusqu'à quitter le réseau. Nous n'avons pas
  utilisé de paires origine-destination uniformes : sur un carrefour à 4 branches, un tiers
  des véhicules tournerait alors à gauche et saturerait les voies de tourne-à-gauche bien avant
  les voies principales.
- **Corridor** : les deux extrémités de l'artère reçoivent un débit 3 fois supérieur à celui
  des rues transversales, pour qu'une onde verte soit réellement utile.
- **Profils de demande** : multiplicateurs appliqués au débit de base `entry_rate` :

| Profil | Multiplicateur |
|---|---|
| `low` | 0,5 |
| `medium` | 1,0 |
| `high` | 1,5 |
| `peak` | 0,5 → 1,0 → 1,7 → 1,7 → 1,0 → 0,5 (interpolation linéaire sur 1 h) |
| `saturated` | 2,0 par défaut, ajusté par scénario (voir ci-dessous) ; étude de robustesse |
| `stress` | 2,0, uniquement pour `grid5x5` ; étude de robustesse seule |

**Niveau `saturated`.** Il est défini comme un niveau **au-delà de la saturation du plan fixe,
que le contrôle actionné parvient encore à écouler, téléportation désactivée**. Le
multiplicateur est donc calibré par scénario et vérifié sur les 5 fichiers de routes :
`single` ×2,0, `corridor` ×2,0, `grid3x3` ×1,8 (×2,0 se bloque sur 2 fichiers sur 5),
`grid5x5` ×1,8 (×2,0 se bloque définitivement).

**Niveau `stress`** (`grid5x5` uniquement). ×2,0, simulé avec téléportation. Il sert seulement à
mesurer combien de temps chaque méthode retarde le blocage et quel débit elle maintient. Il
est exclu des tableaux de comparaison principaux.

  Le profil `peak` reproduit une heure de pointe non stationnaire qui dépasse brièvement le
  niveau `high`.
- **Séparation entraînement / évaluation** : 3 fichiers de routes d'entraînement (graines
  1001-1003) et 2 fichiers d'évaluation (graines 2001-2002) par profil. Une graine distincte est
  dérivée pour chaque couple (graine, profil). Les politiques ne sont donc jamais évaluées sur un
  trafic vu pendant l'entraînement.
- Chaque fichier contient **1 h d'insertions** de véhicules.

### Calibration du débit de base

Le débit de base de chaque réseau a été choisi pour que le profil `high` se situe **juste sous
la saturation** du plan fixe par défaut. Ainsi :

- `low` est fluide, `medium` modérément chargé et `high` proche de la saturation ;
- le profil `peak` sature brièvement le réseau, sans blocage généralisé (*gridlock*).

Le balayage se reproduit avec `networks/calibrate_demand.py`. Temps d'attente moyen (s) sous plan
fixe, demande constante, graine 7 :

| Débit par entrée (véh/h) | single | corridor | grid3x3 | grid5x5 |
|---|---|---|---|---|
| 300 | – | 33 | – | – |
| 450 | – | 47 | 45 | 73 |
| 550 | – | 71 | – | – |
| 600 | 19 | – | 47 | 75 |
| 650 | – | 75 | – | – |
| 750 | – | – | 50 | 81 |
| 900 | 20 | – | 57 | 93 |
| 1100 | – | – | 107 | 153 |
| 1200 | 32 | – | – | – |
| 1300 | – | – | 165 | 254 |
| 1400 | 75 | – | – | – |
| 1500 | – | – | 191 (2 téléportations) | – |
| 1600 | 84 | – | – | – |
| 1700 | – | – | 205 (16 téléportations) | – |

Débits retenus (`medium`) : `single` 800, `corridor` 320 (×3 sur l'artère), `grid3x3` 650,
`grid5x5` 600 véh/h par entrée.

## 4. Scénario réaliste : LuST découpé

Source : *Luxembourg SUMO Traffic (LuST) Scenario*, L. Codeca et al., IEEE ITS Magazine, 2017
(licence MIT), cloné dans `external/LuSTScenario`. Le réseau complet (≈ 200 feux, 24 h de trafic)
est trop lourd pour l'apprentissage ; nous en extrayons le centre-ville.

Procédure (`networks/lust/crop_lust.py`, paramètres dans `configs/scenarios/lust.yaml`) :

1. **Sélection** des feux dont le centre se trouve dans un carré de 1,2 km × 1,2 km centré sur
   la **Place d'Armes** (49,6113 N ; 6,1302 E) : 18 feux.
2. **Découpe** avec `netconvert`. Nous conservons toutes les routes accessibles aux voitures
   situées dans le carré, ainsi que toutes les approches des carrefours sélectionnés, afin que
   leurs plans de feux restent valides. Les 4 feux situés en bordure, dont certaines approches
   sont coupées, deviennent des carrefours sans feux. Seule la plus grande composante connexe
   est gardée. Un groupe de 4 feux, relié au reste uniquement par l'extérieur du carré, est
   ainsi écarté. Enfin, un feu dont toutes les phases donnent le vert aux mêmes mouvements
   (`-22774`, probablement un feu piéton privé de sa phase piétons) est transformé en carrefour
   sans feux : aucune décision ne pouvait y changer quoi que ce soit (décision D22). **Il reste
   13 carrefours à feux et 149 tronçons.**
3. **Plans de feux** : nous gardons les plans fixes de LuST (`tll.static.xml`) dans le réseau, ce
   qui donne la ligne de base « plan fixe ». Les plans actionnés d'origine de LuST sont placés
   dans un fichier séparé (`tls_actuated.add.xml`) pour la ligne de base « actionnée ». Ces plans
   LuST ne définissent pas de durées minimale et maximale : SUMO les exécute alors à durée fixe,
   et ils se comportent exactement comme les plans fixes. Nous leur ajoutons donc la plage
   5-50 s par défaut de SUMO. Garder
   les deux plans dans le réseau fait planter SUMO avec SUMO-RL, qui réécrit le premier plan de
   chaque feu (le plan actionné) alors que le plan fixe est actif.
4. **Routes** : les itinéraires LuST sont découpés avec l'outil SUMO `cutRoutes.py`. Celui-ci
   recale l'heure de départ de chaque véhicule sur son entrée dans la zone. Nous utilisons les
   routes à **équilibre dynamique (DUE)** calculées par LuST pour les plans fixes, plus le trafic
   de transit. Les lignes de bus ne sont pas incluses.
5. **Fenêtres horaires** (pointe du matin) : entraînement **07:00-08:00**, évaluation
   **08:00-09:00**. Les véhicules sont disjoints et chaque fenêtre est ramenée à t = 0.
6. **Profils** : la demande réelle est mise à l'échelle par tirage aléatoire avec graine fixe
   (suppression ou duplication de véhicules) : `low` ×0,6, `medium` ×1,0, `high` ×1,3, `peak`
   0,6 → 1,5 → 0,6 sur l'heure, `saturated` ×1,5.

   **Cas particulier du niveau `saturated` pour LuST.** Au-delà de la demande réelle (×1,0),
   le réseau découpé se bloque définitivement dès que la téléportation est désactivée, à tous
   les multiplicateurs testés (×1,2 à ×1,8). La cause est un interblocage à l'intérieur des
   carrefours, lié à la géométrie réelle, et non un excès de demande. L'option de LuST
   `ignore-junction-blocker` ne règle qu'une partie des cas. Avec téléportation, réglage commun
   à toutes les méthodes, ces mêmes simulations s'écoulent : 0,2-0,3 % de téléportations à
   ×1,5. Pour LuST, le niveau `saturated` (×1,5) est donc validé par un critère de
   téléportation (≤ 1 % sous plans actionnés, tous les véhicules arrivés). L'essai sans
   téléportation est enregistré mais non exigé (décision D15).

**Pourquoi DUE plutôt que DUA ?** Nous avons d'abord essayé les routes DUA (plus court chemin)
de LuST. Avec elles, tous les conducteurs empruntent le même boulevard, et la zone découpée se
bloque même à 60 % de la demande : attente moyenne supérieure à 280 s et téléportations, avec
les plans fixes comme avec les plans actionnés. Les routes DUE répartissent le trafic de façon
réaliste.

![Zone LuST découpée](figures/lust_crop.png)

*Figure : réseau LuST autour de la zone découpée (gris), tronçons conservés (bleu), carrefours à
feux contrôlés par les agents (rouge), carré de sélection (pointillés).*

Les carrefours LuST sont hétérogènes : de 2 à 4 phases vertes et de 6 à 12 voies d'arrêt.
L'environnement leur donne la même observation de taille fixe que les réseaux synthétiques, par
remplissage par zéros (*padding*), masquage des actions inexistantes et caractéristiques de type
de carrefour (voir `docs/mdp.md`).

## 5. Validation

Chaque fichier de routes est simulé sans interface graphique jusqu'à ce que tous les véhicules
aient quitté le réseau. Les téléportations sont, dans SUMO, le signe d'un blocage : un véhicule
immobile depuis 300 s est téléporté.

- **Profils `low`, `medium`, `high`, `peak`** : simulation avec les plans fixes, plafond de
  7 200 s. SUMO ne doit signaler aucune erreur, au plus **1 %** des véhicules peuvent être
  téléportés, et tous les véhicules doivent être insérés et arrivés.
- **Profil `saturated`** : simulation avec les **plans actionnés**, plafond de 14 400 s. De la
  congestion est attendue : les téléportations sont donc **rapportées sans seuil** (sauf pour
  LuST, voir section 4). Pour vérifier l'absence de **blocage permanent**, nous refaisons la
  simulation **sans téléportation** et exigeons que tous les véhicules arrivent. Cette
  simulation s'arrête dès qu'aucun véhicule n'est arrivé pendant 600 s alors que d'autres sont
  encore présents : le blocage est alors constaté.
- **Profil `stress`** : plans actionnés, téléportation active, statistiques enregistrées sans
  critère de réussite.

Résultats (moyenne sur les fichiers d'entraînement et d'évaluation de chaque profil) :

| Scénario | Profil | Plans | Fichiers validés | Véhicules (moy.) | Attente moy. (s) | Durée trajet moy. (s) | Téléportations max | Vidange sans téléportation |
|---|---|---|---|---|---|---|---|---|
| `single` | low | fixe | 5/5 | 1580 | 16,8 | 53,3 | 0,00 % | – |
| `single` | medium | fixe | 5/5 | 3199 | 20,0 | 58,0 | 0,00 % | – |
| `single` | high | fixe | 5/5 | 4804 | 45,4 | 93,1 | 0,00 % | – |
| `single` | peak | fixe | 5/5 | 3843 | 44,3 | 91,3 | 0,00 % | – |
| `single` | saturated | actionné | 5/5 | 6396 | 83,7 | 129,5 | 0,00 % | 5/5 |
| `corridor` | low | fixe | 5/5 | 2259 | 28,8 | 81,2 | 0,00 % | – |
| `corridor` | medium | fixe | 5/5 | 4489 | 32,8 | 86,9 | 0,00 % | – |
| `corridor` | high | fixe | 5/5 | 6767 | 63,5 | 127,6 | 0,00 % | – |
| `corridor` | peak | fixe | 5/5 | 5280 | 53,9 | 114,7 | 0,00 % | – |
| `corridor` | saturated | actionné | 5/5 | 8953 | 46,2 | 107,4 | 0,00 % | 5/5 |
| `grid3x3` | low | fixe | 5/5 | 3894 | 43,1 | 117,6 | 0,00 % | – |
| `grid3x3` | medium | fixe | 5/5 | 7850 | 47,7 | 124,6 | 0,00 % | – |
| `grid3x3` | high | fixe | 5/5 | 11575 | 63,5 | 145,6 | 0,00 % | – |
| `grid3x3` | peak | fixe | 5/5 | 9201 | 65,8 | 148,7 | 0,00 % | – |
| `grid3x3` | saturated | actionné | 5/5 | 14008 | 113,5 | 205,0 | 0,00 % | 5/5 |
| `grid5x5` | low | fixe | 5/5 | 6028 | 70,2 | 182,9 | 0,00 % | – |
| `grid5x5` | medium | fixe | 5/5 | 12034 | 76,0 | 191,7 | 0,00 % | – |
| `grid5x5` | high | fixe | 5/5 | 17963 | 92,5 | 214,0 | 0,00 % | – |
| `grid5x5` | peak | fixe | 5/5 | 14173 | 92,1 | 213,1 | 0,00 % | – |
| `grid5x5` | saturated | actionné | 5/5 | 21520 | 135,5 | 269,8 | 0,00 % | 5/5 |
| `grid5x5` | stress (robustesse seule) | actionné | 5/5 | 24017 | 254,6 | 403,7 | 1,09 % | – |
| `lust` | low | fixe | 2/2 | 2594 | 27,5 | 116,7 | 0,00 % | – |
| `lust` | medium | fixe | 2/2 | 4349 | 50,8 | 154,6 | 0,29 % | – |
| `lust` | high | fixe | 2/2 | 5676 | 115,7 | 267,7 | 0,47 % | – |
| `lust` | peak | fixe | 2/2 | 4900 | 109,9 | 253,4 | 0,92 % | – |
| `lust` | saturated | actionné | 2/2 | 6559 | 156,0 | 323,4 | 0,16 % | 0/2 |

Total : 115/115 fichiers de routes validés.

**Évolution de LuST depuis la phase 1.** La version validée en phase 1 comptait 14 feux. Le
feu `-22774`, dont toutes les phases donnaient le vert aux mêmes mouvements, a depuis été
transformé en carrefour sans feux (décision D22). Ce carrefour applique désormais des règles de
priorité, ce qui rend la zone un peu plus congestionnée. Valeurs moyennes, entraînement et
évaluation confondus :

| Profil (plans) | Attente moy. phase 1 → actuelle (s) | Téléportations max phase 1 → actuelle |
|---|---|---|
| low (fixe) | 27,6 → 27,5 | 0,00 % → 0,00 % |
| medium (fixe) | 36,8 → 50,8 | 0,00 % → 0,29 % |
| high (fixe) | 110,7 → 115,7 | 0,10 % → 0,47 % |
| peak (fixe) | 83,7 → 109,9 | 0,11 % → 0,92 % |
| saturated ×1,5 (actionné) | 170,3 → 156,0 | 0,31 % → 0,16 % |

Tous les fichiers LuST restent sous le seuil de 1 % de téléportations. Le fichier `peak` de
l'entraînement en est toutefois proche (0,92 %).

*Les lignes `saturated` et `stress` sont simulées avec les plans **actionnés** : elles ne sont
pas comparables aux lignes simulées avec les plans fixes. Le corridor attend par exemple moins
en `saturated` (actionné) qu'en `high` (plan fixe).*

Le rapport complet (une ligne par fichier) est produit par
`python -m trafficmarl scenarios validate all` dans `results/scenario_validation/`. La
validation s'exécute en parallèle, un processus par simulation, et reprend là où elle s'était
arrêtée : chaque résultat est écrit dès la fin de sa simulation.
