# Résultats

Ce document présente les résultats de l'étude finale. Tous les tableaux et figures sont
régénérés depuis `results/` par `python analysis/study.py` puis `python analysis/summary.py`
(voir README, section « Reproducing the study »). Les figures citées sont copiées dans
`docs/figures/`.

## 1. Protocole

- **Réseaux :** `single` (1 carrefour), `corridor` (4), `grid3x3` (9), `lust` (13 feux, zone
  centrale découpée de LuST, voir `docs/scenarios.md`), `grid5x5` (25).
- **Demandes :** apprentissage sur la demande `medium` uniquement ; évaluation sur `medium`,
  `high`, `peak` (heure de pointe non stationnaire) et `saturated` (au-delà de la saturation du plan
  fixe, calibrée par scénario, décisions D10 et D15).
- **Évaluation :** fichiers de routes **réservés à l'évaluation** (jamais vus à l'apprentissage),
  politique déterministe, mêmes graines SUMO pour toutes les méthodes, indicateurs calculés par le
  même code (`docs/metrics.md`).
- **Graines :** lignes de base 5 graines (10 sur LuST). Méthodes apprises : **3 graines
  d'apprentissage** (5 pour `ippo_local` et `ippo_extended` sur corridor et grid3x3), au lieu des
  5 du protocole initial (décision D48). Chaque tableau indique son `n`.
- **Statistiques :** moyenne ± demi-largeur de l'intervalle de confiance à 95 % (loi de Student) ;
  comparaisons par test de Welch (D27). Une méthode apprise est « acceptée » si elle bat
  significativement la meilleure ligne de base à programme fixe, sans dégrader le retard en queue
  de distribution ni le débit (D33) ; battre max-pressure n'est pas exigé.
- **Budget :** 300 000 pas d'environnement par apprentissage pour toutes les méthodes apprises,
  avec les mêmes hyperparamètres PPO pour IPPO, MAPPO, PPO centralisé et le fédéré.

### Méthodes

| Groupe | Méthode | Nom dans les tableaux |
|---|---|---|
| Traditionnel | Plan fixe SUMO, Webster, actionné (*actuated*) | `fixed_time`, `webster`, `actuated` |
| Traditionnel | Max-pressure, avec et sans limite de rouge maximal | `max_pressure`, `max_pressure_maxred` |
| Centralisé | Un seul agent PPO pour tout le réseau (Stable-Baselines3) | `centralized_ppo` |
| Indépendant | IPPO, récompense locale, partage de paramètres | `ippo_local` |
| Coordination implicite | IPPO, récompense de voisinage / globale | `ippo_neighborhood`, `ippo_global` |
| Coordination explicite | IPPO avec observation des voisins (messages) | `ippo_extended` |
| CTDE | MAPPO, QMIX, MADDPG (action de durée de phase) | `mappo`, `qmix`, `maddpg` |
| Fédéré | Politique par feu, FedAvg tous les 1, 5 ou 20 itérations | `fedavg_r1`, `fedavg_r5`, `fedavg_r20` |

Le partage de paramètres est utilisé par toutes les variantes IPPO et par MAPPO, QMIX et MADDPG
(réseau d'agent commun). Il n'est pas utilisé par les politiques indépendantes et fédérées.

## 2. Tableau principal

Temps d'attente moyen par véhicule (s), évaluation sur demande moyenne et heure de pointe. Les
cases vides correspondent à des combinaisons non lancées (décision D48). « n=1 » : sonde de
passage à l'échelle (décision D35).

| Méthode | single moy. | single pointe | corridor moy. | corridor pointe | grid3x3 moy. | grid3x3 pointe | grid5x5 moy. | grid5x5 pointe | lust moy. | lust pointe |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed_time | 20.2 | 45.9 | 33.1 | 54.6 | 47.9 | 64.1 | 75.6 | 89.0 | 41.6 | 100.4 |
| webster | 16.1 | 47.0 | 20.7 | 38.3 | 53.1 | 90.3 | 69.0 | 120.1 | 43.5 | 114.5 |
| actuated | 14.9 | 43.7 | 17.4 | 24.1 | 35.1 | 58.8 | 55.0 | 79.7 | 35.2 | 93.4 |
| max_pressure | 7.4 | **26.2** | **9.7** | **12.6** | **14.8** | **25.1** | **22.8** | **31.8** | 24.4 | 76.7 |
| max_pressure_maxred | 8.5 | 39.4 | 11.4 | 15.1 | 19.6 | 39.1 | 31.1 | 45.5 | **23.7** | 86.8 |
| centralized_ppo | 13.6 | 45.3 | 9.8 | 14.0 | 19.6 | 35.9 | 178.5 (n=1) | 178.1 (n=1) | | |
| ippo_local | **6.9** | 31.1 | 10.7 | 14.3 | 19.8 | 31.7 | 31.6 | 42.6 | 40.0 | **72.6** |
| ippo_neighborhood | | | 10.4 | 14.0 | 19.4 | 32.1 | | | | |
| ippo_global | | | 16.1 | 21.7 | 41.6 | 61.9 | | | | |
| ippo_extended | | | 10.8 | 14.2 | 18.7 | 31.4 | 28.4 | 39.4 | 29.8 | 102.6 |
| mappo | | | 17.9 | 22.3 | 46.6 | 62.1 | | | | |
| qmix | | | 13.2 | 15.8 | 24.0 | 39.2 | | | | |
| maddpg | | | 27.0 | 32.8 | 48.6 | 75.6 | | | | |

Les tableaux complets (file d'attente, débit, CO₂, retard total, 95e centile, téléportations,
intervalles de confiance) sont dans `results/tables/summary.md`, `main.md` et `coordination.md`.
Figures : `docs/figures/metrics_<scénario>.png`.

**Débit et détournement de la récompense.** Sur medium et pointe, toutes les méthodes font
arriver exactement les mêmes véhicules (par exemple 7 766 sur grid3x3 medium), sans véhicule non
terminé ni téléportation. Aucune méthode apprise n'améliore donc son temps d'attente en bloquant
l'entrée du réseau. Seule exception : la sonde centralisée sur grid5x5 (11 884 véhicules au lieu
de 12 052).

**CO₂.** Les émissions suivent le temps d'attente. Sur grid3x3 medium : 1 951 kg avec le plan
fixe, 1 585 kg avec max-pressure, 1 645 kg avec IPPO local et 1 636 kg avec IPPO et observation
des voisins.

### Lecture

1. **Toutes les méthodes apprises raisonnables battent les plans de feux classiques.** IPPO
   réduit le temps d'attente de 59 % sur grid3x3 medium par rapport au plan fixe (19.8 s contre
   47.9 s) et de 44 % par rapport au contrôle actionné. Le critère d'acceptation de la phase 5
   (D33) est atteint sur single et corridor.
2. **Max-pressure reste la meilleure méthode dans presque tous les cas.** Sur grid3x3 medium :
   14.8 s, contre 18.7 s pour la meilleure méthode apprise. Max-pressure choisit à chaque pas la
   phase qui écoule le plus de véhicules, ce qui est presque optimal sur des grilles régulières.
   Elle n'a cependant **aucune limite de rouge maximal** : le rouge le plus long atteint 1 774 s
   sur grid3x3 medium, contre environ 230 s pour les agents RL, qui respectent la contrainte
   d'équité du MDP. À contrainte égale (`max_pressure_maxred`), max-pressure est au niveau d'IPPO
   (19.6 s contre 19.8 s sur grid3x3 medium) et derrière IPPO en pointe (39.1 s contre 31.7 s).
3. **Sur l'intersection isolée, IPPO bat max-pressure** (6.9 s contre 7.4 s).
4. **Parmi les méthodes apprises, IPPO avec observation des voisins est la meilleure** sur les
   grilles. Le PPO centralisé fait jeu égal sur les petits réseaux, mais ne passe pas à
   l'échelle (section 5).

## 3. Coordination et ablations (récompense, observation)

Tableau `results/tables/ablation.md`, figure `docs/figures/coordination_curves.png`. Référence :
IPPO, récompense locale. Différence relative du temps d'attente (test de Welch) :

| Scénario | Récompense de voisinage | Récompense globale | Observation des voisins |
|---|---|---|---|
| corridor medium | −2.8 % (p = 0.47) | **+50 %** (p = 0.001) | +0.8 % (p = 0.86) |
| grid3x3 medium | −1.7 % (p = 0.41) | **+110 %** (p < 0.001) | **−5.2 %** (p = 0.048) |
| grid3x3 high | +2.2 % (p = 0.33) | **+89 %** (p = 0.007) | **−4.3 %** (p = 0.039) |
| grid5x5 medium | — | — | **−10.2 %** (p = 0.001) |
| grid5x5 high | — | — | **−9.5 %** (p < 0.001) |
| grid5x5 peak | — | — | **−7.4 %** (p = 0.009) |
| lust high | — | — | **−29.5 %** (p = 0.039) |

- **La récompense globale dégrade fortement l'apprentissage** (+50 % à +110 %). Avec un seul
  signal partagé par tous les agents, chaque agent ne sait plus quelle part de la récompense
  dépend de ses propres actions (problème d'attribution du crédit). L'effet grandit avec le
  nombre d'agents.
- **La récompense de voisinage (α = 0.5) ne change rien de significatif** sur corridor et
  grid3x3.
- **La coordination explicite (observation des voisins) est utile, et son effet grandit avec la
  taille du réseau** : nul sur corridor, −5 % sur grid3x3 medium, −7 à −10 % sur grid5x5 pour
  toutes les demandes. Sur LuST, le gain est significatif en demande forte mais les résultats
  varient beaucoup d'une graine à l'autre (section 8).
- **Coût de communication :** 36 nombres par pas de décision sur corridor et 144 sur grid3x3,
  à l'exécution comme à l'apprentissage. Les récompenses de voisinage et globale ne coûtent rien
  à l'exécution, seulement 6 à 24 nombres par pas pendant l'apprentissage. Interprétation
  réseau mobile : un message par carrefour et par pas de 5 s vers ses voisins directs, ce qui
  tient dans une liaison V2I ou de calcul en périphérie (*edge computing*) même modeste.

L'ablation de la pénalité de changement de phase n'a pas été lancée (D48). Seule la valeur par
défaut c = 0.5 est mesurée : 234 changements de phase par feu et par épisode sur grid3x3.

## 4. Méthodes CTDE (QMIX, MADDPG, MAPPO)

| Méthode | corridor medium | grid3x3 medium | grid3x3 pointe |
|---|---|---|---|
| IPPO, local (référence) | 10.7 ± 0.6 | 19.8 ± 1.0 | 31.7 ± 1.3 |
| QMIX | 13.2 ± 4.9 | 24.0 ± 6.2 | 39.2 ± 19.4 |
| MAPPO | 17.9 ± 1.9 | 46.6 ± 7.2 | 62.1 ± 4.3 |
| MADDPG | 27.0 ± 9.1 | 48.6 ± 14.6 | 75.6 ± 31.8 |

- Les courbes d'apprentissage se stabilisent sur grid3x3 pour les trois méthodes (critère de la
  phase 7 ; `results/tables/training.md`, `docs/figures/curves_grid3x3.png`), à deux réserves
  près : MADDPG corridor graine 0 n'a pas atteint de plateau, et MADDPG grid3x3 graine 1 régresse
  en fin d'apprentissage.
- **QMIX** est la meilleure méthode CTDE et la plus proche d'IPPO, avec une forte variance entre
  graines. Il a connu quelques blocages pendant l'apprentissage (2, 2 et 9 épisodes sur grid3x3).
- **MAPPO** apprend avec la récompense globale (D45) et se comporte comme `ippo_global` (46.6 s
  contre 41.6 s). Le critique centralisé ne compense pas le problème d'attribution du crédit
  d'une récompense commune.
- **MADDPG** agit sur la durée des phases et non sur le choix de la phase suivante. Cette action
  est moins réactive, et l'algorithme est connu pour sa sensibilité aux hyperparamètres, qui
  n'ont pas été optimisés (D45).

**Conclusion :** sur ces réseaux, l'apprentissage centralisé avec exécution décentralisée
n'apporte rien par rapport à IPPO avec partage de paramètres et récompense locale. Les agents
étant homogènes et le couplage entre carrefours modéré, une politique partagée apprise sur des
signaux locaux suffit.

## 5. Passage à l'échelle

Tableau `results/tables/scalability.md`, figure `docs/figures/scalability.png` (amélioration
par rapport à Webster, temps et vitesse d'apprentissage en fonction du nombre de carrefours).

| Carrefours | Max-pressure | IPPO local | IPPO + voisins | PPO centralisé | Temps d'apprentissage IPPO |
|---|---|---|---|---|---|
| 1 (single) | 7.4 s | 6.9 s | — | 13.6 s | 6 min |
| 4 (corridor) | 9.7 s | 10.7 s | 10.8 s | 9.8 s | 14 min |
| 9 (grid3x3) | 14.8 s | 19.8 s | 18.7 s | 19.6 s | 40 min |
| 13 (LuST) | 24.4 s | 40.0 s | 29.8 s | — | 41 min |
| 25 (grid5x5) | 22.8 s | 31.6 s | 28.4 s | 178.5 s (sonde) | 106 min |

- **Le PPO centralisé ne passe pas à l'échelle.** Il est compétitif jusqu'à 9 carrefours.
  Sur 25 carrefours, la sonde (100 000 pas, une graine, décision D35) donne 178.5 s, plus du
  double du plan fixe (75.6 s), avec des véhicules non terminés. L'espace d'actions joint et
  l'observation concaténée croissent avec le nombre de carrefours, alors que chaque pas
  d'apprentissage ne fournit qu'une seule transition.
- **Les méthodes multi-agents passent à l'échelle.** Avec le partage de paramètres, chaque pas
  fournit une transition par carrefour : la quantité de données croît avec la taille du réseau.
  IPPO reste à 54–59 % de mieux que Webster sur 25 carrefours.
- **Coût d'apprentissage :** le temps d'apprentissage croît à peu près linéairement avec le
  nombre de carrefours (la simulation SUMO domine). La vitesse passe de 818 pas/s (1 carrefour) à
  48 pas/s (25 carrefours). Les temps mesurés sont surestimés, car deux apprentissages tournaient
  en parallèle sur la même machine (D44, D48).

## 6. Robustesse (demande non vue)

Tableau `results/tables/robustness.md`, figure `docs/figures/robustness.png`. Apprentissage sur
medium uniquement, test sur high, pointe et saturée.

| Scénario | Méthode | medium | high | pointe | saturée |
|---|---|---|---|---|---|
| grid3x3 | Plan fixe | 47.9 | 64.4 | 64.1 | 137.4 |
| grid3x3 | Max-pressure | 14.8 | 25.0 | 25.1 | 65.3 |
| grid3x3 | IPPO local | 19.8 | 32.2 | 31.7 | 77.1 |
| grid3x3 | IPPO + voisins | 18.7 | 30.8 | 31.4 | 86.6 |
| grid5x5 | Plan fixe | 75.6 | 92.8 | 89.0 | 151.3 |
| grid5x5 | Max-pressure | 22.8 | 34.4 | 31.8 | 64.0 |
| grid5x5 | IPPO local | 31.6 | 46.2 | 42.6 | 75.3 |
| grid5x5 | IPPO + voisins | 28.4 | 41.8 | 39.4 | 75.2 |

- Les politiques apprises **généralisent à des demandes non vues** : elles restent nettement
  meilleures que le plan fixe et le contrôle actionné en demande forte, en pointe et en
  saturation, sur toutes les grilles.
- L'écart avec max-pressure se maintient, sans s'aggraver de façon systématique, quand la demande
  augmente.
- En saturation, la variance entre graines augmente (par exemple IPPO voisinage sur grid3x3 :
  91.2 ± 31.0 s). Les politiques n'ont jamais vu ce régime à l'apprentissage.

## 7. Transfert grid3x3 → grid5x5

Tableau `results/tables/transfer.md`. Les politiques apprises sur grid3x3 sont évaluées sur
grid5x5 **sans réapprentissage**. Le partage de paramètres et une observation de taille fixe par
carrefour rendent cela possible.

| Demande | Méthode | Transférée | Réapprise sur grid5x5 | Actionné | Max-pressure |
|---|---|---|---|---|---|
| medium | IPPO + voisins | 30.5 ± 1.1 | 28.4 | 55.0 | 22.8 |
| medium | IPPO local | 31.5 ± 3.8 | 31.6 | 55.0 | 22.8 |
| high | IPPO + voisins | 42.9 ± 0.3 | 41.8 | 88.3 | 34.4 |
| pointe | IPPO + voisins | 40.9 ± 0.8 | 39.4 | 79.7 | 31.8 |

Le transfert fonctionne : la politique transférée reste à moins de 8 % de la politique réapprise, et bien
meilleure que les plans de feux classiques. Interprétation pratique : une politique apprise sur un
quartier peut être déployée sur un réseau plus grand, ce qui économise les 106 minutes
d'apprentissage sur grid5x5.

## 8. Scénario réaliste (LuST)

Le scénario LuST est le plus difficile. Le réseau est irrégulier et les carrefours sont hétérogènes
(voir `docs/scenarios.md`). Max-pressure reste la
meilleure méthode en demande moyenne (24.4 s). IPPO avec observation des voisins (29.8 s) bat le
plan fixe (41.6 s) et le contrôle actionné (35.2 s). **La variance entre graines est forte** :
la graine 0 d'IPPO local a appris une politique médiocre (d'où 40.0 ± 63.5 s), et IPPO + voisins
se dégrade en pointe (102.6 ± 104.5 s). Avec 3 graines, ces résultats LuST sont indicatifs.
En pointe, la meilleure méthode est IPPO local (72.6 s, contre 76.7 s pour max-pressure).

## 9. Carte de congestion

Figures `docs/figures/heatmap_grid5x5_medium.png` et `heatmap_grid5x5_peak.png` : nombre moyen
de véhicules à l'arrêt par carrefour, sur grid5x5. En pointe, ce nombre vaut 12.1 avec le plan
fixe, 4.1 avec max-pressure, 5.3 avec IPPO local et 4.9 avec IPPO + voisins. Le plan fixe
congestionne tout le réseau. Les méthodes adaptatives répartissent la file d'attente de façon
homogène, sans carrefour en cascade de congestion.

## 10. Extension fédérée

Tableau `results/tables/federated.md`, figure `docs/figures/federated.png`. Scénario corridor,
une politique PPO par feu, agrégation FedAvg (décision D46).

| Schéma | Attente medium (s) | Attente pointe (s) | Tours | Mo échangés (apprentissage) | Données brutes partagées ? |
|---|---|---|---|---|---|
| Partage de paramètres (IPPO) | 10.7 ± 0.6 | 14.3 ± 0.6 | — | 242 (trajectoires) | oui |
| Politiques indépendantes | 10.8 ± 1.0 | 14.5 ± 0.5 | 0 | 0 | non |
| FedAvg à chaque itération | 10.5 ± 0.3 | 13.9 ± 0.1 | 74 | 373 | non |
| FedAvg toutes les 5 itérations | 10.5 ± 1.0 | 14.1 ± 0.3 | 15 | 76 | non |
| FedAvg toutes les 20 itérations | 10.2 ± 0.5 | 13.8 ± 0.6 | 4 | 20 | non |

- **Le fédéré atteint la même performance que le partage de paramètres centralisé**, sans
  qu'aucune observation ni trajectoire ne quitte le carrefour. Aucune différence n'est
  significative sur medium. FedAvg toutes les 20 itérations est même légèrement meilleur en high
  (−2.6 %, p = 0.026).
- **Compromis communication :** agréger moins souvent ne dégrade pas la performance ici. Toutes
  les 20 itérations, l'apprentissage complet échange 20 Mo (5 Mo par tour), soit 12 fois moins
  que les 242 Mo de trajectoires brutes que le partage de paramètres centralise.
- **Limite :** sur corridor, les 4 carrefours sont presque homogènes, donc les politiques
  indépendantes font aussi bien. L'intérêt du fédéré (mutualiser l'expérience sans partager les
  données) devrait apparaître sur des réseaux hétérogènes ou plus grands. Les études sur grid3x3
  et avec confidentialité différentielle (implémentée et testée) n'ont pas été lancées (D48).
- **Lecture « réseaux mobiles » :** chaque carrefour est un nœud de calcul en périphérie qui
  apprend localement. Le serveur d'agrégation peut être un nœud MEC du quartier. Seuls des
  paramètres (5 Mo par tour) transitent, à une fréquence réglable selon la bande passante
  disponible.

## 11. Limites

- **3 graines au lieu de 5** pour la plupart des méthodes apprises (D48) : les intervalles de
  confiance sont larges, et « pas de différence significative » est une preuve faible
  d'équivalence.
- **Hyperparamètres de QMIX et MADDPG non optimisés** (D45) : leurs résultats sont une borne
  basse de ce que ces méthodes peuvent atteindre.
- **Combinaisons non lancées** (D48) : CTDE et récompenses de voisinage/globale sur grid5x5 et
  LuST, fédéré sur grid3x3, ablation de la pénalité de changement, confidentialité
  différentielle. Les fichiers de configuration du protocole complet sont fournis (D47).
- **Délais d'échantillonnage RLlib sur grid5x5** (D49) : quelques itérations initiales des graines
  0 et 2 d'IPPO local ont utilisé des lots réduits.
- **Temps d'apprentissage surestimés** : deux apprentissages en parallèle partageaient le
  processeur.
- **Max-pressure n'a pas de contrainte de rouge maximal**, contrairement aux agents RL. La
  comparaison à contrainte égale (`max_pressure_maxred`) est plus favorable aux agents.
- **Convergence :** la plupart des courbes atteignent un plateau à 300 000 pas, mais pas toutes
  (MADDPG corridor graine 0). Un budget plus long pourrait améliorer certaines méthodes.

## 12. Conclusion

1. **Un agent RL par carrefour** (IPPO avec partage de paramètres) réduit le temps d'attente de
   58 à 68 % par rapport au plan fixe SUMO sur single, corridor et les grilles, sans perte de débit, et passe à l'échelle
   jusqu'à 25 carrefours.
2. **La meilleure stratégie de coopération est la coordination explicite légère** : observer un
   résumé de l'état des voisins. Son gain grandit avec la taille du réseau (−10 % sur grid5x5),
   pour un coût de communication faible. La récompense globale est à éviter.
3. **Le contrôle centralisé** est compétitif sur petit réseau mais s'effondre à 25 carrefours.
4. **Max-pressure reste une référence très forte** : les agents appris ne la dépassent que sur
   l'intersection isolée et sur LuST en pointe. À contrainte d'équité égale (rouge maximal), ils
   font jeu égal ou mieux.
5. **Le fédéré** atteint la performance du partage de paramètres sans partager de données brutes,
   avec un coût de communication réglable.

## 13. Démonstration

La procédure d'enregistrement de la vidéo comparant le plan fixe et la meilleure méthode apprise
dans l'interface graphique de SUMO est décrite dans le README (section « Demo (SUMO GUI) »). Par
exemple :

```bash
unset LIBSUMO_AS_TRACI
python experiments/demo.py --scenario grid5x5 --profile peak --method fixed_time
python experiments/demo.py --scenario grid5x5 --profile peak --method ippo_extended --seed 0
```
