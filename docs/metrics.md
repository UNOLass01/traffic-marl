# Indicateurs d'évaluation

Toutes les méthodes, lignes de base comme agents appris, sont évaluées par **le même code**
(`src/trafficmarl/metrics/`), à partir des fichiers produits par SUMO lui-même. L'évaluation se
fait sur les fichiers de routes **réservés à l'évaluation**, avec les **mêmes graines SUMO** pour
toutes les méthodes. Chaque couple (graine, fichier d'évaluation) constitue un épisode, simulé
dans son propre processus.

## Indicateurs par épisode

| Indicateur | Définition |
|---|---|
| Temps d'attente moyen (s) | `waitingTime` de SUMO pour les véhicules **arrivés** (indicateur principal) |
| Attente maximale et 95e centile (s) | sur tous les véhicules insérés, arrivés ou non : mesure la privation de vert vue par les usagers |
| Retard d'insertion (s) | `departDelay` des véhicules insérés ; pour un véhicule jamais inséré : fin − départ prévu |
| Véhicules non terminés | encore dans le réseau à la fin + jamais insérés |
| Retard total (s, véh·h) | retard d'insertion + temps perdu (`timeLoss`) pour les véhicules arrivés ou en cours ; fin − départ prévu pour les véhicules jamais insérés ; moyenne sur tous les véhicules dont le départ était prévu |
| Débit | nombre de véhicules arrivés à destination |
| Temps de parcours, temps perdu (s) | véhicules arrivés |
| CO₂ (kg, g/véhicule) | modèle d'émissions HBEFA de SUMO, tous les véhicules insérés |
| File d'attente moyenne | véhicules à l'arrêt, moyennés dans le temps, sur les approches (≤ 200 m) et dans tout le réseau |
| Téléportations | nombre et taux (÷ véhicules insérés) ; signalement au-delà de 1 % |
| Blocage | épisode arrêté par la détection de blocage (aucune arrivée pendant 600 s) |
| Durées de vert, rouge le plus long par phase, changements de phase | mesurés sur l'état des feux enregistré par SUMO à chaque seconde, identiquement pour les plans de feux et pour les agents |
| Communication | nombres échangés par pas de décision et par épisode (observation étendue) |

Le temps d'attente SUMO ne compte que les véhicules arrivés. Un contrôleur qui bloquerait
l'entrée du réseau pourrait donc paraître bon sur ce seul critère. C'est pourquoi le retard
total, les véhicules non terminés et le débit sont **toujours** rapportés à côté (décisions D17
et D26).

## Règle de comparaison

Les statistiques portent sur les valeurs **par graine** (moyenne sur les fichiers d'évaluation de
la graine) : moyenne, écart-type et intervalle de confiance à 95 % (loi de Student). Une méthode
**bat** une référence (par défaut le plan fixe de Webster) si les deux conditions suivantes sont
réunies (décision D27) :

1. son temps d'attente moyen est plus faible, avec des intervalles de confiance à 95 % disjoints
   ou un test de Welch donnant p < 0,05 ;
2. elle n'augmente ni les téléportations ni les véhicules non terminés : une hausse n'est retenue
   que si elle n'est pas due au bruit entre graines (test de Welch unilatéral, p < 0,05).

## Utilisation

```bash
python -m trafficmarl evaluate single --method fixed_time --profile medium --seeds 0 1 2 3 4
python experiments/evaluate.py --config configs/experiments/single_medium_baselines.yaml --method actuated
```

Chaque exécution écrit `results/<expérience>/<méthode>/<graine>/metrics.csv`, avec une ligne par
fichier d'évaluation, ainsi que la configuration résolue et le commit git.
