# Spécification du MDP

Ce document décrit, avec leurs justifications, le processus de décision markovien multi-agent
utilisé par tous les contrôleurs appris : observations, actions, contraintes et récompenses.
Il sera complété au fil de la phase 2 ; les choix déjà arrêtés sont décrits ci-dessous.

## Portée de l'observation sur chaque approche

**Choix.** Les caractéristiques par voie entrante (longueur de file d'attente, densité, vitesse
moyenne) sont mesurées sur **toute l'approche en amont de la ligne d'arrêt**, en remontant les
tronçons successifs, **jusqu'à une distance maximale fixe et configurable** (par défaut
**200 m**). Chaque grandeur est normalisée par rapport à cette distance : la file d'attente et
la densité sont divisées par le nombre de véhicules qui tiennent dans 200 m, la vitesse par la
vitesse limite.

**Justification.**

- SUMO-RL n'observe que le dernier tronçon de voie avant le carrefour. Dans nos réseaux
  synthétiques, ce tronçon ne mesure que 100 m, car la voie de tourne-à-gauche coupe l'approche
  en deux. Il est souvent plus court encore dans LuST. Les files plus longues sont donc tronquées,
  et l'agent ne distingue pas une file de 100 m d'une file de 300 m.
- Plafonner à une distance fixe garde une **taille d'observation constante** et une
  **normalisation identique** pour tous les carrefours et tous les scénarios. C'est indispensable
  pour partager une même politique entre agents (décision D4) et pour l'étude de transfert
  (grid3x3 → grid5x5).
- 200 m correspond à la longueur d'un îlot dans les réseaux synthétiques. C'est aussi un ordre
  de grandeur réaliste pour la portée d'une caméra ou d'une boucle de détection en amont.
  L'observation reste ainsi locale, sans empiéter sur le carrefour voisin.

La distance est un paramètre de configuration (`env.observation.max_distance`) et pourra faire
l'objet d'une ablation.

## Contraintes de durée des phases et équité de la comparaison

Toutes les méthodes, contrôleurs appris comme lignes de base (plan fixe, actionné,
max-pressure), respectent **les mêmes contraintes** :

- **vert minimal** : une phase verte ne peut pas être interrompue avant sa durée minimale ;
- **vert maximal** : une phase verte ne peut pas être prolongée au-delà de sa durée maximale,
  pour éviter qu'une direction soit privée de vert indéfiniment ;
- **jaune** : de même durée pour toutes les méthodes, inséré automatiquement à chaque
  changement de phase.

La comparaison est ainsi équitable : aucune méthode ne gagne en enfreignant une contrainte que
les autres respectent.

Pour les agents, ces contraintes sont appliquées par **masquage d'actions** (décision D13).
Pendant le vert minimal, seule l'action « garder la phase courante » est autorisée. Une fois le
vert maximal atteint, cette même action est interdite. Le même masque cache les phases
inexistantes des carrefours LuST qui ont moins de phases que le maximum. Toute action transmise
au simulateur est donc exécutée telle quelle. Le nombre de pas où une contrainte était active
est enregistré avec les résultats.

## Conditions de simulation identiques pour toutes les méthodes

Toutes les méthodes, lignes de base comme agents appris, à l'entraînement comme à l'évaluation,
sont simulées dans **exactement les mêmes conditions** (décision D14) :

- **téléportation après 300 s d'immobilité** : valeur par défaut de SUMO. SUMO-RL désactive la
  téléportation par défaut ; nous rétablissons ce comportement. Le nombre et le taux de
  téléportations sont rapportés avec chaque résultat. Une exécution dépassant 1 % de véhicules
  téléportés est signalée dans les tableaux et les figures (décision D17), afin qu'une politique
  ne puisse pas paraître performante en provoquant des blocages que les téléportations
  résorbent ;
- **durée d'épisode fixe** : 1 h d'insertion de véhicules, suivie d'une période fixe de vidange
  du réseau (1 800 s par défaut) ;
- **détection de blocage** : si aucun véhicule n'arrive à destination pendant 600 s alors que
  des véhicules sont encore présents, l'épisode s'arrête et est marqué comme bloqué.
