# Spécification du MDP

Ce document décrit, avec leurs justifications, le processus de décision markovien multi-agent
utilisé par tous les contrôleurs appris : agents, observations, actions, contraintes, récompenses
et déroulement des épisodes. Les paramètres se trouvent dans la section `env` de
`configs/base.yaml` ; l'implémentation est dans `src/trafficmarl/envs/`. Les décisions
correspondantes sont tracées dans `docs/decisions.md` (D12 à D14 et D19 à D23).

## 1. Agents

Chaque carrefour à feux est un agent : 1 (`single`), 4 (`corridor`), 9 (`grid3x3`), 25
(`grid5x5`) et 13 (`lust`). L'environnement suit l'interface `ParallelEnv` de PettingZoo : tous
les agents décident en même temps, toutes les `delta_time` = **5 s** de simulation.

Il s'appuie sur SUMO-RL, étendu uniquement par ses points d'extension officiels : classe
d'observation (`observation_class`) et fonction de récompense (`reward_fn`). Une seule méthode
est surchargée par héritage, sans copier le code de SUMO-RL : le pas de simulation, qui compte
arrivées et téléportations à chaque seconde.

## 2. Observation

### 2.1 Portée de l'observation sur chaque approche

**Choix.** Les caractéristiques par voie entrante sont mesurées sur **toute l'approche en amont
de la ligne d'arrêt**, en remontant les tronçons successifs, **jusqu'à une distance maximale fixe
et configurable** (`env.observation.max_distance`, **200 m** par défaut). Le parcours s'arrête au
carrefour à feux précédent et à la limite du réseau : l'observation reste locale.

Une voie amont qui alimente plusieurs voies d'arrêt est partagée entre elles. C'est le cas du
tronçon situé avant la voie de tourne-à-gauche : s'il alimente k voies, il compte pour 1/k dans
chacune, de sorte que la somme sur les voies d'arrêt redonne le nombre réel de véhicules.

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

En pratique, l'approche mesure en moyenne 170 à 180 m dans les réseaux synthétiques, la distance
entre carrefours moins leur emprise. Dans LuST, elle mesure 142 m en moyenne et 34 m au minimum,
car les carrefours y sont plus proches.

### 2.2 Observation locale

Pour un carrefour, avec `L` = nombre maximal de voies d'arrêt du scénario (12) et `P` = nombre
maximal de phases vertes (4) :

| Bloc | Taille | Normalisation |
|---|---|---|
| File d'attente par voie (véhicules à l'arrêt, < 0,1 m/s) | L | ÷ nombre de véhicules tenant dans 200 m (7,5 m par véhicule), écrêtée à 1 |
| Densité par voie (véhicules présents) | L | idem |
| Vitesse moyenne par voie | L | ÷ vitesse limite ; voie vide = 1 (circulation libre), voie de remplissage = 0 |
| Phase verte courante (one-hot) | P | — |
| Temps depuis le dernier changement de phase | 1 | ÷ vert maximal, écrêté à 1 |
| Indicateur « vert minimal en cours » | 1 | 0 / 1 |
| Type de carrefour : nombre de phases vertes (one-hot) | P | — |
| Type de carrefour : nombre de voies d'arrêt | 1 | ÷ L |
| *Option :* identifiant d'agent (one-hot) | N | — |

Soit **47 valeurs**, toutes dans [0, 1]. Les carrefours qui ont moins de voies ou de phases sont
complétés par des zéros (*padding*). Les caractéristiques de type permettent à une même politique
partagée de distinguer les carrefours hétérogènes de LuST (2 à 4 phases, 6 à 12 voies) tout en
restant transférables d'un réseau à l'autre. L'identifiant d'agent, qui empêcherait ce transfert,
n'est qu'une option d'ablation (`env.observation.agent_id`, décision D20).

### 2.3 Observation étendue (coordination explicite)

L'observation étendue ajoute quatre emplacements de voisins, un par direction **N, E, S, O**.
Chacun contient le « message » du voisin, c'est-à-dire ce qu'il partage :

- file moyenne et densité moyenne sur ses voies d'arrêt ;
- sa phase verte courante (one-hot, P valeurs) ;

suivi d'un indicateur de présence. Cela fait 4 × (2 + 4 + 1) = 28 valeurs, soit 75 au total.

**Graphe de voisinage.** Deux carrefours sont voisins si une route les relie sans traverser un
troisième carrefour à feux.

**Affectation aux emplacements.** Un voisin est placé dans l'emplacement correspondant à
l'**orientation (cap) du carrefour vers ce voisin**, par secteurs de 90° centrés sur N, E, S et O.
Si plusieurs voisins tombent dans le même emplacement, ou si un carrefour a plus de 4 voisins,
leurs messages sont **moyennés**. Une phase moyennée devient alors une répartition entre phases.
Cette règle est exacte sur les grilles, où il y a au plus un voisin par direction. Sur LuST, elle
réalise une agrégation par direction dont la fréquence est mesurée :

| Scénario | Voisins (min / moy. / max) | Agents avec > 4 voisins | Emplacements agrégés | Agents concernés |
|---|---|---|---|---|
| corridor | 1 / 1,5 / 2 | 0 | 0 | 0 |
| grid3x3 | 2 / 2,67 / 4 | 0 | 0 | 0 |
| grid5x5 | 2 / 3,2 / 4 | 0 | 0 | 0 |
| lust | 1 / 4,15 / 7 | 4 | 12 | 10 sur 13 |

**Coût de communication.** Chaque voisin envoie un message de 2 + P = 6 nombres par pas de
décision. Le volume total échangé par pas de décision (somme sur tous les agents de leurs
voisins) est de 36 nombres pour `corridor`, 144 pour `grid3x3`, 480 pour `grid5x5` et 324 pour
`lust`. Ces volumes seront comparés au coût de l'agrégation fédérée (phase 9).

Les récompenses partagées (section 4) ne coûtent qu'**à l'entraînement** : une politique
entraînée n'utilise plus de récompense. `neighborhood` envoie le terme de trafic de chaque agent
à chacun de ses voisins (1 nombre par lien orienté : 6 pour `corridor`, 24 pour `grid3x3`, 80
pour `grid5x5`) ; `global` passe par un agrégateur, auquel chaque agent envoie son terme et dont
il reçoit la moyenne (2 nombres par agent : 8, 18 et 50). L'évaluation rapporte les deux volumes,
à l'exécution et à l'entraînement (`coordination/communication.py`, décision D43).

## 3. Actions et contraintes

**Action.** Chaque agent choisit la **prochaine phase verte** parmi P = 4 (espace discret
commun, complété par masquage pour les carrefours qui ont moins de phases). Choisir la phase
courante la prolonge de 5 s. Choisir une autre phase déclenche automatiquement une phase jaune,
de 3 s dans les réseaux synthétiques et de 4 s dans LuST, comme dans leurs plans de feux.

**Masque d'actions (décision D13).** À chaque décision, un masque indique les phases autorisées.
Il est fourni dans l'observation (`{"observations", "action_mask"}`, format attendu par le module
de masquage de RLlib) et dans `info`. Il est réutilisé pour SB3 (MaskablePPO), QMIX
(*available actions*) et MADDPG (logits masqués).

- **Vert minimal (5 s)** : tant que la règle de SUMO-RL interdit un changement (temps depuis le
  dernier changement < jaune + vert minimal), seule l'action « garder la phase courante » est
  autorisée. Aucune demande n'est donc ignorée silencieusement.
- **Vert maximal (50 s)** : si garder la phase prolongeait le vert au-delà de 50 s avant la
  décision suivante, l'action « garder » est interdite.
- **Phases inexistantes** (*padding*) : toujours interdites.
- **Rouge maximal** : non imposé par défaut (`env.max_red: null`, décision D19). Le **rouge le
  plus long subi par chaque phase** est mesuré et rapporté pour toutes les méthodes. Si on
  l'impose, seules les phases qui dépasseraient la limite restent autorisées. Les agents
  appris (IPPO) l'imposent à **220 s** (`configs/algorithms/ippo.yaml`, décision D40) : sans
  lui, ils ne servent jamais certaines phases (rouge le plus long d'environ 3 700 s). Les plans
  fixes, Webster et actionnés ne dépassent jamais 147 s de rouge : la contrainte ne les
  limiterait pas, la comparaison reste donc équitable. Des phases en retard au même moment sont
  servies l'une après l'autre : une phase peut donc attendre jusqu'à max_red + vert minimal +
  jaune (228 s, 229 s sur LuST), ce que vérifie un test sur les états des feux enregistrés par
  SUMO. La variante
  `max_pressure_maxred` de max-pressure l'impose à 120 s (`baselines.max_pressure.max_red`,
  décision D32). Le rouge d'une phase protégée ne signifie pas toujours une privation : un
  mouvement autorisé en vert permissif (`g`) dans une autre phase continue de passer. Le temps
  d'attente maximal et au 95e centile par véhicule mesure la privation réellement subie.

Une action interdite provoque une erreur au lieu d'être ignorée. Les pas où une contrainte était
active (vert minimal, changement forcé par le vert maximal) sont comptés et rapportés.

**Preuve.** Les tests (`tests/test_env_sim.py`) font enregistrer par SUMO lui-même l'état de
chaque feu à chaque seconde. Ils vérifient ensuite, sur les 5 scénarios et pour trois politiques
(aléatoire, toujours garder, toujours changer), qu'aucun vert ne dure moins de 5 s ni plus de
50 s.

## Contraintes de durée des phases et équité de la comparaison

Toutes les méthodes, contrôleurs appris comme lignes de base (plan fixe, actionné,
max-pressure), respectent **les mêmes contraintes** :

- **vert minimal** : une phase verte ne peut pas être interrompue avant sa durée minimale ;
- **vert maximal** : une phase verte ne peut pas être prolongée au-delà de sa durée maximale,
  pour éviter qu'une direction soit privée de vert indéfiniment ;
- **jaune** : de même durée pour toutes les méthodes, inséré automatiquement à chaque
  changement de phase.

La comparaison est ainsi équitable : aucune méthode ne gagne en enfreignant une contrainte que
les autres respectent. Les plans fixes de référence satisfont déjà ces bornes : leurs verts
durent de 6 à 33 s, et les plans actionnés de 5 à 50 s.

## 4. Récompenses

Le terme de trafic local d'un agent est calculé sur la même approche de 200 m que l'observation
(`env.reward.measure`) :

- `waiting` : moins la variation, sur un pas de décision, du temps d'attente cumulé
  des véhicules présents sur l'approche, divisée par **100 s**. Un véhicule qui quitte l'approche
  emporte son temps d'attente : vider une file est donc récompensé. C'est l'idée de la
  récompense `diff-waiting-time` de SUMO-RL, appliquée à toute l'approche ;
- `queue` (par défaut, décision D40) : moins le nombre de véhicules à l'arrêt sur l'approche,
  rapporté à la capacité d'une voie. Il donne un temps d'attente moyen environ deux fois plus
  faible que `waiting` sur single et corridor (décision D39) ;
- `pressure` : moins la valeur absolue de (véhicules en approche − véhicules en sortie),
  rapportée à la capacité.

**Schémas de coopération** (`env.reward.type`), avec t le terme de trafic et p la pénalité de
changement de phase (**0,5** par défaut, décision D25) :

| Schéma | Récompense de l'agent i |
|---|---|
| `local` | r_i = t_i − p_i |
| `neighborhood` | r_i = t_i + α · moyenne_{j voisin de i} t_j − p_i, avec α = **0,5** |
| `global` | r_i = moyenne_j t_j − p_i |

La pénalité de changement reste **individuelle** dans tous les schémas : elle sanctionne la
décision propre de l'agent, et la partager brouillerait l'attribution du mérite (décision D21).
Le schéma global utilise une moyenne plutôt qu'une somme, pour que l'échelle de la récompense ne
dépende pas de la taille du réseau.

**Ordre de grandeur** (grid3x3, demande `medium`, fichier d'évaluation, épisode complet) :

| Politique | Terme de trafic : écart-type | min / max | Pénalité moyenne | Pénalité / écart-type du trafic |
|---|---|---|---|---|
| aléatoire | 8,29 | −37,5 / +40,6 | 0,043 | 0,5 % |
| toujours garder | 3,17 | −15,2 / +17,2 | 0,010 | 0,3 % |
| toujours changer | 3,68 | −12,0 / +17,3 | 0,050 | 1,4 % |

Ces mesures ont été faites avec une pénalité de 0,1. À cette valeur, la pénalité de changement
est **négligeable** devant le terme de trafic. Elle est donc portée à **0,5** (décision D25),
soit de 2 à 7 % de l'écart-type du terme de trafic dans ces mêmes épisodes. Les valeurs 0,1 et
1,0 feront l'objet d'une ablation. Sur un épisode qui se termine réseau vide, la moyenne du terme `waiting` vaut exactement
0 : c'est une différence qui se télescope. Ces deux constats sont soumis à l'équipe avant
l'entraînement (phase 5).

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

## 5. Fin d'épisode

La sémantique suit Gymnasium / PettingZoo, ce qui garantit un calcul correct de la valeur
d'amorçage (*bootstrapping*) :

| Cas | Signal | Pénalité terminale |
|---|---|---|
| Réseau vide après la fin des insertions | `terminated` | aucune |
| Blocage détecté (aucune arrivée pendant 600 s) | `terminated`, `info["gridlock"] = True` | oui |
| Limite de temps (insertions + vidange) | `truncated` | aucune (l'épisode aurait pu continuer) |

**Pénalité de blocage (décision D21).** Mettre fin à l'épisode en bloquant le réseau ne doit
jamais être avantageux : sans pénalité, un blocage interromprait une suite de récompenses
négatives. La pénalité par défaut (`gridlock_penalty: auto`) vaut une borne de la récompense
d'un pas, multipliée par le nombre de pas restants avant la limite de temps. La borne d'un pas
suppose que tous les véhicules qui tiennent sur l'approche attendent pendant tout le pas (× (1 +
α) en schéma de voisinage). Elle dépasse donc toute récompense que les pas restants auraient pu
produire.

Cette borne étant très grande (environ 12 800 pour `single` en début d'épisode, contre un
écart-type de 2 à 10 par pas), la pénalité est **plafonnée** à 100 fois l'écart-type par pas du
terme de trafic. Cet écart-type est mesuré pour chaque scénario et chaque type de récompense, sous
politique aléatoire. Par exemple, le plafond vaut 828,5 pour `grid3x3` avec la récompense
`waiting` (décision D24). Chaque fin d'épisode par blocage est journalisée et comptée.

Chaque épisode d'entraînement tire au sort un fichier de routes d'entraînement et une graine
SUMO, à partir de la graine de l'expérience. L'évaluation parcourt les fichiers d'évaluation
dans un ordre fixe.
