# Facteurs linéaires pour la prédiction des rendements d'actions

Modèle soumis au challenge [ENS Challenge Data n° 72](https://challengedata.ens.fr/challenges/72/)
proposé par Qube Research & Technologies (QRT, 2022) : apprendre un modèle à facteurs linéaires qui
prédit le rendement du lendemain de chaque action à partir de ses 250 derniers rendements quotidiens.

Score public : **0,0855**, contre 0,0354 pour le benchmark officiel à facteurs aléatoires et
0,0823 pour un ridge non pondéré.

## Problème

### Données

On observe les rendements quotidiens $r_{i,t}$ de 50 actions sur 754 dates. Pour chaque action $i$
et chaque date $t \in \{250, \dots, 753\}$ (soit $T = 504$ dates), on note

$$x_{i,t} = (r_{i,t-1}, r_{i,t-2}, \dots, r_{i,t-250}) \in \mathbb{R}^{250}$$

le vecteur des 250 rendements passés et $y_{i,t} = r_{i,t}$ la cible. Le jeu de test caché contient
50 autres actions sur la même période : le modèle doit généraliser d'une action à l'autre, pas
d'une période à l'autre.

### Modèle imposé

La prédiction est un modèle à $F = 10$ facteurs linéaires :

$$\hat y_{i,t} = \beta^\top A^\top x_{i,t}, \qquad A \in \mathbb{R}^{250 \times 10}, \quad \beta \in \mathbb{R}^{10},$$

où la matrice $A$ doit appartenir à la variété de Stiefel :

$$A^\top A = I_{10}.$$

Une soumission qui ne respecte pas cette contrainte (à $10^{-6}$ près sur chaque coefficient)
obtient le score $-1$.

### Métrique

Pour chaque date, on compare le vecteur des prédictions $\hat y_t = (\hat y_{i,t})_i$ au vecteur des
rendements réalisés $y_t = (y_{i,t})_i$ par similarité cosinus, puis on moyenne sur les dates :

$$\mathrm{score} = \frac{1}{T} \sum_{t} \frac{\langle \hat y_t, y_t \rangle}{\lVert \hat y_t \rVert \, \lVert y_t \rVert}.$$

La métrique ne dépend que de la direction des prédictions à chaque date, pas de leur échelle.

## Réduction du problème

La prédiction ne dépend de $(A, \beta)$ qu'à travers le vecteur

$$w = A\beta \in \mathbb{R}^{250}, \qquad \hat y_{i,t} = w^\top x_{i,t}.$$

Réciproquement, tout $w \neq 0$ s'écrit $A\beta$ avec $A$ sur la variété de Stiefel : on complète
$w / \lVert w \rVert$ par 9 vecteurs orthonormés (décomposition QR de $[\,w, Z\,]$, où $Z$ est une
matrice gaussienne $250 \times 9$ tirée avec une graine fixe), puis on pose $\beta = A^\top w$.
Comme $w$ appartient à l'image de $A$, on a bien $A\beta = w$.

La contrainte de Stiefel ne restreint donc pas l'ensemble des prédicteurs accessibles : le problème
se ramène à l'estimation d'un filtre linéaire $w$ de 250 coefficients, commun à toutes les actions.
Les 9 colonnes complémentaires de $A$ n'ont aucun effet sur la prédiction.

## Modèle

Le modèle retenu est la somme de deux estimateurs normalisés de $w$.

### 1. Ridge pondéré par date

$$\hat w_R = \arg\min_w \sum_t \omega_t \sum_i \left(y_{i,t} - w^\top x_{i,t}\right)^2 + \lambda \lVert w \rVert^2,
\qquad \omega_t \propto \frac{1}{\lVert y_t \rVert},$$

avec des poids $\omega_t$ normalisés à une moyenne de 1 et $\lambda = 10^4$. La solution est
explicite :

$$\hat w_R = \left(X^\top \Omega X + \lambda I\right)^{-1} X^\top \Omega y.$$

La pondération corrige une incohérence entre les moindres carrés et la métrique. La métrique est un
cosinus par date, donc insensible à l'amplitude des rendements de la date. Sans pondération,
quelques séances très volatiles dominent la somme des carrés. Avec $\omega_t \propto 1/\lVert y_t \rVert$,
chaque date contribue à l'estimation en proportion de son amplitude plutôt que de son carré.

Le paramètre $\lambda$ a peu d'effet dans la plage $[1, 10^4]$ : avec 25 200 observations pour
250 coefficients, le problème est bien conditionné.

### 2. Maximisation directe du cosinus avec lissage

$$\hat w_C = \arg\max_{\lVert u \rVert = 1} \;
\frac{1}{T} \sum_t \frac{\langle X_t u, y_t \rangle}{\lVert X_t u \rVert \, \lVert y_t \rVert}
\;-\; \lambda_s \, d \sum_{k=1}^{d-1} (u_{k+1} - u_k)^2,$$

où $X_t$ est la matrice des $x_{i,t}$ à la date $t$, $d = 250$ et $\lambda_s = 3 \cdot 10^{-5}$.

Le premier terme est exactement la métrique du challenge sur l'échantillon d'entraînement. Le
second pénalise les variations du filtre d'un retard au suivant et limite le surapprentissage. La
normalisation $\lVert u \rVert = 1$ est nécessaire : le cosinus étant invariant par changement
d'échelle de $u$, une pénalité appliquée à un vecteur libre pourrait être annulée en faisant tendre
sa norme vers 0. Pour la même raison, une pénalité $L^2$ serait ici constante et sans effet.

L'optimisation porte sur $v \in \mathbb{R}^{250}$ avec $u = v / \lVert v \rVert$, par L-BFGS avec
gradient analytique, à partir de $\hat w_R / \lVert \hat w_R \rVert$. Sans pénalité, des
initialisations aléatoires convergent toutes vers la même solution, de score in-sample 0,1491 :
c'est le maximum de la métrique d'entraînement sur l'ensemble des filtres linéaires.

### 3. Mélange

$$w = \frac{\hat w_R}{\lVert \hat w_R \rVert} + \hat w_C.$$

Les deux composantes ont le même poids ; ce poids n'est pas ajusté sur les données. Les deux filtres
sont proches (cosinus 0,94 entre eux) mais pas identiques. Leur moyenne réduit la variance
d'estimation : c'est le seul changement testé qui améliore à la fois le score in-sample et le score
en validation croisée de façon significative.

La soumission est obtenue en écrivant $w = A\beta$ selon la construction décrite plus haut.

## Validation

Le test porte sur d'autres actions aux mêmes dates. La validation croisée découpe donc les
**actions**, jamais les dates : 5 plis de 10 actions, répétés sur 5 découpages aléatoires
(graines 42 à 46), soit 25 plis. Tout ce qui est appris des données (poids des dates, solution
du ridge, optimisation du cosinus) est recalculé sur les seules actions d'entraînement de chaque
pli.

Les modèles sont comparés pli par pli sur les mêmes découpages (gain apparié). Les 25 plis
réutilisent les mêmes 50 actions et ne sont pas indépendants : l'erreur type divise l'écart-type
des plis par $\sqrt{5}$ et non par $\sqrt{25}$. Un gain n'est considéré comme réel que s'il dépasse
deux erreurs types et ne dégrade pas le score in-sample.

| Modèle | In-sample | Validation croisée | Gain apparié | Public |
|---|---|---|---|---|
| Ridge non pondéré ($\lambda = 10^4$) | 0,1298 | 0,0445 | -0,0075 ± 0,0028 | 0,0823 |
| Ridge pondéré par date | 0,1407 | 0,0521 | référence | 0,0851 |
| Cosinus lissé seul | 0,1478 | 0,0572 | +0,0051 ± 0,0026 | 0,0834 |
| **Mélange (modèle retenu)** | **0,1462** | **0,0557** | **+0,0037 ± 0,0013** | **0,0855** |

Le mélange est le seul modèle dont le gain dépasse deux erreurs types (2,8). Avec 50 actions de
test, les écarts publics inférieurs à environ 0,003 ne sont pas significatifs ; le choix repose sur
la validation croisée, pas sur le classement public.

## Approches écartées

Les essais suivants ont été évalués avec le même protocole et n'ont pas été retenus. Ce dépôt ne
contient que le modèle final ; le code de ces essais n'y figure pas.

| Approche | Résultat |
|---|---|
| Bases fixes de 10 facteurs (blocs de retards, DCT, exponentielles) | validation croisée 0,015 à 0,023 |
| Analyse en composantes principales (10 facteurs) | public 0,0380 |
| Modèle autorégressif AR(10) | public 0,0128 |
| Écrêtage des rendements à 2 écarts-types et perte de Huber | meilleure validation croisée (0,066), mais in-sample 0,119 et public 0,0724 |
| Maximisation du cosinus sans lissage | gain non significatif, public 0,0838 |
| Pénalité $L^1$ sur le filtre | aucun effet, le filtre optimal n'est pas creux |
| Bagging sur les actions ou sur des blocs de dates | gain négatif |
| Actions synthétiques (combinaisons de paires corrélées) | équivalent à une repondération, sans gain |
| Repondération des actions selon leur groupe de corrélation | équivalent à une pondération par l'inverse de la volatilité, in-sample dégradé |
| Filtre estimé sur la moyenne transversale (timing de marché) | la composante de marché ne porte que 5 % de la variance des cibles, gain négatif |

Deux constats limitent les gains possibles. Le score in-sample est borné par 0,1491, maximum de la
métrique sur tous les filtres linéaires, et le ridge pondéré en atteint déjà 95 %. Par ailleurs,
la performance hors échantillon provient presque entièrement des actions de test qui ont une
action fortement corrélée dans l'échantillon d'entraînement.

## Structure du dépôt

```
data/                     données d'entraînement du challenge (source : challengedata.ens.fr)
qrt_model/
    data.py               chargement des données et matrice des retards
    metrics.py            métrique du challenge, contrôle de la contrainte de Stiefel
    model.py              ridge pondéré, cosinus lissé, mélange, construction de (A, beta)
    validation.py         validation croisée entre actions
    submission.py         écriture et relecture des fichiers de soumission
notebooks/modele.ipynb    évaluation : scores in-sample, validation croisée, filtres estimés
scripts/make_submission.py
soumission/
    soumission_modele4.csv            soumission retenue (score public 0,0855)
    soumission_example/               format attendu par la plateforme
tests/                    tests unitaires
```

## Utilisation

Python 3.11 ou plus récent.

```
pip install -r requirements.txt
python -m pytest -q tests
python scripts/make_submission.py soumission/nouvelle_soumission.csv
```

Le script entraîne le modèle sur les 50 actions, vérifie la contrainte de Stiefel et écrit un
fichier de 2 510 lignes : les 2 500 coefficients de $A$ en ordre ligne par ligne, puis les 10
coefficients de $\beta$. Il refuse d'écraser un fichier existant. Les tests vérifient notamment
que le modèle reproduit à l'identique le fichier `soumission/soumission_modele4.csv`.
