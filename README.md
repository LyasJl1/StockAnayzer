# Stock Analyzer & Trading Assistant

Application web Streamlit d'analyse d'actions. À partir d'un ticker Yahoo Finance
(`AAPL`, `MSFT`, `TTE.PA`, etc.), elle présente :

- les principaux indicateurs fondamentaux disponibles ;
- un score multifactoriel transparent et pondéré selon le mode Investisseur ou Trader / Swing ;
- des arguments favorables et des risques justifiés par les valeurs réellement disponibles ;
- le cours, les MM50/MM200, le RSI, le MACD et la volatilité historique ;
- un diagnostic pédagogique de tendance, de valorisation, de croissance et de risque ;
- le cours et sa moyenne mobile à 200 séances (MM200) sur un graphique interactif ;
- un diagnostic pédagogique de tendance et de fondamentaux ;
- un plan indicatif avec prix d'entrée, Stop-Loss, Take-Profit et ratio
  risque/récompense.

> L'application est fournie à titre informatif et ne constitue pas un conseil
> financier.

## Installation

Python 3.10 ou une version plus récente est recommandé. Depuis la racine du
repository, créez si souhaité un environnement virtuel, puis installez les
dépendances :

```bash
python -m venv .venv
source .venv/bin/activate  # Sous Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Lancement

Toujours depuis la racine du repository :

```bash
streamlit run app.py
```

Streamlit affiche ensuite dans le terminal l'adresse locale à ouvrir dans le
navigateur. Une connexion Internet est nécessaire pour récupérer les données
Yahoo Finance.

## Indice multifactoriel — Scoring Engine V2

Analyse, Comparateur, Screener et Watchlist utilisent `evaluate_scoring()`.
Les règles des métriques et les poids des deux profils sont conservés. Une
métrique absente ou inexploitable reste absente ; une catégorie sans métrique
scorable vaut `None` et s'affiche `N/D`, avec une barre neutre de largeur nulle.

Le score brut est la moyenne des catégories disponibles, pondérée par `WEIGHTS` :
`Σ(score_catégorie × poids) / Σ(poids_disponibles)`. Une catégorie absente
n'apporte ni zéro ni 50. Moins de deux catégories : indice non déterminable.
Deux ou trois catégories : avertissement de couverture limitée.

L'indice affiché est normalisé : `score_brut / borne_théorique_disponible × 100`,
plafonné entre 0 et 100. Le brut et sa borne sont affichés dans les vues et les
exports. Les scores par catégorie restent **bruts**. La borne d'une catégorie
est la moyenne des maxima des seules métriques scorables présentes, puis les
bornes des catégories sont pondérées avec les mêmes poids disponibles.
`SCORING_METRIC_MAXIMA` documente les maxima des règles existantes :

| Catégorie | Maxima des métriques | Borne avec toutes les métriques |
| --- | --- | --- |
| Fondamentaux | Marge 92, ROE 92, FCF 78 | 87,33 |
| Valorisation | P/E 90, P/B 82 | 86 |
| Croissance | CA 92, bénéfices 92 | 92 |
| Technique | MM50 78, MM200 82, RSI 82, MACD 72 | 78,5 |
| Risque | Dette 88, volatilité 88, bêta 82 | 86 |

Avec toutes les métriques, la borne globale brute vaut **86,03** pour
Investisseur et **84,10** pour Trader / Swing. Ces bornes décrivent les règles,
pas une probabilité ni une performance attendue. Les données partielles peuvent
relever une borne : sans le FCF (plafond 78), Fondamentaux peut atteindre 92.
C'est pourquoi les vues passent les métriques disponibles à
`calculate_global_score()`. Un appel avec seulement les catégories et le profil
emploie une enveloppe conservatrice (maximum des maxima par catégorie), et le
champ `normalization` distingue ce cas. Les nouvelles bandes de verdict sur
l'indice normalisé sont : 85+ Très favorable, 70+ Favorable, 55+ Équilibré,
40+ Prudence, puis Défavorable.

Deux couvertures distinctes accompagnent l'indice :

- **Couverture du modèle** : poids des catégories présentes / poids total.
- **Couverture des données** : nombre de métriques exploitables / **14**.
  Ce décompte inclut le MACD avec son signal comme une seule métrique ; un P/E
  ou P/B non positif, une MM nulle ou un MACD sans signal ne comptent pas.

La couverture ne mesure aucune confiance prédictive. Le Screener filtre sur
la couverture des données (50, 60, 70, 80 ou 90 %, défaut 70 %) ; les changements
de filtre réappliquent les critères aux candidats déjà chargés. La Watchlist
conserve les lignes incomplètes et expose leur couverture et leur réserve.
Le Comparateur réserve le verdict global si les couvertures diffèrent d'au
moins **20 points**, si les catégories disponibles diffèrent, ou si un indice
est indéterminable. Ce seuil est une règle de présentation explicite ; aucun
produit « score × couverture » n'est utilisé.

Les moteurs Timing V1/V2/V3, leurs seuils, le Backtest et le Walk-Forward
conservent leurs règles. Leur affichage accepte un indice multifactoriel
indéterminable sans modifier le calcul Timing.

## Tests

Depuis la racine du repository, avec les dépendances installées :

```bash
python -m pip install pytest
python -m pytest -q
```

Les tests du scoring et des vues Streamlit utilisent des données locales,
sans accès Yahoo Finance ni clé Gemini.
