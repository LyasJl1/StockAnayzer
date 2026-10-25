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

## Research Lab

`🔬 Research Lab` est un outil **expérimental d'analyse historique**, séparé du
Timing Lab. Il teste des facteurs techniques prédéfinis, sans optimisation de
seuils, apprentissage, nouvelle stratégie, nouveau score Timing ni V4. Ses
résultats ne modifient ni l'Analyse, ni le Scoring V2, ni le Screener, ni la
Watchlist. Ils ne garantissent aucune performance future.

Configuration : univers initial AAPL, MSFT, GOOGL, NVDA, JPM, XOM, KO, SPY,
ou jusqu'à 20 tickers distincts ; périodes de 3, 5, 8 ou 10 ans (10 par défaut),
horizons de 5, 20 ou 60 **séances** (20 par défaut). Le téléchargement utilise
des dates calculées, y compris pour 8 ans, avec 550 jours calendaires de
warm-up supplémentaires. Les observations exigent au moins 252 séances
antérieures/disponibles. SPY est chargé comme contexte même lorsqu'il ne fait
pas partie de l'univers personnalisé. Les cours OHLC sont ajustés par Yahoo
Finance. Les symboles indisponibles et les séances sans contexte SPY sont
signalés ; l'absence de SPY empêche le diagnostic de régime.

### Facteurs et bornes fixes

Les constantes `RESEARCH_BUCKET_EDGES` et `RESEARCH_BUCKET_LABELS` de
`research_lab.py` définissent les buckets **avant** tout test. Les intervalles
sont fermés à gauche, ouverts à droite ; une frontière entre dans le bucket
suivant. Par exemple RSI 35 appartient à 35–40 et RSI 70 à ≥70. Le dernier
bucket de position 52 semaines inclut 100 %. Les valeurs absentes et infinies
restent exclues, sans imputation.

| Facteur | Bornes intérieures fixes |
| --- | --- |
| Régime de l'action MM200 | Cours > / < MM200 ; MM200 > / < MM200 vingt séances plus tôt, séparément |
| Distance MM50 | −8, −5, −2, +2, +5, +10 % |
| Distance MM200 | −15, −5, 0, +10, +25 % |
| RSI | 30, 35, 40, 45, 50, 60, 70 |
| MACD | > / < signal ; gap croissant / stable ou décroissant sur 5 séances, séparément |
| ATR14 / cours | 1,5, 2,5, 4 % |
| Volatilité annualisée | Écart-type des rendements sur 20 séances × √252 ; bornes 15, 25, 35, 50, 70 % |
| Position 52 semaines | 20, 40, 60, 80 % ; range High/Low des 252 dernières séances |
| Drawdown récent | Cours / maximum des clôtures sur 60 séances − 1 ; bornes −20, −10, −5 % |
| Momentum 20 / 60 / 120 séances | −20, −10, −5, 0, +5, +10, +20 % pour chaque horizon |
| Force relative 60 séances | Rendement de l'action moins rendement SPY ; −10, −5, 0, +5, +10 **points** |

SPY n'est jamais comparé à lui-même pour la force relative. L'alignement entre
marchés utilise la même date de séance, sans backfill. Le régime **Bull**
requiert SPY > MM200 **et** une MM200 supérieure à sa valeur vingt séances
plus tôt ; **Non bull** est le contraire lorsque le contexte est disponible.
Un contexte absent reste inconnu, il n'est pas assimilé à Non bull.

### Cibles, baselines et agrégations

À T, toutes les features utilisent uniquement des données ≤ T. Les cibles sont
construites séparément avec les fonctions historiques existantes :
`Close[T+h]/Close[T] − 1` et excursion minimale des clôtures de T à T+h,
rapportée au cours de T. Le drawdown futur n'est pas le drawdown d'un
portefeuille ; `forward_drawdown_60` reste distinct du facteur rétrospectif
`drawdown_60`. Une observation dont la sortie dépasse la fin de période ou de
validation est exclue pour cet horizon.

L'**alpha** est une différence historique par rapport à une baseline, pas un
alpha économétrique ajusté du risque. La baseline globale est la moyenne des
rendements de toutes les séances admissibles du **même ticker**, au même
horizon et dans la même période d'analyse. La baseline principale est celle du
**même ticker et du même régime SPY**. Dans la vue Tous, chaque observation
utilise la baseline de son propre régime ; la moyenne reflète ainsi le mix de
régimes réellement observé dans le bucket, sans confondre surperformance et
exposition à un marché haussier. Les séances sans régime identifiable ne sont
pas admissibles aux diagnostics. Les baselines des fenêtres sont recalculées
uniquement à l'intérieur de chaque validation.

Le Lab restitue nombre d'observations, moyenne, médiane, taux positif,
drawdown moyen/pire, baselines et alphas global/régime. Il sépare la moyenne
**par observation** de la moyenne **à poids égaux par ticker**. La stabilité
inclut tickers à alpha positif / tickers valides, médiane des alphas par
ticker, meilleur et pire ticker. Les fenêtres utilisent trois années
d'observation, une année de validation et un pas annuel, sans apprentissage.
L'observation sert au découpage ; aucun paramètre n'y est choisi. À trois ans,
aucune validation complète n'existe : c'est affiché comme indéterminable.

Les labels restent descriptifs : « Encourageant » exige au moins 100
observations, alphas moyen par observation, moyen par ticker et médian
positifs, plus de 50 % des tickers et fenêtres positifs, et au moins deux
tickers et deux fenêtres valides. « Faible » indique une moyenne par ticker
non positive ou moins de la moitié des tickers/fenêtres positifs. Les autres
cas sont « Mitigé ». Les échantillons <30 portent « Échantillon très faible » ;
30–100 portent « Échantillon limité ». Un label ne constitue pas une preuve
de robustesse prédictive.

### Exploration et combinaisons

La synthèse trie sur l'alpha régime au même horizon (20 séances par défaut).
Les filtres portent sur facteur, horizon, régime, ticker et minimum
d'observations. Un graphique de buckets avec ligne zéro et des expanders
séparent statistiques, stabilité, comparaison Bull/Non bull, corrélations et
combinaisons. Les corrélations sont les moyennes à poids égaux des Pearson
intra-ticker, avec au moins 30 paires ; |r| ≥0,80 signale une redondance
potentielle sans supprimer de feature.

Les cinq combinaisons sont écrites explicitement et leurs bornes inclusives :

- A : Bull + distance MM50 entre −5 et +2 %.
- B : Bull + RSI entre 35 et 45.
- C : Bull + force relative 60j >0.
- D : Bull + distance MM50 entre −5 et +2 % + force relative >0.
- E : Bull + drawdown récent entre −15 et −5 % + force relative >0.

Il n'y a ni recherche exhaustive ni classement de « meilleures stratégies ».
Les rendements quotidiens futurs 20j/60j se chevauchent : aucune p-value
classique ni interprétation naïve de significativité statistique n'est
présentée. Tester de nombreux buckets reste exposé aux comparaisons multiples,
au biais de survivance de l'univers choisi et à des effets de régime.

L'export `research_lab_results.csv` contient les filtres courants, les deux
agrégations, les baselines, les alphas, les drawdowns, les décomptes de stabilité
et les réserves d'échantillon. Les valeurs numériques sont en ratios : 0,01
vaut 1 % de rendement ou un point d'alpha. Features et targets sont calculées
une seule fois par ticker ; téléchargement, features et diagnostics sont
mis en cache. Aucun indicateur n'est recalculé pour chaque bucket.
