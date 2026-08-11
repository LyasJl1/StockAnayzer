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
