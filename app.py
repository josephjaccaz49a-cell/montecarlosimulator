
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st

st.set_page_config(page_title="Simulateur Monte Carlo de Jojo", layout="wide")

st.title("🚀 Simulateur de Monte Carlo multi-actifs de Jojo")

st.markdown("""
Bienvenue, ici, on utilise la méthode de Monte Carlo qui est une méthode algorithmique via de l'aléatoire : ça veut dire qu’au lieu de prévoir un seul futur, 
on génère des dizaines de milliers de futurs possibles, avec du hasard dans les rendements, l’inflation et les crises.  
On peut ensuite voir la zone la plus probable (80 % des cas), un scénario central, 
et des scénarios défavorables/favorables.  

💡 Cet outil n’a pas été créé pour “optimiser” la richesse individuelle, mais pour montrer, 
de manière simple, comment l’épargne régulière se transforme dans un système économique capitaliste 
où l’argent dort rarement tout seul et est actuellement utilisé par les banques pour générer leurs profits,
via des investissements qui ne respecteraient pas forcément vos critères éthiques.  

👉 Pourquoi ?  
Parce que comprendre les mécanismes financiers, c’est aussi se donner les moyens de les 
**reprendre en main collectivement** :  
- voir comment l’inflation grignote le pouvoir d’achat,  
- mesurer le rôle des crises dans la fragilité du capital,  
- comparer l’investissement actif avec les solutions classiques (Livret A, matelas…).  

Ici, pas de promesse magique : juste une façon de montrer **comment les règles du jeu 
fonctionnent réellement**, afin que chacun puisse se les approprier et réfléchir à ce qu’on 
pourrait en faire, ensemble.  

✊ Bref, un outil pour apprendre, pas pour vendre du rêve.
""")

# ================== UI : Paramètres ==================
colA, colB, colC, colD, colE = st.columns(5)
with colA:
    n_sims = st.number_input("Nb de simulations",
                         min_value=1, max_value=50000,
                         value=10000, step=100)        # tous INT
with colB:
    years  = st.number_input("Horizon (années)",
                         min_value=1, max_value=60,
                         value=35, step=1)              # tous INT
with colC:
    weekly_contribution = st.number_input("DCA hebdo (€)",
                                      min_value=0.0,
                                      value=100.0, step=10.0)   # tous FLOAT
with colD:
    start_value = st.number_input(
    "Capital initial (€)",
    min_value=0.0,
    value=1000.0,   # ← float
    step=100.0      # ← float
)

with colE:
    inflation_pct = st.number_input("Inflation moyenne (%)",
                                min_value=0.0,
                                value=2.0, step=0.1)           # tous FLOAT
inflation_annual = inflation_pct / 100.0

scenario_label = st.selectbox("Scénario de crises", ["Doux (crises rares/courtes)","Central (réaliste)","Stress (fréquentes/longues/fortes)"])
show_sample_paths = st.checkbox("Afficher des trajectoires individuelles", value=True)
n_sample_paths = st.number_input(
    "Nombre de trajectoires à afficher (max 30)",
    min_value=0, max_value=30, value=12, step=1
)
n_sample_paths = int(n_sample_paths)

index_contrib_to_inflation = st.checkbox("Augmenter ton versement chaque année selon l’inflation (pour garder le même pouvoir d’achat)", value=False)


use_custom_corr = st.checkbox(
    "Corrélations réalistes : les actions montent/baissent souvent ensemble, les obligations sont plus stables",
    value=False
)


# ==== Scénarios ====
SCENARIOS = {
    "doux": dict(
        label="🌱 Doux — crises rares (25%), peu intenses (-4%/an), vol×1.3, courtes (0.5–2 ans)",
        p_crisis=0.25, crisis_mu_shift=-0.04, crisis_sigma_multiplier=1.3,
        short_share=0.70, short_range=(0.5,2.0),
        mid_share=0.25,  mid_range=(2.0,4.0),
        long_share=0.05, long_range=(4.0,6.0),
        inflation_annual=None
    ),
    "central": dict(
        label="⚖️ Central — crises modérées (45%), réalistes (-6%/an), vol×1.5, durées 0.5–10 ans",
        p_crisis=0.45, crisis_mu_shift=-0.06, crisis_sigma_multiplier=1.5,
        short_share=0.60, short_range=(0.5,2.0),
        mid_share=0.30,  mid_range=(2.0,5.0),
        long_share=0.10, long_range=(5.0,10.0),
        inflation_annual=None
    ),
    "stress": dict(
        label="🔥 Stress — crises fréquentes (60%), sévères (-10%/an), vol×1.7, longues (0.5–12 ans), inflation 3%",
        p_crisis=0.60, crisis_mu_shift=-0.10, crisis_sigma_multiplier=1.7,
        short_share=0.40, short_range=(0.5,2.0),
        mid_share=0.35,  mid_range=(2.0,6.0),
        long_share=0.25, long_range=(6.0,12.0),
        inflation_annual=0.03
    ),
}

# ---- Selectbox basé sur les labels, mapping sûr vers la clé ----
label_to_key = {v["label"]: k for k, v in SCENARIOS.items()}
scenario_label = st.selectbox(
    "Scénario de crises",
    list(label_to_key.keys()),
    index=1
)
scenario_key = label_to_key[scenario_label]
_scn = SCENARIOS[scenario_key]

# Si le scénario fixe une inflation, on l’applique
if _scn["inflation_annual"] is not None:
    inflation_annual = _scn["inflation_annual"]


# ================== Sélecteur de portefeuille ==================
st.subheader("🧺 Choisis un portefeuille type")

PRESETS = {
    "Standard équilibré": [
        {"name": "MSCI World",                     "weight": 0.55, "mu": 0.065, "sigma": 0.15, "dividend_yield": 0.018, "beta_crisis": 1.0},
        {"name": "MSCI Emerging Markets",          "weight": 0.15, "mu": 0.075, "sigma": 0.20, "dividend_yield": 0.022, "beta_crisis": 1.0},
        {"name": "Obligations EUR (court terme)",  "weight": 0.20, "mu": 0.020, "sigma": 0.05, "dividend_yield": 0.000, "beta_crisis": 0.3},
        {"name": "Europe Quality Dividend",        "weight": 0.10, "mu": 0.055, "sigma": 0.14, "dividend_yield": 0.030, "beta_crisis": 1.0},
    ],

    # Concentré et mal diversifié : quasi tout sur EM, peu d'oblig, volatilité + élevée.
    "Éclaté mais drôle (ultra spicy)": [
    
       {"name": "NASDAQ 100 x3 (levier)",          "weight": 0.25, "mu": 0.10,  "sigma": 0.45, "dividend_yield": 0.000, "beta_crisis": 1.30},
       {"name": "Crypto (BTC+ETH)",                "weight": 0.25, "mu": 0.12,  "sigma": 0.80, "dividend_yield": 0.000, "beta_crisis": 1.50},
       {"name": "Meme Stocks (panier)",            "weight": 0.15, "mu": 0.08,  "sigma": 0.50, "dividend_yield": 0.000, "beta_crisis": 1.40},
       {"name": "Uranium Juniors",                 "weight": 0.10, "mu": 0.09,  "sigma": 0.40, "dividend_yield": 0.000, "beta_crisis": 1.20},
       {"name": "Biotech Microcaps",               "weight": 0.10, "mu": 0.09,  "sigma": 0.55, "dividend_yield": 0.000, "beta_crisis": 1.40},
       {"name": "Or (spot)",                       "weight": 0.10, "mu": 0.03,  "sigma": 0.18, "dividend_yield": 0.000, "beta_crisis": 0.60},
       {"name": "Cash / Monétaire EUR",            "weight": 0.05, "mu": 0.02,  "sigma": 0.01, "dividend_yield": 0.000, "beta_crisis": 0.00},
   ],
    
    # Le plus éthique, Critères ESG fictifs
    "Portefeuille Éthique (ESG/Green)": [
       
        {"name": "MSCI World ESG Screened",        "weight": 0.40, "mu": 0.060, "sigma": 0.15, "dividend_yield": 0.018, "beta_crisis": 1.0},
        {"name": "MSCI Emerging Markets ESG",      "weight": 0.15, "mu": 0.070, "sigma": 0.20, "dividend_yield": 0.020, "beta_crisis": 1.0},
        {"name": "MSCI Europe ESG Dividend",       "weight": 0.15, "mu": 0.055, "sigma": 0.14, "dividend_yield": 0.030, "beta_crisis": 1.0},
        {"name": "Obligations vertes (Green Bonds)","weight": 0.20, "mu": 0.025, "sigma": 0.05, "dividend_yield": 0.000, "beta_crisis": 0.3},
        {"name": "ETF Énergies Renouvelables",    "weight": 0.10, "mu": 0.065, "sigma": 0.22, "dividend_yield": 0.000, "beta_crisis": 1.2},
    ],

    # Portefeuille Vice pas bien
    "Portefeuille Vice (pétrole, tabac, alcool)": [
     
        {"name": "ExxonMobil",        "weight": 0.30, "mu": 0.055, "sigma": 0.22, "dividend_yield": 0.040, "beta_crisis": 1.2},
        {"name": "Chevron",           "weight": 0.20, "mu": 0.055, "sigma": 0.22, "dividend_yield": 0.038, "beta_crisis": 1.2},
        {"name": "Philip Morris",     "weight": 0.20, "mu": 0.045, "sigma": 0.18, "dividend_yield": 0.050, "beta_crisis": 1.0},
        {"name": "British American Tobacco", "weight": 0.10, "mu": 0.045, "sigma": 0.18, "dividend_yield": 0.060, "beta_crisis": 1.0},
        {"name": "Diageo (spiritueux)", "weight": 0.10, "mu": 0.050, "sigma": 0.16, "dividend_yield": 0.030, "beta_crisis": 0.9},
        {"name": "AB InBev (bière)",    "weight": 0.10, "mu": 0.050, "sigma": 0.18, "dividend_yield": 0.025, "beta_crisis": 1.0},
    ],


    # Très offensif : max actions, peu d'oblig, risque élevé.
    "Super agressif": [
        {"name": "MSCI World",                     "weight": 0.55, "mu": 0.070, "sigma": 0.17, "dividend_yield": 0.015, "beta_crisis": 1.0},
        {"name": "MSCI Emerging Markets",          "weight": 0.30, "mu": 0.085, "sigma": 0.26, "dividend_yield": 0.018, "beta_crisis": 1.0},
        {"name": "Obligations EUR (court terme)",  "weight": 0.05, "mu": 0.018, "sigma": 0.05, "dividend_yield": 0.000, "beta_crisis": 0.3},
        {"name": "Europe Quality Dividend",        "weight": 0.10, "mu": 0.055, "sigma": 0.15, "dividend_yield": 0.030, "beta_crisis": 1.0},
    ],
}

preset_name = st.selectbox(
    "Sélection du portefeuille",
    list(PRESETS.keys()),
    index=0,
    help="Choisis un set de poids/hypothèses tout prêts pour tester rapidement."
)

portfolio = PRESETS[preset_name]
assets = pd.DataFrame(portfolio).copy()

# Sécurité : si la somme de poids ≠ 1, on renormalise
if not np.isclose(assets["weight"].sum(), 1.0):
    assets["weight"] = assets["weight"] / assets["weight"].sum()

# ------- Affichage propre du portefeuille -------
df_portfolio = assets.assign(
    Poids=(assets["weight"]*100).round(1).astype(str) + " %",
    Rendement=(assets["mu"]*100).round(2).astype(str) + " %",
    Volatilité=(assets["sigma"]*100).round(2).astype(str) + " %",
    Dividende=(assets["dividend_yield"]*100).round(2).astype(str) + " %"
)[["name", "Poids", "Rendement", "Volatilité", "Dividende"]].rename(columns={"name": "Actif"})

st.dataframe(df_portfolio, width="stretch")
st.caption("💡 Les rendements/volatilités sont des hypothèses pédagogiques. Les poids sont renormalisés si besoin.")


# ================== Fonctions utilitaires ==================
def annual_to_week_sigma(sigma_annual, dt): return sigma_annual * np.sqrt(dt)
def annual_to_week_infl(x, dt):             return (1 + x)**dt - 1

def build_corr(n_assets, assets, use_custom_corr, rho_equity=0.25, rho_bond_with_equity=0.05):
    if not use_custom_corr:
        return np.eye(n_assets)
    R = np.eye(n_assets)
    is_bond = assets["name"].str.contains("Bond|Oblig", case=False)
    for i in range(n_assets):
        for j in range(n_assets):
            if i == j: 
                continue
            if (not is_bond.iloc[i]) and (not is_bond.iloc[j]):
                R[i, j] = rho_equity
            elif is_bond.iloc[i] != is_bond.iloc[j]:
                R[i, j] = rho_bond_with_equity
            else:
                R[i, j] = 0.10  # corr modeste bond/bond
    # Cholesky safe
    try:
        np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        R = R + 1e-6 * np.eye(n_assets)
    return R

def sample_crisis_years(size, short_share, short_range, mid_share, mid_range, long_share, long_range):
    u = np.random.rand(size)
    yrs = np.empty(size, dtype=float)
    m_short = (u < short_share)
    yrs[m_short] = np.random.uniform(*short_range, m_short.sum())
    m_mid = (u >= short_share) & (u < short_share + mid_share)
    yrs[m_mid] = np.random.uniform(*mid_range, m_mid.sum())
    m_long = (u >= short_share + mid_share)
    yrs[m_long] = np.random.uniform(*long_range, m_long.sum())
    return yrs


def run_monte_carlo(n_sims, years, weekly_contribution, start_value, inflation_annual,
                    index_contrib_to_inflation, assets, scenario, use_custom_corr):
    dt = 1/52.0
    weeks = int(52 * years)
    n_assets = len(assets)

    sigma_week_vec = annual_to_week_sigma(assets["sigma"].values, dt)
    mu_c_week_vec  = (assets["mu"].values - 0.5 * (assets["sigma"].values**2)) * dt
    div_week_vec   = annual_to_week_infl(assets["dividend_yield"].values, dt)
    beta_crisis    = assets["beta_crisis"].values
    infl_w         = annual_to_week_infl(inflation_annual, dt)

    R = build_corr(n_assets, assets, use_custom_corr)
    L = np.linalg.cholesky(R if np.allclose(R, R.T) else (R + R.T)/2)

    # Crises
    enable_crisis = True
    p_crisis = scenario["p_crisis"]
    crisis_mu_shift = scenario["crisis_mu_shift"]
    crisis_sigma_multiplier = scenario["crisis_sigma_multiplier"]
    short_share, short_range = scenario["short_share"], scenario["short_range"]
    mid_share,   mid_range   = scenario["mid_share"],   scenario["mid_range"]
    long_share,  long_range  = scenario["long_share"],  scenario["long_range"]

    has_crisis = (np.random.rand(n_sims) < p_crisis) if enable_crisis else np.zeros(n_sims, dtype=bool)
    if enable_crisis:
        crisis_years_arr = sample_crisis_years(n_sims, short_share, short_range, mid_share, mid_range, long_share, long_range)
        crisis_weeks_arr = (crisis_years_arr * 52).astype(int)
    else:
        crisis_weeks_arr = np.zeros(n_sims, dtype=int)

    min_start = int(0.5 * 52)
    crisis_start = np.full(n_sims, -1, dtype=int)
    if enable_crisis:
        max_start = weeks - crisis_weeks_arr
        valid = (max_start > min_start) & has_crisis
        crisis_start[valid] = np.random.randint(low=min_start, high=max_start[valid], size=valid.sum())
    crisis_end = crisis_start + crisis_weeks_arr

    # Contributions
    contrib_by_week = weekly_contribution * assets["weight"].values
    if index_contrib_to_inflation:
        infl_growth = (1 + infl_w) ** np.arange(1, weeks + 1)

    # Init valeurs
    values = np.tile((start_value * assets["weight"].values), (n_sims, 1))
    path_total_nominal = np.zeros((weeks + 1, n_sims), dtype=np.float64)
    path_total_nominal[0, :] = values.sum(axis=1)

    # Simulation
    for t in range(1, weeks + 1):
        Z = np.random.randn(n_sims, n_assets) @ L.T
        mu_c    = np.tile(mu_c_week_vec,  (n_sims, 1))
        sigma_w = np.tile(sigma_week_vec, (n_sims, 1))

        if enable_crisis:
            in_crisis = (has_crisis & (crisis_start <= (t-1)) & ((t-1) < crisis_end))
            if in_crisis.any():
                mu_shift_week = (crisis_mu_shift * beta_crisis) * dt
                sigma_mult    = 1 + (crisis_sigma_multiplier - 1) * beta_crisis
                mu_c[in_crisis, :]    = mu_c_week_vec + mu_shift_week
                sigma_w[in_crisis, :] = annual_to_week_sigma(assets["sigma"].values * sigma_mult, dt)

        weekly_mult = np.exp(mu_c + sigma_w * Z) * (1 + div_week_vec[None, :])

        if index_contrib_to_inflation:
            add = (infl_growth[t-1] * contrib_by_week)[None, :]
        else:
            add = contrib_by_week[None, :]

        values = (values + add) * weekly_mult
        path_total_nominal[t, :] = values.sum(axis=1)

    # Réel
    deflator = (1 + annual_to_week_infl(inflation_annual, dt)) ** np.arange(0, weeks + 1)
    path_total_real = path_total_nominal / deflator[:, None]

    # Dates
    start_date = datetime.today()
    dates = pd.to_datetime([start_date + timedelta(weeks=i) for i in range(weeks + 1)])

    # Stats fin
    finals_nom  = pd.Series(path_total_nominal[-1, :])
    finals_real = pd.Series(path_total_real[-1, :])
    prop_with_crisis = has_crisis.mean() if enable_crisis else 0.0

    # CAGR basé médiane (si start_value > 0)
    if start_value > 0 and years > 0:
        cagr_nom  = (finals_nom.median()  / start_value) ** (1/years) - 1
        cagr_real = (finals_real.median() / start_value) ** (1/years) - 1
    else:
        cagr_nom = cagr_real = np.nan

    return dict(
        dates=dates,
        path_total_nominal=path_total_nominal,
        path_total_real=path_total_real,
        finals_nom=finals_nom,
        finals_real=finals_real,
        prop_with_crisis=prop_with_crisis,
        cagr_nom=cagr_nom,
        cagr_real=cagr_real,
        weeks=weeks
    )

# ================== Lancer la simulation ==================
if st.button("🎬 Lancer la simulation"):
    with st.spinner("Ça turbine fort…"):
        res = run_monte_carlo(n_sims, years, weekly_contribution, start_value, inflation_annual,
                              index_contrib_to_inflation, assets, _scn, use_custom_corr)

    dates = res["dates"]
    pt_nom = res["path_total_nominal"]
    pt_real = res["path_total_real"]

    # Bandes
    def bands(arr2d, idx_dates):
        df = pd.DataFrame(arr2d, index=idx_dates)
        return df.quantile(0.10, axis=1), df.quantile(0.50, axis=1), df.quantile(0.90, axis=1)
    q10_nom, q50_nom, q90_nom = bands(pt_nom, dates)
    q10_real, q50_real, q90_real = bands(pt_real, dates)

    # Baselines Livret A + Matelas
    dt = 1/52.0
    infl_w = (1 + inflation_annual)**dt - 1
    livret_rate = 0.017
    r_week_livret = (1 + livret_rate)**(1/52) - 1

    livret_path = np.zeros(res["weeks"] + 1); livret_path[0] = start_value
    matelas_path = np.zeros(res["weeks"] + 1); matelas_path[0] = start_value
    for t in range(1, res["weeks"] + 1):
        c = weekly_contribution * ((1 + infl_w)**(t-1)) if index_contrib_to_inflation else weekly_contribution
        livret_path[t]  = livret_path[t-1] * (1 + r_week_livret) + c
        matelas_path[t] = matelas_path[t-1] + c
    deflator = (1 + ((1 + inflation_annual)**dt - 1)) ** np.arange(0, res["weeks"] + 1)
    livret_real = livret_path / deflator
    matelas_real = matelas_path / deflator

    # ===== Graphiques (1 fenêtre, 2 sous-graphes) =====
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # (1) Nominal
    ax = axes[0]
    ax.fill_between(dates, q10_nom.values, q90_nom.values, alpha=0.20, label="Fourchette probable (80%)")
    ax.plot(dates, q50_nom.values, label="Scénario central (50/50)")
    ax.plot(dates, q10_nom.values, linestyle="--", linewidth=1, label="Scénario défavorable (90 % de chances d’être au-dessus)")
    ax.plot(dates, q90_nom.values, linestyle="--", linewidth=1, label="Scénario favorable (90 % de chances d’être en dessous)")
    ax.plot(dates, livret_path, color="black", label="Livret A (nominal)")
    ax.plot(dates, matelas_path, color="grey", linestyle=":", label="Matelas (0%)")
    ax.set_ylabel("€ (nominal)")
    ax.set_title("Évolution nominale")

    # (2) Réel
    ax = axes[1]
    ax.fill_between(dates, q10_real.values, q90_real.values, alpha=0.20,
                    label="Fourchette probable (80 % des cas)")
    ax.plot(dates, q50_real.values, label="Scénario central (50/50)")
    ax.plot(dates, q10_real.values, linestyle="--", linewidth=1,
            label="Scénario défavorable (90 % de chances d’être au-dessus)")
    ax.plot(dates, q90_real.values, linestyle="--", linewidth=1,
            label="Scénario favorable (90 % de chances d’être en dessous)")
    ax.plot(dates, livret_real, color="black", label="Livret A (réel)")
    ax.plot(dates, matelas_real, color="grey", linestyle=":", label="Matelas (0 %, réel)")
    ax.set_xlabel("Date"); ax.set_ylabel("€ constants (pouvoir d’achat)")
    ax.set_title("Évolution corrigée de l’inflation")


    # Trajectoires individuelles (mêmes indices pour haut/bas)
    if show_sample_paths and n_sample_paths > 0:
       n_total = pt_nom.shape[1]
       k = min(n_sample_paths, n_total)  # borne par le nb de runs
       idx = np.random.choice(n_total, size=k, replace=False)
       for i, k in enumerate(idx):
           label = "Trajectoires (échantillon)" if i == 0 else None
           axes[0].plot(dates, pt_nom[:, k], linewidth=0.7, alpha=0.35, label=label)
           axes[1].plot(dates, pt_real[:, k], linewidth=0.7, alpha=0.35)

    for ax in axes:
        ax.legend(loc="best", title="Lecture :")
    fig.suptitle(f"Monte Carlo — DCA {weekly_contribution:.0f} €/sem | Horizon {years} ans | Scénario: {scenario_label}", y=0.98)
    fig.tight_layout()

    st.pyplot(fig)

    import io

    # -- Export PNG du graphe --
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    st.download_button(
        label="📥 Télécharger le graphique (PNG)",
        data=buf.getvalue(),
        file_name="simulation_jojo.png",
        mime="image/png",
    )
    
    # -- Export CSV des courbes clés --
    export_df = pd.DataFrame({
        "date": dates,
        "q10_nom": q10_nom.values,
        "q50_nom": q50_nom.values,
        "q90_nom": q90_nom.values,
        "q10_real": q10_real.values,
        "q50_real": q50_real.values,
        "q90_real": q90_real.values,
        "livret_nom": livret_path,
        "matelas_nom": matelas_path,
        "livret_real": livret_real,
        "matelas_real": matelas_real,
    })
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Télécharger les percentiles (CSV)",
        data=csv_bytes,
        file_name="simulation_jojo_percentiles.csv",
        mime="text/csv",
    )


    # ===== Synthèse métriques =====
    finals_nom = res["finals_nom"]; finals_real = res["finals_real"]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Médiane (réel)", f"{finals_real.median():,.0f} €")
        st.metric("Taux de croissance annuel composé médian (réel)", "N/A" if np.isnan(res["cagr_real"]) else f"{res['cagr_real']*100:.2f}%/an")
    with col2:
        st.metric("Médiane (nominal)", f"{finals_nom.median():,.0f} €")
        st.metric("Taux de croissance annuel composé médian (nominal)", "N/A" if np.isnan(res["cagr_nom"]) else f"{res['cagr_nom']*100:.2f}%/an")
    with col3:
        st.metric("Proportion de runs avec crise", f"{res['prop_with_crisis']*100:.1f}%")
        st.metric("Nb de simulations", f"{int(n_sims):,}")

    st.markdown("""
    👉 **Comment lire les résultats :**  
    - Les graphiques montrent deux choses :  
       - en haut : l’évolution en **valeur nominale** (ce que tu verrais sur ton compte en banque)  
       - en bas : l’évolution en **euros constants** (corrigée de l’inflation, donc en pouvoir d’achat).  
    - La **zone grisée** correspond aux **80 % de cas les plus probables** (entre scénario défavorable et favorable).  
    - La **ligne médiane** est le scénario “central” (le plus typique).  
    - Les lignes **noires et grises** servent de comparaison :  
       - Livret A à 1.7 %  
       - Matelas (0 %, juste accumuler le cash sous l’oreiller).  
    - Tu peux aussi voir quelques trajectoires individuelles (fines) qui montrent à quel point les marchés sont imprévisibles.  
    
    💡 **Attention :** Ce n’est pas une prédiction !  
    C’est une **simulation statistique** basée sur des hypothèses de rendement, volatilité et inflation.  
    Le but est pédagogique, pour mieux comprendre la puissance des intérêts composés et l’incertitude des marchés.
    """)

    st.success("✅ Simulation terminée, merci de l'avoir utilisée, j'espère qu'elle vous a été utile. Joseph")
else:
    st.info("Choisis tes paramètres puis clique sur **Lancer la simulation**.")
