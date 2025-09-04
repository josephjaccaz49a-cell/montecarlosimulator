# Monte Carlo DCA multi-actifs (30 ans) — Jojo
# - DCA hebdo, inflation, dividendes réinvestis, corrélations entre actifs
# - Crises aléatoires PAR SIMULATION (peuvent ne pas avoir lieu) avec durée tirée
# - Zone grisée 80 %, baseline Livret A, "matelas" 0 %, résumé console (n_sims + CAGR)
# - Titres de graphiques rappelant le DCA hebdo et l’horizon

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ========= INTERFACE TEXTE =========

print("=====================================")
print(" Bienvenue dans le Simulateur Monte Carlo de Jojo 🚀")
print("=====================================")

# Nombre de simulations Monte Carlo
try:
    n_sims = int(input("👉 Combien de simulations veux-tu effectuer ? (defaut = 20000) : ") or "20000")
except ValueError:
    n_sims = 20000

# Horizon d’investissement (années)
try:
    years = int(input("👉 Sur combien d'années veux-tu simuler ? (defaut = 35) : ") or "35")
except ValueError:
    years = 35

# Montant du DCA hebdomadaire
try:
    weekly_contribution = float(input("👉 Quel montant veux-tu investir chaque semaine (€) ? (defaut = 100) : ") or "100")
except ValueError:
    weekly_contribution = 100.0

# Capital initial
try:
    start_value = float(input("👉 Quel est ton capital de départ (€) ? (defaut = 0) : ") or "0")
except ValueError:
    start_value = 0

# Inflation par défaut (sera écrasée si un scénario en impose une)
try:
    inflation_annual = float(input("👉 Inflation moyenne attendue (en %, defaut = 2) : ") or "2") / 100
except ValueError:
    inflation_annual = 0.02

# Paramètres supplémentaires
seed = None   # mets un entier si tu veux un résultat reproductible
index_contrib_to_inflation = False  # True = contributions indexées à l’inflation
show_sample_paths = True   # superposer quelques trajectoires individuelles
n_sample_paths = 12        # combien de trajectoires afficher


print("\n--- Résumé de tes choix ---")
print(f" Simulations Monte Carlo : {n_sims:,}")
print(f" Horizon : {years} ans")
print(f" DCA hebdo : +{weekly_contribution:.0f} €/semaine")
print(f" Capital initial : {start_value:,.2f} €")
print(f" Inflation : {inflation_annual*100:.1f}%/an")
print("=====================================\n")


# ========= SÉLECTEUR DE SCÉNARIO (crises uniquement) =========

import sys

SCENARIOS = {
    "1": {  # Doux
        "label": "Doux (crises rares et courtes)",
        "p_crisis": 0.25,
        "crisis_mu_shift": -0.04,      # choc modéré
        "crisis_sigma_multiplier": 1.3,# vol un peu ↑
        "short_share": 0.70, "short_range": (0.5, 2.0),
        "mid_share":   0.25, "mid_range":   (2.0, 4.0),
        "long_share":  0.05, "long_range":  (4.0, 6.0),
        "inflation_annual": None,      # None = ne pas toucher à ta valeur
    },
    "2": {  # Central (par défaut)
        "label": "Central (réaliste)",
        "p_crisis": 0.45,
        "crisis_mu_shift": -0.06,
        "crisis_sigma_multiplier": 1.5,
        "short_share": 0.60, "short_range": (0.5, 2.0),
        "mid_share":   0.30, "mid_range":   (2.0, 5.0),
        "long_share":  0.10, "long_range":  (5.0,10.0),
        "inflation_annual": None,
    },
    "3": {  # Stress
        "label": "Stress (crises fréquentes, longues et fortes)",
        "p_crisis": 0.60,
        "crisis_mu_shift": -0.10,
        "crisis_sigma_multiplier": 1.7,
        "short_share": 0.40, "short_range": (0.5, 2.0),
        "mid_share":   0.35, "mid_range":   (2.0, 6.0),
        "long_share":  0.25, "long_range":  (6.0,12.0),
        "inflation_annual": 0.03,  # ex: inflation plus haute en stress
    },
}

def choose_scenario(default_key="2"):
    """Choix interactif si possible, sinon défaut 'central'."""
    if not sys.stdin.isatty():
        print(f"(Pas d'entrée interactive détectée) → scénario par défaut: {default_key} - {SCENARIOS[default_key]['label']}")
        return SCENARIOS[default_key]
    try:
        print("\nChoisis un scénario de crise :")
        for k, v in SCENARIOS.items():
            print(f" {k} - {v['label']}")
        c = input("Votre choix [1/2/3] (Enter = 2) : ").strip() or default_key
        if c not in SCENARIOS:
            print(f"Choix invalide → scénario {default_key}")
            c = default_key
        return SCENARIOS[c]
    except (EOFError, KeyboardInterrupt):
        print(f"\nEntrée interrompue → scénario {default_key}")
        return SCENARIOS[default_key]

# -- Appliquer le scénario (ne touche PAS au portfolio)
_scn = choose_scenario()

enable_crisis = True  # on laisse activé, c'est le but du sélecteur
p_crisis = _scn["p_crisis"]
crisis_mu_shift = _scn["crisis_mu_shift"]
crisis_sigma_multiplier = _scn["crisis_sigma_multiplier"]

# Durées (mélange discret) — utilisées par ta fonction sample_crisis_years
short_share = _scn["short_share"]; short_range = _scn["short_range"]
mid_share   = _scn["mid_share"];   mid_range   = _scn["mid_range"]
long_share  = _scn["long_share"];  long_range  = _scn["long_range"]

# Inflation (optionnel) — si None, on garde ta valeur existante
if _scn["inflation_annual"] is not None:
    inflation_annual = _scn["inflation_annual"]

print(f"\nScénario sélectionné : {_scn['label']}")
print(f"  p_crisis={p_crisis:.0%} | choc={crisis_mu_shift:+.1%}/an | vol×={crisis_sigma_multiplier}")
print(f"  Durées ~ court {short_share*100:.0f}% {short_range} ans | moyen {mid_share*100:.0f}% {mid_range} ans | long {long_share*100:.0f}% {long_range} ans")
if _scn['inflation_annual'] is not None:
    print(f"  Inflation fixée à {inflation_annual*100:.1f}%/an (par scénario)")


# ========= PORTFEUILLE MULTI-ACTIFS =========
# Édite/complète selon ton portefeuille (les weights doivent ~somme à 1.0)
# mu = rendement annualisé ESPÉRÉ (total return hors dividend_yield ci-dessous) en nominal
# sigma = volatilité annualisée
# dividend_yield = rendement dividendes annualisé (réinvesti), séparé de mu pour jouer facilement
# beta_crisis = sensibilité à la crise (1 = pleine, 0.5 = moitié, 0 = insensible)
portfolio = [
    {"name": "MSCI World",           "weight": 0.55, "mu": 0.065, "sigma": 0.15, "dividend_yield": 0.018, "beta_crisis": 1.0},
    {"name": "MSCI Emerging Markets","weight": 0.15, "mu": 0.075, "sigma": 0.20, "dividend_yield": 0.022, "beta_crisis": 1.0},
    {"name": "Obligations EUR (court terme)", "weight": 0.20, "mu": 0.020, "sigma": 0.05, "dividend_yield": 0.000, "beta_crisis": 0.3},
    {"name": "Europe Quality Dividend", "weight": 0.10, "mu": 0.055, "sigma": 0.14, "dividend_yield": 0.030, "beta_crisis": 1.0},
]



# NB: adapte poids & paramètres à ta sauce. Si la somme != 1, on renormalise.

# --- Résumé du portefeuille ---
print("\n=== PORTFOLIO ACTUEL ===")
total_weight = sum([a["weight"] for a in portfolio])
for asset in portfolio:
    pct = asset["weight"] / total_weight * 100
    print(f" - {asset['name']:<30} {pct:5.1f}%  |  μ={asset['mu']*100:4.1f}%  σ={asset['sigma']*100:4.1f}%  Div={asset['dividend_yield']*100:4.1f}%")
print("=====================================\n")


# -------- Corrélations (optionnel) --------
# Matrice de corrélation entre actifs (NxN). Si None: identité (indépendants).
# Si tu veux quelque chose de réaliste et simple, mets un rho global (ex: 0.2) pour les actions entre elles
# et ~0.0 avec les obligations courtes.
use_custom_corr = False
rho_global_equity = 0.25  # pour remplir automatiquement si use_custom_corr=True
rho_bond_with_equity = 0.05

# ========= BASELINES COMPARATIVES =========
livret_rate = 0.017   # 1,7%/an, sans plafond
# ==========================================

# --------------- Préparation ---------------
if seed is not None:
    np.random.seed(seed)

weeks = int(52 * years)
dt = 1/52.0

assets = pd.DataFrame(portfolio)
if not np.isclose(assets["weight"].sum(), 1.0):
    assets["weight"] = assets["weight"] / assets["weight"].sum()

n_assets = len(assets)

def annual_to_week_sigma(sigma_annual): return sigma_annual * np.sqrt(dt)
def annual_to_week_infl(x):             return (1 + x)**dt - 1

sigma_week_vec = annual_to_week_sigma(assets["sigma"].values)                   # (N,)
mu_c_week_vec  = (assets["mu"].values - 0.5 * (assets["sigma"].values**2)) * dt # (N,)
div_week_vec   = annual_to_week_infl(assets["dividend_yield"].values)           # approx (N,)
beta_crisis    = assets["beta_crisis"].values                                   # (N,)

infl_w = annual_to_week_infl(inflation_annual)

# -------- Matrice de corrélation --------
if use_custom_corr:
    R = np.eye(n_assets)
    # heuristique: tout ce qui n’est PAS ERNX (oblig court terme) est "equity"
    is_bond = assets["name"].str.contains("Bond", case=False) | assets["name"].str.contains("Oblig", case=False)
    for i in range(n_assets):
        for j in range(n_assets):
            if i == j: continue
            if (not is_bond.iloc[i]) and (not is_bond.iloc[j]):
                R[i, j] = rho_global_equity
            elif (is_bond.iloc[i]) != (is_bond.iloc[j]):
                R[i, j] = rho_bond_with_equity
            else:
                R[i, j] = 0.10  # corr modeste bond/bond
else:
    R = np.eye(n_assets)  # indépendants par défaut

# Cholesky pour générer des chocs corrélés
# (si R n’est pas semi-définie positive à cause d’approx, on force légèrement la diagonale)
try:
    L = np.linalg.cholesky(R)
except np.linalg.LinAlgError:
    eps = 1e-6
    L = np.linalg.cholesky(R + eps * np.eye(n_assets))

# -------- Crises par simulation --------
def sample_crisis_years(size):
    u = np.random.rand(size)
    yrs = np.empty(size, dtype=float)
    mask_short = (u < short_share)
    yrs[mask_short] = np.random.uniform(*short_range, mask_short.sum())
    mask_mid = (u >= short_share) & (u < short_share + mid_share)
    yrs[mask_mid] = np.random.uniform(*mid_range, mask_mid.sum())
    mask_long = (u >= short_share + mid_share)
    yrs[mask_long] = np.random.uniform(*long_range, mask_long.sum())
    return yrs

has_crisis = (np.random.rand(n_sims) < p_crisis) if enable_crisis else np.zeros(n_sims, dtype=bool)
if enable_crisis:
    crisis_years_arr = sample_crisis_years(n_sims)
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

# -------- Contributions hebdo --------
contrib_by_week = weekly_contribution * assets["weight"].values  # (N,)
if index_contrib_to_inflation:
    infl_growth = (1 + infl_w) ** np.arange(1, weeks + 1)        # (weeks,)
    # On appliquera infl_growth[t-1] * contrib_by_week à la semaine t

# -------- Init valeurs --------
values = np.tile((start_value * assets["weight"].values), (n_sims, 1))  # (n_sims, N)
path_total_nominal = np.zeros((weeks + 1, n_sims), dtype=np.float64)
path_total_nominal[0, :] = values.sum(axis=1)

# -------- Simulation hebdo --------
for t in range(1, weeks + 1):
    # Chocs corrélés ~ N(0, R)
    U = np.random.randn(n_sims, n_assets)
    Z = U @ L.T  # (n_sims, N)

    # >>> matrices par simulation (n_sims, N) pour matcher le masque in_crisis
    mu_c    = np.tile(mu_c_week_vec,  (n_sims, 1))   # (n_sims, N)
    sigma_w = np.tile(sigma_week_vec, (n_sims, 1))   # (n_sims, N)

    # Ajustements crise pour les simulations concernées
    if enable_crisis:
        in_crisis = (has_crisis &
                     (crisis_start <= (t - 1)) &
                     ((t - 1) < crisis_end))        # (n_sims,)
        if in_crisis.any():
            # shift de drift (weekly) et vol multipliée selon la sensibilité par actif (beta_crisis)
            mu_shift_week = (crisis_mu_shift * beta_crisis) * dt                  # (N,)
            sigma_mult    = 1 + (crisis_sigma_multiplier - 1) * beta_crisis       # (N,)

            mu_c[in_crisis, :]     = mu_c_week_vec + mu_shift_week               # (N,)
            sigma_w[in_crisis, :]  = annual_to_week_sigma(
                assets["sigma"].values * sigma_mult
            )                                                                    # (N,)

    # Multiplicateur hebdo: prix lognormal * dividendes réinvestis
    weekly_mult = np.exp(mu_c + sigma_w * Z) * (1 + div_week_vec[None, :])       # (n_sims, N)

    # Ajout des contributions (indexées ou non) puis évolution
    if index_contrib_to_inflation:
        add = (infl_growth[t-1] * contrib_by_week)[None, :]                       # (1, N)
    else:
        add = contrib_by_week[None, :]                                            # (1, N)

    values = (values + add) * weekly_mult                                         # (n_sims, N)
    path_total_nominal[t, :] = values.sum(axis=1)

# -------- Réel (déflateur inflation global) --------
deflator = (1 + annual_to_week_infl(inflation_annual)) ** np.arange(0, weeks + 1)
path_total_real = path_total_nominal / deflator[:, None]

# -------- Dates --------
start_date = datetime.today()
dates = pd.to_datetime([start_date + timedelta(weeks=i) for i in range(weeks + 1)])

# -------- Bandes (10/50/90) --------
def bands(arr2d, idx_dates):
    df = pd.DataFrame(arr2d, index=idx_dates)
    return df.quantile(0.10, axis=1), df.quantile(0.50, axis=1), df.quantile(0.90, axis=1)

q10_nom, q50_nom, q90_nom = bands(path_total_nominal, dates)
q10_real, q50_real, q90_real = bands(path_total_real, dates)

# -------- Baselines: Livret A + Matelas --------
# Livret A (nominal)
r_week_livret = (1 + livret_rate)**(1/52) - 1
livret_path = np.zeros(weeks + 1)
livret_path[0] = start_value
for t in range(1, weeks + 1):
    c = weekly_contribution * ((1 + infl_w)**(t-1)) if index_contrib_to_inflation else weekly_contribution
    livret_path[t] = livret_path[t-1] * (1 + r_week_livret) + c
livret_real = livret_path / deflator

# Matelas (0%)
matelas_path = np.zeros(weeks + 1)
matelas_path[0] = start_value
for t in range(1, weeks + 1):
    c = weekly_contribution * ((1 + infl_w)**(t-1)) if index_contrib_to_inflation else weekly_contribution
    matelas_path[t] = matelas_path[t-1] + c
matelas_real = matelas_path / deflator

# -------- Résumé console --------
finals_nom = pd.Series(path_total_nominal[-1, :])
finals_real = pd.Series(path_total_real[-1, :])
prop_with_crisis = has_crisis.mean() if enable_crisis else 0.0

# CAGR basé sur la MEDIANE
if start_value > 0:
    cagr_nom = (finals_nom.median() / start_value) ** (1/years) - 1
    cagr_real = (finals_real.median() / start_value) ** (1/years) - 1
else:
    cagr_nom = float('nan')
    cagr_real = float('nan')


print("\n=== SYNTHÈSE MONTE CARLO (multi-actifs) ===")
print(f"Horizon : {years} ans  |  {weeks} semaines  |  Simulations : {n_sims:,}")
print(f"DCA hebdo : +{weekly_contribution:.0f} €/semaine  |  Inflation moyenne : {inflation_annual*100:.1f}%/an")
if enable_crisis:
    print(f"Proportion de simulations avec crise : {prop_with_crisis:.1%}  (p_cible={p_crisis:.0%})")
else:
    print("Crises : désactivées")

print("\n--- Réel (pouvoir d’achat d’aujourd’hui) ---")
print(f" Médiane : {finals_real.median():,.0f} €   |  P10 : {np.percentile(finals_real,10):,.0f} €   |  P90 : {np.percentile(finals_real,90):,.0f} €")
if not np.isnan(cagr_real):
    print(f" CAGR médian (réel) : {cagr_real*100:.2f}%/an")
else:
    print(" CAGR médian (réel) : N/A (capital initial = 0)")


print("\n--- Nominal (valeur affichée sur le compte) ---")
print(f" Médiane : {finals_nom.median():,.0f} €   |  P10 : {np.percentile(finals_nom,10):,.0f} €   |  P90 : {np.percentile(finals_nom,90):,.0f} €")
if not np.isnan(cagr_nom):
    print(f" CAGR médian (nominal) : {cagr_nom*100:.2f}%/an")
else:
    print(" CAGR médian (nominal) : N/A (capital initial = 0)")



print("\n--- Comparaisons ---")
print(f" Livret A (réel) : {livret_real[-1]:,.0f} €")
print(f" Sous le matelas (réel) : {matelas_real[-1]:,.0f} €   |   Sous le matelas (nominal) : {matelas_path[-1]:,.0f} €")
print("==============================================================\n")

# ==================== GRAPHIQUES (1 fenêtre, 2 sous-graphes) ====================
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

# --- (1) Nominal ---
ax = axes[0]
ax.fill_between(dates, q10_nom.values, q90_nom.values, alpha=0.20, label="Fourchette probable (80% des cas)")
ax.plot(dates, q50_nom.values, label="Scénario central (médiane, nominal)")
ax.plot(dates, q10_nom.values, linestyle="--", linewidth=1, label="Scénario défavorable (P10)")
ax.plot(dates, q90_nom.values, linestyle="--", linewidth=1, label="Scénario favorable (P90)")
ax.plot(dates, livret_path, color="black", label="Livret A (nominal)")
ax.plot(dates, matelas_path, color="grey", linestyle=":", label="Sous le matelas (0%)")
ax.set_ylabel("EUR (nominal)")
ax.set_title("Évolution nominale")

# --- (2) Réel (corrigé de l'inflation) ---
ax = axes[1]
ax.fill_between(dates, q10_real.values, q90_real.values, alpha=0.20, label="Fourchette probable (80% des cas)")
ax.plot(dates, q50_real.values, label="Scénario central (médiane, réel)")
ax.plot(dates, q10_real.values, linestyle="--", linewidth=1, label="Scénario défavorable (P10)")
ax.plot(dates, q90_real.values, linestyle="--", linewidth=1, label="Scénario favorable (P90)")
ax.plot(dates, livret_real, color="black", label="Livret A (réel)")
ax.plot(dates, matelas_real, color="grey", linestyle=":", label="Sous le matelas (0%, réel)")
ax.set_xlabel("Date")
ax.set_ylabel("EUR constants (pouvoir d’achat d’aujourd’hui)")
ax.set_title("Évolution corrigée de l’inflation")

# --- Petites fonctions utilitaires pour tracer X trajectoires sans pourrir la légende ---
def plot_sample_paths(ax, data_2d, dates, n_show, label_once):
    # data_2d: shape = (T, n_sims)
    n_total = data_2d.shape[1]
    n_show = min(n_show, n_total)
    idx = np.random.choice(n_total, size=n_show, replace=False)
    first = True
    for k in idx:
        # label seulement une fois pour éviter une légende énorme
        ax.plot(dates, data_2d[:, k], linewidth=0.8, alpha=0.35,
                label=(label_once if first else None))
        first = False

# --- (1) Nominal : tracer quelques trajectoires individuelles ---
if show_sample_paths:
    plot_sample_paths(axes[0], path_total_nominal, dates, n_sample_paths,
                      label_once="Trajectoires individuelles (échantillon)")
# --- (2) Réel : idem ---
if show_sample_paths:
    plot_sample_paths(axes[1], path_total_real, dates, n_sample_paths,
                      label_once="Trajectoires individuelles (échantillon)")


# Légendes propres
for ax in axes:
    ax.legend(loc="best", title="Lecture :")

# Titre global rappelant le DCA / horizon / scénario
fig.suptitle(
    f"Monte Carlo multi-actifs — DCA {weekly_contribution:.0f} €/sem | Horizon {years} ans | Scénario: {_scn['label']}",
    y=0.97
)

fig.tight_layout()
plt.show()

# Empêche la console de se fermer tout de suite (utile en double-clic)

input("\nAppuie sur Entrée pour quitter le simulateur, merci de l'avoir utilisé, en éspérant que cela ait été utile !")

