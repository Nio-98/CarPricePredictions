[README.md](https://github.com/user-attachments/files/31887024/README.md)
# Predikcija cene polovnih automobila 🚗

Regresioni ML projekat koji procenjuje tržišnu cenu polovnog automobila (`priceUSD`) na
osnovu njegovih karakteristika (marka, model, godina, kilometraža, tip goriva,
zapremina motora, menjač, pogon, segment, boja...).

Projekat je urađen kao zadatak iz modula o regresiji i prati kompletan tok rada:

```
EDA → čišćenje podataka → inženjering karakteristika → pretprocesiranje
→ treniranje modela → poređenje modela → evaluacija → dokumentovanje
```

## Sadržaj

```
car-price-prediction/
├── data/
│   ├── cars.csv                  # originalni skup podataka
│   ├── cars_clean.csv            # podaci nakon čišćenja
│   └── cars_features.csv         # podaci nakon inženjeringa karakteristika
├── notebooks/
│   └── 01_eda.ipynb              # eksplorativna analiza podataka (EDA)
├── src/
│   ├── data_cleaning.py          # čišćenje sirovih podataka
│   ├── feature_engineering.py    # kreiranje novih karakteristika
│   ├── data_preprocessing.py     # preprocessing pipeline (imputacija, skaliranje, OHE)
│   ├── model_training.py         # treniranje i čuvanje finalnog modela
│   ├── model_evaluation.py       # evaluacija finalnog modela
│   └── model_comparison.py       # poređenje više regresionih algoritama
├── models/
│   ├── car_price_model.joblib          # sačuvan finalni model (pipeline)
│   ├── test_data.joblib                # test skup korišćen za evaluaciju
│   └── model_comparison_results.csv    # rezultati poređenja modela
├── reports/
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   └── model_comparison_mae.png
├── requirements.txt
└── README.md
```

## Skup podataka

`cars.csv` sadrži oglase polovnih automobila sa sledećim kolonama: `make`, `model`,
`priceUSD` (ciljna promenljiva), `year`, `condition`, `mileage(kilometers)`,
`fuel_type`, `volume(cm3)`, `color`, `transmission`, `drive_unit`, `segment`.

## Kako pokrenuti projekat

### 1. Instalacija

```bash
git clone https://github.com/<tvoj-nalog>/car-price-prediction.git
cd car-price-prediction
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pokretanje pipeline-a od nule

Skripte se pokreću iz root direktorijuma projekta, tim redom:

```bash
python src/data_cleaning.py          # data/cars.csv -> data/cars_clean.csv
python src/feature_engineering.py    # data/cars_clean.csv -> data/cars_features.csv
python src/model_comparison.py       # trenira i poredi 4 modela, ispisuje metrike
python src/model_training.py         # trenira finalni model i čuva ga u models/
python src/model_evaluation.py       # evaluira finalni model, prikazuje primere
```

### 3. EDA notebook

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 4. Korišćenje sačuvanog modela za novu predikciju

```python
import joblib
import pandas as pd
from src.feature_engineering import add_features

pipeline = joblib.load("models/car_price_model.joblib")

new_car = pd.DataFrame([{
    "make": "volkswagen",
    "model": "golf",
    "year": 2014,
    "condition": "with mileage",
    "mileage(kilometers)": 180000,
    "fuel_type": "diesel",
    "volume(cm3)": 1600,
    "color": "black",
    "transmission": "mechanics",
    "drive_unit": "front-wheel drive",
    "segment": "c",
}])

new_car = add_features(new_car)
predicted_price = pipeline.predict(new_car)
print(f"Predviđena cena: {predicted_price[0]:.0f}$")
```

## Čišćenje podataka (`data_cleaning.py`)

- uklonjeni duplikati redova;
- kolone `priceUSD`, `year`, `mileage(kilometers)`, `volume(cm3)` konvertovane u numerički tip;
- uklonjeni redovi bez ciljne promenljive (`priceUSD`);
- uklonjene nerealne cene (van opsega 100–200,000 USD) i nerealne godine proizvodnje (van 1980–2026);
- uklonjena nerealna kilometraža (van opsega 0–1,000,000 km);
- nevalidne (≤0) vrednosti `volume(cm3)` postavljene na `NaN` da bi ih preprocessing pravilno imputirao,
  umesto da unose šum kao lažne nule;
- tekstualne kolone standardizovane (trim + lowercase), nedostajuće kategorijske vrednosti popunjene sa `"unknown"`.

Rezultat: od originalnih redova ostaje **55,529** validnih zapisa, sa svega 47 nedostajućih
vrednosti u `volume(cm3)` (koje rešava imputer u preprocessing koraku).

## Inženjering karakteristika (`feature_engineering.py`)

| Karakteristika | Opis | Zašto je korisna |
|---|---|---|
| `car_age` | trenutna godina − `year` | starost je direktnije povezana sa amortizacijom vrednosti nego sama godina proizvodnje |
| `mileage_per_year` | `mileage(kilometers)` / (`car_age` + 1) | razlikuje "malo vožen stariji auto" od "mnogo vožen noviji auto" — dve stvari koje `mileage` samo ne hvata |
| `engine_volume_liters` | `volume(cm3)` / 1000 | čitljivija skala za motor (npr. 1.6L umesto 1600cm³) |
| `is_newer_car` | 1 ako je `car_age` < 5 | eksplicitan signal za "skoro nov" segment tržišta koji se drugačije cenovno ponaša |
| `is_high_mileage` | 1 ako je kilometraža > 200,000 km | označava vozila kod kojih kilometraža počinje značajnije da obara cenu |
| `brand_model` | `make` + `_` + `model` | hvata kombinacije marke i modela (npr. da li je baš taj model te marke premium); korišćena samo za analizu, **ne ulazi u model** (vidi napomenu ispod) |

**Napomena:** `brand_model` i `model` imaju veliku kardinalnost (na hiljade jedinstvenih
vrednosti), pa bi njihovo uključivanje kroz `OneHotEncoder` napravilo ogroman broj kolona
i povećalo rizik od overfittinga na malom broju primera po modelu automobila. Zato finalni
model kao ulaz koristi `make` (marku), koja već nosi glavninu te informacije, dok `model`
i `brand_model` ostaju dostupni za buduću analizu (npr. target/frequency encoding).

## Pretprocesiranje (`data_preprocessing.py`)

- **Numeričke kolone** (`year`, `mileage(kilometers)`, `volume(cm3)`, `car_age`,
  `mileage_per_year`, `engine_volume_liters`, `is_newer_car`, `is_high_mileage`):
  `SimpleImputer(strategy="median")` → `StandardScaler()`
- **Kategorijske kolone** (`make`, `condition`, `fuel_type`, `color`, `transmission`,
  `drive_unit`, `segment`): `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")`
- Sve je spojeno u jedan `ColumnTransformer`, ugrađen u `sklearn.Pipeline` zajedno sa
  modelom — čime se izbegava curenje podataka (data leakage) između train i test skupa.

## Treniranje i poređenje modela

Testirana su četiri regresiona algoritma, na **istom** train/test split-u (80/20,
`random_state=42`), radi fer poređenja:

| Model | MAE (USD) | RMSE (USD) | R² |
|---|---:|---:|---:|
| **Random Forest** | **1,228.73** | **3,090.51** | **0.848** |
| Decision Tree | 1,429.08 | 3,449.60 | 0.810 |
| Gradient Boosting | 1,513.03 | 3,202.93 | 0.836 |
| Linear Regression | 2,635.14 | 4,919.51 | 0.614 |

![Poređenje modela po MAE](reports/model_comparison_mae.png)

**Izabrani finalni model: Random Forest Regressor** (`n_estimators=100, max_depth=14`).

Obrazloženje: Random Forest ima ubedljivo najnižu grešku (MAE i RMSE) i najviši R² od
svih testiranih modela. Linearna regresija zaostaje jer je odnos cene prema godini/kilometraži
izrazito nelinearan (auto gubi vrednost brže u prvim godinama), dok Decision Tree i Gradient
Boosting rade dobro, ali malo overfituju odnosno underfituju u odnosu na ansambl od 100 stabala
koji Random Forest usrednjava.

## Evaluacija finalnog modela

Na test skupu (11,106 automobila koje model nije video tokom treninga):

- **MAE = 1,228.73 USD** → u proseku, model greši za oko **1,229 dolara** pri proceni cene.
- **RMSE = 3,090.51 USD** → veći od MAE, što ukazuje da postoji manji broj automobila
  sa znatno većom greškom (npr. retki/luksuzni modeli), koji RMSE penalizuje jače.
- **R² = 0.848** → model objašnjava približno **84.8%** varijanse cene u test skupu.

![Stvarna vs. predviđena cena](reports/actual_vs_predicted.png)

Primeri predviđanja iz test skupa:

| stvarna cena | predviđena cena | greška |
|---:|---:|---:|
| 7,690 | 7,624 | 66 |
| 11,250 | 11,039 | 211 |
| 5,350 | 5,795 | 445 |
| 7,400 | 8,393 | 993 |
| 11,000 | 9,022 | 1,978 |

Najznačajnije karakteristike za model su godina proizvodnje, starost automobila i
zapremina motora:

![Značaj karakteristika](reports/feature_importance.png)

### Ograničenja

- Model ne poznaje neke faktore koji realno utiču na cenu (broj vlasnika, istorija
  saobraćajnih nezgoda, oprema/paket, region prodaje), pa je greška od ~1,200$ očekivana.
- Skupi/retki automobili (npr. luksuzni brendovi sa malo primera u podacima) imaju veću
  grešku jer model ima manje primera na kojima uči njihov cenovni obrazac.
- Kolona `model` nije direktno korišćena zbog visoke kardinalnosti — potencijalno
  poboljšanje za budući rad je target/frequency encoding za `model` i `brand_model`.

## Korišćene biblioteke

Videti `requirements.txt`. Glavne: `pandas`, `numpy`, `scikit-learn`, `matplotlib`,
`seaborn`, `joblib`, `jupyter`.
