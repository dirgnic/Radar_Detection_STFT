# Datasets pentru Proiectul de Detecție Radar a Avioanelor

## Cuprins
1. [Surse pentru Analiză](#surse-pentru-analiză)
2. [Surse pentru Testare](#surse-pentru-testare)
3. [Date de Referință (Ground Truth)](#date-de-referință)
4. [Cum să folosiți datele](#cum-să-folosiți-datele)

---

## Surse pentru Analiză

### 1. OpenSky Network ⭐ (Recomandat)
**Descriere:** Date ADS-B în timp real și istorice pentru avioane din întreaga lume

- **URL:** https://opensky-network.org/
- **API Python:** https://github.com/openskynetwork/opensky-api
- **Tipuri de date:** State vectors, zboruri, metadata avioane
- **Acces:** Gratuit cu cont (rate limit), Premium pentru acces complet
- **Format:** JSON/CSV via API sau Impala SQL
- **Dimensiune:** Miliarde de înregistrări
- **Licență:** CC BY-NC 4.0 (non-comercial)

**Instalare:**
```bash
pip install opensky-api
```

**Utilizare:**
```python
from opensky_api import OpenSkyApi

api = OpenSkyApi()
states = api.get_states()

for s in states.states:
    print(f"{s.callsign}: {s.latitude}, {s.longitude}, {s.velocity} m/s")
```

**Use case:** Analiza traficului aerian, validare algoritmi tracking, date real-time

---

### 2. Flightradar24
**Descriere:** Date despre zboruri globale cu vizualizare în timp real

- **URL:** https://www.flightradar24.com/
- **Tipuri de date:** Traiectorii de zbor, informații avioane, date aeroporturi
- **Acces:** API comercial, date limitate gratuit
- **Format:** JSON via API
- **Licență:** Comercial

**Use case:** Referință pentru verificare, date despre rute

---

### 3. ADS-B Exchange
**Descriere:** Feed ADS-B nefiltrată, include avioane militare

- **URL:** https://www.adsbexchange.com/
- **API:** https://rapidapi.com/adsbx/api/adsbexchange-com1
- **Tipuri de date:** Raw ADS-B, avioane militare, date istorice
- **Acces:** Gratuit pentru date istorice, API plătit
- **Format:** JSON
- **Licență:** Open data

**Use case:** Analiza completă incluzând avioane militare

---

### 4. EUROCONTROL DDR2
**Descriere:** Date despre zboruri în spațiul aerian european

- **URL:** https://www.eurocontrol.int/ddr
- **Tipuri de date:** Planuri de zbor, traiectorii, statistici trafic
- **Acces:** Cerere acces pentru cercetare academică
- **Format:** SO6, ALL_FT+
- **Licență:** Academic/Research

**Use case:** Analiză trafic european, planificare

---

## Surse pentru Testare

### 1. Kaggle Aircraft Detection Datasets
**Descriere:** Multiple datasets pentru detecție avioane

- **URL:** https://www.kaggle.com/search?q=aircraft+detection
- **Datasets notabile:**
  - [Airbus Aircraft Sample Dataset](https://www.kaggle.com/datasets/airbusgeo/airbus-aircrafts-sample-dataset)
  - [Airplane Detection](https://www.kaggle.com/datasets/khlaifiabilel/airplane-detection)
- **Format:** CSV, Images
- **Acces:** Gratuit cu cont Kaggle

**Use case:** Training ML, benchmark detecție

---

### 2. RF Dataset for Radar Target Classification (UCLA)
**Descriere:** Dataset pentru clasificare ținte radar

- **URL:** https://ieee-dataport.org/documents/rf-dataset-radar-target-classification
- **Format:** MAT files (MATLAB)
- **Acces:** IEEE DataPort (poate necesita cont)

**Use case:** Clasificare ținte, ML training

---

### 3. Synthetic Radar Data
**Descriere:** Date radar sintetice pentru testare algoritmi

- **Surse:**
  - https://ieee-dataport.org/keywords/radar
  - https://github.com/topics/radar-simulation
- **Format:** MAT, HDF5, CSV
- **Acces:** Variat

**Use case:** Validare algoritmi în condiții controlate

---

### 4. TI mmWave Ship Dataset
**Descriere:** Range profiles pentru nave - poate fi adaptat pentru avioane

- **URL:** https://ieee-dataport.org/documents/ti-mmwave-down-scaled-ship-dataset-0
- **Format:** MAT files
- **Acces:** IEEE DataPort

**Use case:** Testare procesare Range-Doppler

---

## Date de Referință

### FAA Aircraft Registry
**Descriere:** Registru oficial avioane SUA

- **URL:** https://www.faa.gov/licenses_certificates/aircraft_certification/aircraft_registry
- **Format:** CSV
- **Use case:** Validare identificare avioane, RCS estimation

---

### ICAO Aircraft Type Designators
**Descriere:** Database oficial ICAO pentru tipuri de avioane

- **URL:** https://www.icao.int/publications/DOC8643/Pages/Search.aspx
- **Use case:** Mapping ICAO codes, estimare RCS bazată pe tip

---

## Cum să folosiți datele

### În acest proiect

Modulul `src/aircraft_data.py` oferă funcționalitate pentru:

1. **Descărcare date reale** (OpenSky Network):
```python
from src.aircraft_data import AircraftDataLoader

loader = AircraftDataLoader()

# Date din zona României
romania_bbox = (43.5, 48.5, 20.0, 30.5)
aircraft = loader.fetch_live_aircraft(bbox=romania_bbox)
```

2. **Conversie la ținte radar:**
```python
# Convertește la format radar (relativ la București)
radar_position = (44.4268, 26.1025)
targets = loader.convert_to_radar_targets(aircraft, radar_position)

for t in targets:
    print(f"Distanță: {t['distance']/1000:.1f} km, "
          f"Viteză radială: {t['velocity']:.1f} m/s")
```

3. **Simulare completă:**
```bash
cd /Users/ingridcorobana/Desktop/An_III/final_projs/PS_proj
source venv/bin/activate
python simulations/real_aircraft_simulation.py
```

### Workflow recomandat

```
┌─────────────────────────────────────────────────────────────────┐
│                    WORKFLOW UTILIZARE DATE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. ANALIZĂ (date reale)                                       │
│      └── OpenSky Network API                                     │
│          └── Date ADS-B în timp real                            │
│          └── Poziții, viteze, altitudini reale                  │
│                                                                  │
│   2. TESTARE (date controlate)                                  │
│      └── Date sintetice generate de proiect                     │
│      └── Parametri configurabili pentru edge cases              │
│      └── Kaggle datasets pentru benchmark                       │
│                                                                  │
│   3. VALIDARE (ground truth)                                    │
│      └── Comparare rezultate cu date OpenSky                    │
│      └── Verificare tip avion cu ICAO/FAA                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Bibliografie Datasets

1. M. Schäfer, M. Strohmeier, V. Lenders, I. Martinovic, M. Wilhelm, "Bringing Up OpenSky: A Large-scale ADS-B Sensor Network for Research", IPSN 2014

2. IEEE DataPort - Radar Datasets: https://ieee-dataport.org/keywords/radar

3. Kaggle Aviation Datasets: https://www.kaggle.com/search?q=aircraft

---

*Ultima actualizare: Ianuarie 2025*
