"""
Modul pentru încărcarea și procesarea datelor reale despre avioane
Folosește OpenSky Network API și alte surse de date
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
import os


@dataclass
class AircraftData:
    """Clasa pentru datele unei aeronave"""
    icao24: str              # Identificator unic ICAO
    callsign: str            # Indicativ de apel
    origin_country: str      # Țara de origine
    longitude: float         # Longitudine (grade)
    latitude: float          # Latitudine (grade)
    altitude: float          # Altitudine (m)
    velocity: float          # Viteză (m/s)
    heading: float           # Direcție (grade)
    vertical_rate: float     # Rată de urcare/coborâre (m/s)
    on_ground: bool          # Pe sol sau în zbor
    timestamp: int           # Timestamp Unix


class AircraftDataLoader:
    """
    Încărcător de date pentru avioane din diverse surse
    Folosește API-ul REST al OpenSky Network
    """
    
    OPENSKY_API_URL = "https://opensky-network.org/api/states/all"
    
    def __init__(self, cache_dir: str = "data/aircraft_cache"):
        """
        Inițializează loader-ul
        
        Args:
            cache_dir: Director pentru cache-ul datelor
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Verificăm dacă requests este disponibil
        self.requests_available = False
        try:
            import requests
            self.requests = requests
            self.requests_available = True
            print("✓ Requests library disponibilă pentru OpenSky API")
        except ImportError:
            print("⚠ Requests nu este instalat. Folosiți: pip install requests")
            self.requests = None
    
    def fetch_live_aircraft(self, 
                           bbox: Optional[Tuple[float, float, float, float]] = None
                           ) -> List[AircraftData]:
        """
        Obține date în timp real despre avioane de la OpenSky Network
        
        Args:
            bbox: Bounding box (lat_min, lat_max, lon_min, lon_max)
                  Exemplu: (45.0, 48.0, 22.0, 30.0) pentru România
        
        Returns:
            Lista de AircraftData
        """
        if not self.requests_available:
            print("Requests nu este disponibil. Se folosesc date simulate.")
            return self._generate_simulated_aircraft()
        
        try:
            # Construim URL-ul cu bounding box dacă e specificat
            url = self.OPENSKY_API_URL
            params = {}
            
            if bbox:
                lat_min, lat_max, lon_min, lon_max = bbox
                params = {
                    'lamin': lat_min,
                    'lamax': lat_max,
                    'lomin': lon_min,
                    'lomax': lon_max
                }
            
            print(f"📡 Conectare la OpenSky Network API...")
            response = self.requests.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                print(f"⚠ Eroare API: {response.status_code}")
                return self._generate_simulated_aircraft()
            
            data = response.json()
            states = data.get('states', [])
            
            if not states:
                print("⚠ Nu s-au găsit avioane în zona specificată")
                return self._generate_simulated_aircraft()
            
            aircraft_list = []
            for s in states:
                # Structura răspunsului OpenSky:
                # [0] icao24, [1] callsign, [2] origin_country, [3] time_position,
                # [4] last_contact, [5] longitude, [6] latitude, [7] baro_altitude,
                # [8] on_ground, [9] velocity, [10] true_track, [11] vertical_rate,
                # [12] sensors, [13] geo_altitude, [14] squawk, [15] spi, [16] position_source
                
                if s[5] is not None and s[6] is not None:  # longitude și latitude
                    aircraft = AircraftData(
                        icao24=s[0] or "unknown",
                        callsign=(s[1] or "").strip(),
                        origin_country=s[2] or "Unknown",
                        longitude=s[5],
                        latitude=s[6],
                        altitude=s[7] or s[13] or 0,  # baro sau geo altitude
                        velocity=s[9] or 0,
                        heading=s[10] or 0,
                        vertical_rate=s[11] or 0,
                        on_ground=s[8] or False,
                        timestamp=s[3] or 0
                    )
                    aircraft_list.append(aircraft)
            
            print(f"✓ Obținute {len(aircraft_list)} avioane de la OpenSky Network")
            return aircraft_list
            
        except self.requests.exceptions.Timeout:
            print("⚠ Timeout la conectarea cu OpenSky API")
            return self._generate_simulated_aircraft()
        except self.requests.exceptions.RequestException as e:
            print(f"⚠ Eroare de rețea: {e}")
            return self._generate_simulated_aircraft()
        except Exception as e:
            print(f"⚠ Eroare la obținerea datelor: {e}")
            return self._generate_simulated_aircraft()
    
    def _generate_simulated_aircraft(self, count: int = 10) -> List[AircraftData]:
        """
        Generează date simulate pentru avioane
        
        Args:
            count: Numărul de avioane de simulat
            
        Returns:
            Lista de AircraftData simulate
        """
        import time
        
        # Date realiste pentru avioane comerciale
        aircraft_types = [
            {"prefix": "ROT", "country": "Romania", "speed_range": (200, 250)},
            {"prefix": "DLH", "country": "Germany", "speed_range": (220, 260)},
            {"prefix": "AFR", "country": "France", "speed_range": (210, 250)},
            {"prefix": "BAW", "country": "United Kingdom", "speed_range": (230, 270)},
            {"prefix": "UAE", "country": "United Arab Emirates", "speed_range": (240, 280)},
        ]
        
        aircraft_list = []
        current_time = int(time.time())
        
        for i in range(count):
            aircraft_type = aircraft_types[i % len(aircraft_types)]
            
            # Poziție aleatorie în Europa
            lat = np.random.uniform(35, 60)  # Europa
            lon = np.random.uniform(-10, 40)
            
            # Altitudine tipică de croazieră
            altitude = np.random.uniform(9000, 12000)  # metri
            
            # Viteză
            velocity = np.random.uniform(*aircraft_type["speed_range"])
            
            aircraft = AircraftData(
                icao24=f"{np.random.randint(100000, 999999):06x}",
                callsign=f"{aircraft_type['prefix']}{np.random.randint(100, 999)}",
                origin_country=aircraft_type["country"],
                longitude=lon,
                latitude=lat,
                altitude=altitude,
                velocity=velocity,
                heading=np.random.uniform(0, 360),
                vertical_rate=np.random.uniform(-5, 5),
                on_ground=False,
                timestamp=current_time
            )
            aircraft_list.append(aircraft)
        
        print(f"✓ Generate {len(aircraft_list)} avioane simulate")
        return aircraft_list
    
    def convert_to_radar_targets(self, 
                                aircraft_list: List[AircraftData],
                                radar_position: Tuple[float, float] = (44.4268, 26.1025)
                                ) -> List[Dict]:
        """
        Convertește datele avioanelor în ținte radar
        
        Args:
            aircraft_list: Lista de avioane
            radar_position: Poziția radarului (lat, lon) - default București
            
        Returns:
            Lista de dicționare cu parametri pentru simularea radar
        """
        radar_lat, radar_lon = radar_position
        targets = []
        
        for aircraft in aircraft_list:
            if aircraft.on_ground:
                continue  # Ignorăm avioanele la sol
            
            # Calculăm distanța de la radar (simplificat, folosind haversine)
            distance = self._haversine_distance(
                radar_lat, radar_lon,
                aircraft.latitude, aircraft.longitude
            )
            
            # Convertim în parametri radar
            # Viteza radială (aproximație bazată pe heading)
            angle_to_radar = np.arctan2(
                radar_lon - aircraft.longitude,
                radar_lat - aircraft.latitude
            )
            heading_rad = np.radians(aircraft.heading)
            
            # Componenta radială a vitezei
            radial_velocity = aircraft.velocity * np.cos(heading_rad - angle_to_radar)
            
            # RCS estimat bazat pe tipul aeronavei (simplificat)
            rcs = self._estimate_rcs(aircraft.callsign)
            
            target = {
                'distance': distance,
                'velocity': radial_velocity,
                'rcs': rcs,
                'altitude': aircraft.altitude,
                'callsign': aircraft.callsign,
                'country': aircraft.origin_country,
                'icao24': aircraft.icao24
            }
            targets.append(target)
        
        # Sortăm după distanță
        targets.sort(key=lambda x: x['distance'])
        
        return targets
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """
        Calculează distanța între două puncte pe suprafața Pământului
        
        Returns:
            Distanța în metri
        """
        R = 6371000  # Raza Pământului în metri
        
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_phi/2)**2 + 
             np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def _estimate_rcs(self, callsign: str) -> float:
        """
        Estimează RCS bazat pe indicativul de apel
        
        Returns:
            RCS în m²
        """
        # Prefixe pentru avioane mari (wide-body)
        large_aircraft = ['UAE', 'QTR', 'SIA', 'CPA', 'ANA']
        
        # Prefixe pentru avioane medii
        medium_aircraft = ['DLH', 'AFR', 'BAW', 'KLM', 'ROT']
        
        prefix = callsign[:3] if len(callsign) >= 3 else callsign
        
        if prefix in large_aircraft:
            return np.random.uniform(50, 100)  # Boeing 777, A380
        elif prefix in medium_aircraft:
            return np.random.uniform(20, 50)   # Boeing 737, A320
        else:
            return np.random.uniform(10, 30)   # Avioane regionale
    
    def save_to_cache(self, aircraft_list: List[AircraftData], 
                     filename: str = "aircraft_data.json"):
        """Salvează datele în cache"""
        cache_path = os.path.join(self.cache_dir, filename)
        
        data = []
        for aircraft in aircraft_list:
            data.append({
                'icao24': aircraft.icao24,
                'callsign': aircraft.callsign,
                'origin_country': aircraft.origin_country,
                'longitude': aircraft.longitude,
                'latitude': aircraft.latitude,
                'altitude': aircraft.altitude,
                'velocity': aircraft.velocity,
                'heading': aircraft.heading,
                'vertical_rate': aircraft.vertical_rate,
                'on_ground': aircraft.on_ground,
                'timestamp': aircraft.timestamp
            })
        
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Date salvate în {cache_path}")
    
    def load_from_cache(self, filename: str = "aircraft_data.json"
                       ) -> List[AircraftData]:
        """Încarcă datele din cache"""
        cache_path = os.path.join(self.cache_dir, filename)
        
        if not os.path.exists(cache_path):
            print(f"⚠ Cache-ul {cache_path} nu există")
            return []
        
        with open(cache_path, 'r') as f:
            data = json.load(f)
        
        aircraft_list = []
        for item in data:
            aircraft = AircraftData(
                icao24=item['icao24'],
                callsign=item['callsign'],
                origin_country=item['origin_country'],
                longitude=item['longitude'],
                latitude=item['latitude'],
                altitude=item['altitude'],
                velocity=item['velocity'],
                heading=item['heading'],
                vertical_rate=item['vertical_rate'],
                on_ground=item['on_ground'],
                timestamp=item['timestamp']
            )
            aircraft_list.append(aircraft)
        
        print(f"✓ Încărcate {len(aircraft_list)} avioane din cache")
        return aircraft_list


# ==============================================================================
# DATASETS DISPONIBILE PENTRU ANALIZĂ ȘI TESTARE
# ==============================================================================

AVAILABLE_DATASETS = {
    # ==================== PENTRU ANALIZĂ ====================
    "analysis": {
        "opensky_network": {
            "name": "OpenSky Network",
            "description": "Date ADS-B în timp real și istorice pentru avioane din întreaga lume",
            "url": "https://opensky-network.org/",
            "api": "https://github.com/openskynetwork/opensky-api",
            "data_types": ["state_vectors", "flights", "aircraft_metadata"],
            "access": "Gratuit cu cont (rate limit), Premium pentru acces complet",
            "format": "JSON/CSV via API sau Impala SQL",
            "size": "Miliarde de înregistrări",
            "license": "CC BY-NC 4.0 (non-comercial)",
            "use_case": "Analiza traficului aerian, validare algoritmi tracking"
        },
        "flightradar24": {
            "name": "Flightradar24",
            "description": "Date despre zboruri globale cu vizualizare în timp real",
            "url": "https://www.flightradar24.com/",
            "data_types": ["flight_paths", "aircraft_info", "airport_data"],
            "access": "API comercial, date limitate gratuit",
            "format": "JSON via API",
            "license": "Comercial",
            "use_case": "Referință pentru verificare, date despre rute"
        },
        "ads_b_exchange": {
            "name": "ADS-B Exchange",
            "description": "Feed ADS-B nefiltrată, include avioane militare",
            "url": "https://www.adsbexchange.com/",
            "api": "https://rapidapi.com/adsbx/api/adsbexchange-com1",
            "data_types": ["raw_adsb", "military_aircraft", "historical"],
            "access": "Gratuit pentru date istorice, API plătit",
            "format": "JSON",
            "license": "Open data",
            "use_case": "Analiza completă incluzând avioane militare"
        },
        "eurocontrol_ddr": {
            "name": "EUROCONTROL DDR2",
            "description": "Date despre zboruri în spațiul aerian european",
            "url": "https://www.eurocontrol.int/ddr",
            "data_types": ["flight_plans", "trajectories", "traffic_stats"],
            "access": "Cerere acces pentru cercetare",
            "format": "SO6, ALL_FT+",
            "license": "Academic/Research",
            "use_case": "Analiză trafic european, planificare"
        }
    },
    
    # ==================== PENTRU TESTARE ====================
    "testing": {
        "kaggle_aircraft": {
            "name": "Aircraft Detection Datasets (Kaggle)",
            "description": "Multiple datasets pentru detecție avioane din imagini/radar",
            "url": "https://www.kaggle.com/search?q=aircraft+detection",
            "datasets": [
                "https://www.kaggle.com/datasets/airbusgeo/airbus-aircrafts-sample-dataset",
                "https://www.kaggle.com/datasets/khlaifiabilel/airplane-detection"
            ],
            "format": "CSV, Images",
            "access": "Gratuit cu cont Kaggle",
            "use_case": "Training ML, benchmark detecție"
        },
        "synthetic_radar": {
            "name": "Synthetic Radar Data",
            "description": "Date radar sintetice pentru testare algoritmi",
            "sources": [
                "https://ieee-dataport.org/keywords/radar",
                "https://github.com/topics/radar-simulation"
            ],
            "format": "MAT, HDF5, CSV",
            "access": "Variat",
            "use_case": "Validare algoritmi în condiții controlate"
        },
        "rf_target_classification": {
            "name": "RF Dataset for Radar Target Classification",
            "description": "Dataset UCLA pentru clasificare ținte radar",
            "url": "https://ieee-dataport.org/documents/rf-dataset-radar-target-classification",
            "format": "MAT files",
            "access": "IEEE DataPort (poate necesita cont)",
            "use_case": "Clasificare ținte, ML training"
        },
        "ship_radar": {
            "name": "TI mmWave Ship Dataset",
            "description": "Range profiles pentru nave - poate fi adaptat pentru avioane",
            "url": "https://ieee-dataport.org/documents/ti-mmwave-down-scaled-ship-dataset-0",
            "format": "MAT files",
            "access": "IEEE DataPort",
            "use_case": "Testare procesare Range-Doppler"
        }
    },
    
    # ==================== GROUND TRUTH ====================
    "ground_truth": {
        "faa_aircraft_registry": {
            "name": "FAA Aircraft Registry",
            "description": "Registru oficial avioane SUA",
            "url": "https://www.faa.gov/licenses_certificates/aircraft_certification/aircraft_registry",
            "format": "CSV",
            "use_case": "Validare identificare avioane"
        },
        "icao_aircraft_database": {
            "name": "ICAO Aircraft Type Designators",
            "description": "Database oficial ICAO pentru tipuri de avioane",
            "url": "https://www.icao.int/publications/DOC8643/Pages/Search.aspx",
            "use_case": "Mapping ICAO codes, RCS estimation"
        }
    }
}


def print_available_datasets():
    """Afișează toate dataset-urile disponibile"""
    print("\n" + "="*70)
    print("DATASETS DISPONIBILE PENTRU SISTEM RADAR")
    print("="*70)
    
    for category, datasets in AVAILABLE_DATASETS.items():
        print(f"\n{'='*30} {category.upper()} {'='*30}")
        
        for key, info in datasets.items():
            print(f"\n📦 {info['name']}")
            print(f"   {info['description']}")
            print(f"   🔗 URL: {info.get('url', 'N/A')}")
            print(f"   📋 Format: {info.get('format', 'N/A')}")
            print(f"   🔑 Acces: {info.get('access', 'N/A')}")
            print(f"   💡 Use case: {info.get('use_case', 'N/A')}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Demo
    print_available_datasets()
    
    print("\n" + "="*70)
    print("DEMO: Încărcare date avioane")
    print("="*70)
    
    loader = AircraftDataLoader()
    
    # Generează date simulate (sau folosește OpenSky dacă e disponibil)
    aircraft = loader.fetch_live_aircraft()
    
    if aircraft:
        print(f"\nPrimele 5 avioane:")
        for i, a in enumerate(aircraft[:5], 1):
            print(f"  {i}. {a.callsign:8s} | {a.origin_country:15s} | "
                  f"Alt: {a.altitude:8.0f}m | Vel: {a.velocity:6.1f} m/s")
        
        # Convertire la ținte radar
        targets = loader.convert_to_radar_targets(aircraft)
        
        print(f"\nȚinte radar (relative la București):")
        for i, t in enumerate(targets[:5], 1):
            print(f"  {i}. {t['callsign']:8s} | Dist: {t['distance']/1000:7.1f} km | "
                  f"V_rad: {t['velocity']:7.1f} m/s | RCS: {t['rcs']:5.1f} m²")
        
        # Salvare cache
        loader.save_to_cache(aircraft)
