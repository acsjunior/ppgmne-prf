UF = "PR"
URL_BORDERS = f"https://raw.githubusercontent.com/juniorssz/kml-brasil/master/lib/2010/estados/json/{UF}.json"
URL_STATIONS = (
    "https://drive.google.com/uc?export=download&id=12s-aFZjOFwHYbAeGuODpN2c9gaMw-jdk"
)

URLS_ACCIDENTS = {
    "2018": "https://arquivos.prf.gov.br/arquivos/index.php/s/MaC6cieXSFACNWT/download",
    "2019": "https://arquivos.prf.gov.br/arquivos/index.php/s/kRBUylqz6DyQznN/download",
    "2020": "https://arquivos.prf.gov.br/arquivos/index.php/s/rVfIQjF0wrwHa5P/download",
    "2021": "http://arquivos.prf.gov.br/arquivos/index.php/s/n1T3lymvIdDOzzb/download",
    "2022": "http://arquivos.prf.gov.br/arquivos/index.php/s/OEtK0ObcP55Siei/download",
}

STR_COLS_TO_LOWER = [
    "dia_semana",
    "causa_acidente",
    "tipo_acidente",
    "classificacao_acidente",
    "fase_dia",
    "sentido_via",
    "tipo_pista",
    "tracado_via",
    "uso_solo",
]
STR_COLS_TO_UPPER = ["municipio", "regional", "delegacia", "uop"]

COORDS_MIN_DECIMAL_PLACES = 3
COORDS_PRECISION = 1

CLUSTERING_FEATS = ["quadrant_n_accidents"]
N_CLUSTERS = 8
CLUSTER_DMAX = {1: 180, 2: 180, 3: 120, 4: 120, 5: 120, 6: 60, 7: 60, 8: 60}

MIN_DIST_TOLERANCE = 2
