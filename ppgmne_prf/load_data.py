import json
import os

import kml2geojson
import pandas as pd
from loguru import logger
from pandera import Column, DataFrameSchema
from pandera.errors import SchemaError

from ppgmne_prf.config.params import URL_STATIONS, URLS_ACCIDENTS
from ppgmne_prf.config.paths import PATH_DATA_CACHE_PRF
from ppgmne_prf.utils import csv_zip_to_df, get_binary_from_url


def load_data():
    logger.info("Load data - Início do carregamento os dados de entrada.")

    logger.info("Load data (accidents) - Carregando os dados históricos dos acidentes.")
    df_accidents = __load_accidents()

    logger.info(
        "Load data (stations) - Carregando as coordenadas das UOPs e delegacias."
    )
    dict_stations = __load_stations()

    logger.info("Load data - Fim do carregamento os dados de entrada.")

    return df_accidents, dict_stations


def __load_accidents() -> pd.DataFrame:
    """Função para extração do histórico de acidentes

    Returns
    -------
    pd.DataFrame
        Dados históricos dos acidentes
    """

    # ---- carrega da cache:
    accidents_cache_file = "df_accidents_raw.pkl"
    accidents_cache_path = PATH_DATA_CACHE_PRF / accidents_cache_file
    if os.path.exists(accidents_cache_path):
        logger.warning("Load data (accidents) - Dados carregados da cache.")
        return pd.read_pickle(accidents_cache_path)

    # Data validation:
    key_in = "id"

    df_schema: dict[str, Column] = {
        "id": Column(int),
        "ano": Column(int),
        "data_inversa": Column(str),
        "dia_semana": Column(str),
        "horario": Column(str),
        "uf": Column(str),
        "municipio": Column(str),
        "causa_acidente": Column(str),
        "tipo_acidente": Column(str, nullable=True),
        "classificacao_acidente": Column(str),
        "fase_dia": Column(str),
        "sentido_via": Column(str),
        "tipo_pista": Column(str),
        "tracado_via": Column(str),
        "uso_solo": Column(str),
        "pessoas": Column(int),
        "mortos": Column(int),
        "feridos_leves": Column(int),
        "feridos_graves": Column(int),
        "ilesos": Column(int),
        "ignorados": Column(int),
        "feridos": Column(int),
        "veiculos": Column(int),
        "latitude": Column(str),
        "longitude": Column(str),
        "regional": Column(str, nullable=True),
        "delegacia": Column(str, nullable=True),
        "uop": Column(str, nullable=True),
    }

    # Read data:
    df_out = pd.DataFrame()
    for year in URLS_ACCIDENTS.keys():
        # Lê os dados dos acidentes:
        url = URLS_ACCIDENTS[year]
        file_name = f"datatran{year}.csv"
        df = csv_zip_to_df(url, file_name)
        df["ano"] = year

        logger.info(
            f"Load data (accidents) - Lendo os registros de acidentes de {year}."
        )
        df_out = pd.DataFrame()
        # Valida os dados de entrada:
        try:
            df = DataFrameSchema(
                columns=df_schema,
                unique=key_in,
                coerce=True,
                strict="filter",
            ).validate(df)
        except SchemaError as se:
            logger.error(
                f"Load data (accidents) - Erro ao validar os dados dos acidentes de {year}."
            )
            logger.error(se)

        # Concatena os anos:
        if df_out.shape[0] == 0:
            df_out = df.copy()
        else:
            df_out = pd.concat([df_out, df], ignore_index=True)

    # --- armazena da cache:
    df_out.to_pickle(accidents_cache_path)

    return df_out


def __load_stations() -> dict:
    """Função para obtenção do dicionário com as coordenadas dos limites da região parametrizada

    Returns
    -------
    dict
        Dicionário com as coordenadas dos limites da região parametrizada
    """
    # ---- carrega da cache:
    stations_cache_file = "stations.json"
    stations_cache_path = PATH_DATA_CACHE_PRF / stations_cache_file
    if os.path.exists(stations_cache_path):
        with open(stations_cache_path, "r") as file:
            logger.warning("Load data (stations) - Dados carregados da cache.")
            return json.load(file)

    logger.info(f"Load data (stations) - Lendo as coordenadas das delegacias e UOPs.")
    kml_file = get_binary_from_url(URL_STATIONS)
    out = kml2geojson.main.convert(kml_file)

    # ---- armazena da cache:
    with open(stations_cache_path, "w") as file:
        json.dump(out, file)

    return out
