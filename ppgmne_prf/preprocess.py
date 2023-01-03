import json

import holidays
import numpy as np
import pandas as pd
import scipy.stats as stats
from loguru import logger
from shapely.geometry import Point, Polygon
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from unidecode import unidecode

from ppgmne_prf.config.params import (
    CLUSTER_DMAX,
    CLUSTERING_FEATS,
    COORDS_MIN_DECIMAL_PLACES,
    COORDS_PRECISION,
    MIN_DIST_TOLERANCE,
    N_CLUSTERS,
    STR_COLS_TO_LOWER,
    STR_COLS_TO_UPPER,
    UF,
    URL_BORDERS,
)
from ppgmne_prf.config.paths import PATH_DATA_PRF
from ppgmne_prf.utils import (
    clean_string,
    concatenate_dict_of_dicts,
    get_binary_from_url,
    get_decimal_places,
    get_distance_matrix,
    trace_df,
)


def preprocess(df_accidents: pd.DataFrame, dict_stations: dict) -> pd.DataFrame:
    logger.info("Pre-process - Início do pré-processamento dos dados de entrada.")

    logger.info("Pre-process (accidents) - Início do pré-processamento dos dados dos acidentes.")
    df_accidents = __preprocess_accidents(df_accidents)

    logger.info("Pre-process (stations) - Início do pré-processamento dos dados das estações policiais.")
    df_stations = __preprocess_stations(dict_stations)

    logger.info("Pre-process (quadrants) - Início do pré-processamento dos dados dos quadrantes.")
    df_quadrants = __preprocess_quadrants(df_accidents, df_stations)

    logger.info("Pre-process - Fim do pré-processamento dos dados de entrada.")
    return df_quadrants

########## Accidents ##########

def __preprocess_accidents(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Pre-process (accidents) - Removendo registros incompletos.")
    df = df.dropna().pipe(trace_df).copy()

    df = (
        df.pipe(__filter_uf)
        .pipe(trace_df)
        .pipe(__create_datetime_column)
        .pipe(trace_df)
        .pipe(__classify_holiday_and_weekend)
        .pipe(trace_df)
    )

    logger.info("Pre-process (accidents) - Padronizando os campos do tipo string.")
    df = clean_string(df, STR_COLS_TO_UPPER, "upper")
    df = clean_string(df, STR_COLS_TO_LOWER).pipe(trace_df)

    df = (
        df.pipe(__convert_lat_lon)
        .pipe(trace_df)
        .pipe(__keep_min_decimal_places)
        .pipe(trace_df)
        .pipe(__keep_geo_correct_rows)
        .pipe(trace_df)
        .pipe(__manual_transformations)
        .pipe(trace_df)
        .pipe(__remove_outlier_coords)
        .pipe(trace_df)
    )

    return df

def __preprocess_stations(dict_coords: dict) -> pd.DataFrame:
    
    logger.info("Pre-process (stations) - Estruturando os dados das estações policiais.") 

    dict_chars = {
        "type": [],
        "name": [],
        "station_father": [],
        "station_code": [],
        "address": [],
        "municipality": [],
        "state": [],
        "phone": [],
        "email_del": [],
        "email_uop": [],
        "latitude": [],
        "longitude": [],
    }

    lst_full_description = []

    for d in dict_coords[0]["features"]:
        if d["type"] == "Feature":

            # Extrai as informações parcialmente tratadas:
            full_description = d["properties"]["description"].split("<br>")
            longitude = float(
                str(d["geometry"]["coordinates"][0]).replace(",", ".")
            )
            latitude = float(str(d["geometry"]["coordinates"][1]).replace(",", "."))

            # Insere as informações iniciais no dicionário:
            dict_chars["longitude"].append(longitude)
            dict_chars["latitude"].append(latitude)

            # Insere a descrição completa em uma lista temporária:
            lst_full_description.append(full_description)

            # Extrai as informações detalhadas da descrição:
            for x in lst_full_description:
                name = unidecode(x[0]).strip().upper()

                if "SUPERINTENDENCIA" in name:
                    type = "SPRF"
                elif "DELEGACIA" in name:
                    type = "DEL"
                elif "UOP" in name:
                    type = "UOP"
                else:
                    type = "other"

                address = x[1]

                municipality = unidecode(x[2].split("/")[0]).strip().upper()
                state = unidecode(x[2].split("/")[1]).strip().upper()

                phone = x[4].strip().lower().replace("telefone:", "").strip()

                email_del = x[5].strip().lower().replace("email:", "").strip()

                if len(x) == 7:
                    email_uop = np.nan
                else:
                    email_uop = x[6].strip().lower()

            # Extrai o código do posto a partir do email_del:
            station_father = np.nan
            if type == "SPRF":
                station_father = type
            elif not pd.isnull(email_del):
                station_father = email_del.upper().split(".")[0]

            # Extrai o código do posto a partir do email_uop:
            station_code = np.nan
            if not pd.isnull(email_uop):
                station_code = email_uop.upper().split(".")[0]

            # Insere as informações finais no dicionário:
            dict_chars["type"].append(type)
            dict_chars["name"].append(name)
            dict_chars["station_father"].append(station_father)
            dict_chars["station_code"].append(station_code)
            dict_chars["address"].append(address)
            dict_chars["municipality"].append(municipality)
            dict_chars["state"].append(state)
            dict_chars["phone"].append(phone)
            dict_chars["email_del"].append(email_del)
            dict_chars["email_uop"].append(email_uop)

    df_out = pd.DataFrame(dict_chars)

    logger.info("Pre-process (stations) - Incluindo os códigos das UOPs.")
    with open(PATH_DATA_PRF / "transformations.json") as file:
        stations_to_replace = concatenate_dict_of_dicts(
            json.load(file)["stations_replace"]
        )
        df_out["uop"] = df_out["name"].map(stations_to_replace)

    df_out.pipe(trace_df)

    return df_out


def __preprocess_quadrants(df: pd.DataFrame, df_stations: pd.DataFrame) -> pd.DataFrame:

    # Seleciona as UOPs:
    df_uops = df_stations.query('type == "UOP"').copy()
    df_uops["latitude"] = df_uops["latitude"].round(COORDS_PRECISION)
    df_uops["longitude"] = df_uops["longitude"].round(COORDS_PRECISION)

    df = (
        df.pipe(trace_df)
        .pipe(__identify_quadrant)
        .pipe(trace_df)
        .pipe(__get_quadrant_stats)
        .pipe(trace_df)
        .pipe(__get_quadrant_clusters)
        .pipe(trace_df)
        .pipe(__aggregate_quadrants)
        .pipe(trace_df))

    logger.info("Pre-process (quadrants) - Incluindo o DMAX na base.")
    df["dist_max"] = df["cluster"].map(CLUSTER_DMAX)

    df = __rename_corresp_quadrants(df, df_uops).pipe(trace_df)

    logger.info("Pre-process (quadrants) - Criando as flags 'is_uop' e 'is_only_uop'.")
    df["is_uop"] = ~df["uop_name"].isna()
    df["is_only_uop"] = False
    df.drop(columns="uop_name", inplace=True)

    df = __add_only_uops(df, df_uops)
    df.pipe(trace_df)

    return df

           
def __filter_uf(df: pd.DataFrame) -> pd.DataFrame:
    """Função para filtrar somente os registros nas delegacias da UF desejada.

    Parameters
    ----------
    df : pd.DataFrame
        Base de acidentes.

    Returns
    -------
    pd.DataFrame
        Base de acidentes filtrada.
    """

    logger.info(
        f"Pre-process (accidents) - Mantendo somente os registros das delegacias do {UF}."
    )
    df["delegacia"] = df["delegacia"].str.upper()
    df_out = df[
        df["delegacia"].str.contains("|".join([UF]))
    ].copy()  # função preparada para receber múltiplas UFs

    return df_out


def __create_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """Função para criação do campo 'data_hora' e remoção dos campos 'data_inversa' e 'hora'.

    Parameters
    ----------
    df : pd.DataFrame
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    logger.info("Pre-process (accidents) - Criando o campo data_hora.")
    df["data_hora"] = pd.to_datetime(df["data_inversa"] + " " + df["horario"])
    df.drop(columns=["data_inversa", "horario"], inplace=True)

    return df


def __classify_holiday_and_weekend(df: pd.DataFrame) -> pd.DataFrame:
    """Função para classificar feriados e finais de semana.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame completo.

    Returns
    -------
    pd.DataFrame
        Data frame com a flag "is_holiday".
    """
    logger.info(
        "Pre-process (accidents) - Criando as flags de feriado e final de semana."
    )
    br_holidays = holidays.country_holidays("BR", subdiv=UF)
    df["is_holiday"] = ~(df["data_hora"].apply(br_holidays.get)).isna()
    df["is_weekend"] = df["data_hora"].dt.weekday >= 5

    return df


def __convert_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    """Função para conversão do formato dos campos 'latitude' e 'longitude'.

    Parameters
    ----------
    df : pd.DataFrame
        Base de acidentes.

    Returns
    -------
    pd.DataFrame
        Base de acidentes com as conversões realizadas.
    """
    logger.info(
        "Pre-process (accidents) - Convertendo os tipos dos campos latitude e longitude."
    )
    df["latitude"] = (df["latitude"].str.replace(",", ".")).astype(float)
    df["longitude"] = (df["longitude"].str.replace(",", ".")).astype(float)

    return df


def __keep_min_decimal_places(df: pd.DataFrame) -> pd.DataFrame:
    """Função para garantir na base somente registros com coordenadas atendendo um número mínimo de casas decimais.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame completo.

    Returns
    -------
    pd.DataFrame
        Data frame com os registros removidos.
    """
    logger.info(
        f"Pre-process (accidents) - Eliminando registros com lat/lon com menos de {COORDS_MIN_DECIMAL_PLACES} casas decimais."
    )
    mask_lat = get_decimal_places(df["latitude"]) >= COORDS_MIN_DECIMAL_PLACES
    mask_lon = get_decimal_places(df["longitude"]) >= COORDS_MIN_DECIMAL_PLACES
    df_out = df[mask_lat & mask_lon]

    return df_out


def __keep_geo_correct_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Função para garantir registros ocorridos na região geográfica de interesse.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame completo.

    Returns
    -------
    pd.DataFrame
        Data frame com os registros removidos.
    """
    logger.info(
        f"Pre-process (accidents) - Mantendo somente registros de acidentes ocorridos geograficamente no {UF}."
    )

    polygon = get_polygon()
    isin_polygon = df.apply(
        lambda x: within_polygon(x.longitude, x.latitude, polygon), axis=1
    )

    df_out = df[isin_polygon].copy()

    return df_out


def get_polygon():
    """Função para carregamento do json com as coordenadas e construção do polígono da região de interesse.

    Returns
    -------
    _type_
        Polígono da região de interesse.
    """
    borders = json.load(get_binary_from_url(URL_BORDERS))["borders"][0]
    
    lst_lon = [x["lng"] for x in borders]
    lst_lat = [x["lat"] for x in borders]
    polygon = Polygon(zip(lst_lon, lst_lat))

    return polygon


def within_polygon(lng: float, lat: float, polygon: Polygon) -> bool:
    """Função para identificar se um ponto está dentro de um polígono.

    Parameters
    ----------
    lng : float
        Longitude do ponto.
    lat : float
        Latitude do ponto.
    polygon : Polygon
        Polígono da região de interesse.

    Returns
    -------
    bool
        Verdadeiro se o ponto está dentro do polígono. Falso, caso contrário.
    """
    point = Point(float(lng), float(lat))
    isin_polygon = point.within(polygon)

    return isin_polygon

def __manual_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """Função para aplicar as correções necessárias identificadas após análise.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame dos acidentes.

    Returns
    -------
    pd.DataFrame
        Data frame com as correções aplicadas.
    """
    logger.info(f"Pre-process (accidents) - Aplicação das correções manuais.")

    # Lê o json com as correções manuais:
    with open(PATH_DATA_PRF / "transformations.json") as file:
        transformations = json.load(file)

        accidents_to_delete_by_uop = transformations["accidents_deletion"]["uop"]
        uops_to_replace = transformations["accidents_replace"]["uop"]
        dels_to_replace = transformations["accidents_replace"]["del"]

    # Deleta os registros a serem desconsiderados:
    df_out = df[~df["uop"].isin(accidents_to_delete_by_uop)].copy()

    # Corrige os registros:
    right_dels = df_out["uop"].map(dels_to_replace)
    right_uops = df_out["uop"].map(uops_to_replace)
    df_out["delegacia"] = right_dels.combine_first(df_out["delegacia"])
    df_out["uop"] = right_uops.combine_first(df_out["uop"])

    return df_out

def __remove_outlier_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Função para remover registros de acidentes considerados outliers aos demais pontos alocados na mesma delegacia.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame completo.

    Returns
    -------
    pd.DataFrame
        Data frame com outliers removidos.
    """
    logger.info("Pre-process (accidents) - Eliminando as coordenadas outliers por delegacia.")

    lat_abs_zscore = (
        df.groupby(["delegacia"])["latitude"]
        .transform(lambda x: stats.zscore(x, ddof=1))
        .abs()
    )
    lon_abs_zscore = (
        df.groupby(["delegacia"])["longitude"]
        .transform(lambda x: stats.zscore(x, ddof=1))
        .abs()
    )

    mask = (lat_abs_zscore <= 3) & (lon_abs_zscore <= 3)
    df_out = df[mask]

    return df_out


def __identify_quadrant(df: pd.DataFrame) -> pd.DataFrame:
    """Função para criar identificação única e padronizar o nome do município dos quadrantes.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame completo.

    Returns
    -------
    pd.DataFrame
        Data frame com os campos "quadrant_municipality" e "quadrant_name".
    """

    logger.info("Pre-process (quadrants) - Criando a identificação dos quadrantes.")

    df["quadrant_latitude"] = df["latitude"].round(COORDS_PRECISION)
    df["quadrant_longitude"] = df["longitude"].round(COORDS_PRECISION)

    # Identifica o município do quadrante por ordem de frequência de acidentes:
    df_mun = (
        df.groupby(["quadrant_latitude", "quadrant_longitude", "municipio"])[
            "data_hora"
        ]
        .count()
        .reset_index(name="n_accidents")
    )
    df_mun["seq"] = df_mun.groupby(["quadrant_latitude", "quadrant_longitude"])[
        "n_accidents"
    ].rank("first", ascending=False)
    df_mun = df_mun[df_mun["seq"] == 1].copy()
    df_mun.rename(columns={"municipio": "quadrant_municipality"}, inplace=True)

    # Cria um identificador único para o quadrante:
    zfill_param = len(
        str(
            df_mun.groupby(["quadrant_municipality"])["quadrant_latitude"]
            .count()
            .reset_index(name="n")["n"]
            .max()
        )
    )
    df_mun.sort_values(
        by=["quadrant_municipality", "quadrant_latitude", "quadrant_longitude"],
        inplace=True,
    )
    df_mun["x"] = 1
    df_mun["suf"] = (
        (df_mun.groupby("quadrant_municipality")["x"].rank("first"))
        .astype(int)
        .astype(str)
        .str.zfill(zfill_param)
    )
    df_mun["quadrant_name"] = df_mun["quadrant_municipality"] + " " + df_mun["suf"]

    # Remove os campos desnecessários:
    df_mun.drop(columns=["n_accidents", "seq", "x", "suf"], inplace=True)

    # Inclui os campos no df final:
    df_out = df.merge(df_mun, on=["quadrant_latitude", "quadrant_longitude"])

    return df_out

def __get_quadrant_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Função para calcular as estatísticas por quadrante.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame completo.

    Returns
    -------
    pd.DataFrame
        Data frame com as estatísticas calculadas.
    """
    logger.info("Pre-process (quadrants) - Calculando as estatísticas dos quadrantes.")

    # Calcula as estatísticas:
    df_stats = (
        df.groupby(["quadrant_name"])
        .agg(
            quadrant_n_accidents=("data_hora", "count"),
            quadrant_n_acc_holiday=("is_holiday", sum),
            quadrant_n_acc_weekend=("is_weekend", sum),
            quadrant_n_injuried=("feridos_graves", sum),
            quadrant_n_dead=("mortos", sum),
        )
        .reset_index()
    )

    # Inclui os dados no df final:
    df_out = df.merge(df_stats, on="quadrant_name")

    return df_out

def __get_quadrant_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Função para clusterização dos quadrantes.

    Aplica o método hierárquico de Ward.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame completo.

    Returns
    -------
    pd.DataFrame
        Data frame com o cluster de cada quadrante.
    """

    logger.info("Pre-process (quadrants) - Identificando o cluster de cada quadrante.")

    df_quadrant = (
        df[["quadrant_name"] + CLUSTERING_FEATS]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    df_cluster = df_quadrant[CLUSTERING_FEATS].copy()
    df_cluster.iloc[:, :] = StandardScaler().fit_transform(df_cluster)

    hc = AgglomerativeClustering(
        n_clusters=N_CLUSTERS, metric="euclidean", linkage="ward"
    )

    df_quadrant["cluster"] = (hc.fit_predict(df_cluster)).astype(str)

    # Calcula as estatísticas por cluster:
    df_stats = None
    for cluster in df_quadrant["cluster"].value_counts().index:
        df_stats_i = pd.DataFrame(
            df_quadrant[df_quadrant["cluster"] == cluster][
                "quadrant_n_accidents"
            ].describe()
        ).T
        df_stats_i["cluster"] = cluster
        if df_stats is None:
            df_stats = df_stats_i.copy()
        else:
            df_stats = pd.concat([df_stats, df_stats_i])
    df_stats = df_stats.sort_values(by="mean").reset_index(drop=True)

    # Renomeia os clusters:
    clusters = np.arange(1, N_CLUSTERS + 1, 1)
    df_stats["quadrant_cluster"] = clusters
    df_stats["quadrant_cluster"] = pd.Categorical(
        df_stats["quadrant_cluster"], categories=clusters, ordered=True
    )

    # Inclui os clusters renomeados na base de quadrantes:
    df_quadrant = df_quadrant.merge(
        df_stats[["cluster", "quadrant_cluster"]], on="cluster"
    )

    # Inclui os clusters na base final:
    df_out = df.merge(
        df_quadrant[["quadrant_name", "quadrant_cluster"]], on="quadrant_name"
    )

    return df_out

def __aggregate_quadrants(df: pd.DataFrame) -> pd.DataFrame:
    """Função para criar a base de quadrantes agregados.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame na granularidade de acidentes.

    Returns
    -------
    pd.DataFrame
        Data frame agregado por quadrante.
    """

    logger.info("Pre-process (quadrants) - Agregando os dados por quadrante.")

    preffix = "quadrant_"
    cols = [col for col in df.columns if col[: len(preffix)] == preffix]

    df_out = df[cols].drop_duplicates().reset_index(drop=True)
    df_out.columns = [col.replace(preffix, "") for col in cols]

    return df_out

def __rename_corresp_quadrants(df: pd.DataFrame, df_uops: pd.DataFrame) -> pd.DataFrame:

    logger.info("Pre-process (quadrants) - Renomeando os quadrantes correspondentes.")

    df_corresp = __find_corresp_quadrant(df, df_uops)
    is_corresp_quadrant = ~df_corresp["quadrant_name"].isna()

    df_to_rename = df_corresp[is_corresp_quadrant][["uop", "quadrant_name"]].rename(
        columns={"uop": "uop_name", "quadrant_name": "name"}
    )

    df = df.merge(df_to_rename, how="left", on="name")
    df["name"] = df["uop_name"].combine_first(df["name"])

    return df

def __find_corresp_quadrant(df: pd.DataFrame, df_uops: pd.DataFrame) -> pd.DataFrame:
    """Função para encontrar o quadrante correspondente para cada UOP.

    Parameters
    ----------
    df : pd.DataFrame
        Base de quadrantes.

    df_uops : pd.DataFrame
        Base de UOPs.

    Returns
    -------
    pd.DataFrame
        Base de UOPs com os quadrantes correspondentes encontrados.
    """

    logger.info("Pre-process (quadrants) - Encontrando o quadrante correspondente para cada UOP.")

    # Calcula a matriz de distâncias entre as UOPs e os quadrantes:
    dist_matrix = get_distance_matrix(
        df["latitude"], df["longitude"], df_uops["latitude"], df_uops["longitude"]
    )

    # Transforma a matriz em data frame:
    df_dist = pd.DataFrame(dist_matrix)
    df_dist.index = df["name"]
    df_dist.columns = df_uops["uop"]

    # Encontra o quadrante correspondente para cada UOP:
    names = []
    for col in df_uops["uop"]:
        df_sort = df_dist[col].sort_values().head(1).copy()
        idx = df_sort.index[0]
        dist = df_sort[idx]

        if dist <= MIN_DIST_TOLERANCE:
            names.append(idx)
        else:
            names.append(np.nan)
    df_uops["quadrant_name"] = names

    return df_uops

def __add_only_uops(df: pd.DataFrame, df_uops: pd.DataFrame) -> pd.DataFrame:

    logger.info("Adicionando as UOPs sem registro de acidentes.")

    df_corresp = __find_corresp_quadrant(df, df_uops)
    df_to_add = __get_only_uops(df, df_corresp)

    # Ordena pelo nome do ponto e concatena:
    df = df.sort_values(by=["municipality", "name"])
    df_to_add = df_to_add.sort_values(by=["municipality", "name"])
    df = pd.concat([df, df_to_add], ignore_index=True)

    return df

def __get_only_uops(df: pd.DataFrame, df_uops: pd.DataFrame) -> pd.DataFrame:
    """Função para preparar a base de UOPs (only) para adicionar na base de quadrantes.

    Parameters
    ----------
    df : pd.DataFrame
        Base de quadrantes.
    df_uops : pd.DataFrame
        Base de UOPs.

    Returns
    -------
    pd.DataFrame
        Base de UOPs (only) para adicionar na base de quadrantes.
    """
    cols = ["latitude", "longitude", "municipality", "uop"]
    df_only_uops = df_uops[df_uops["quadrant_name"].isna()][cols].rename(
        columns={"uop": "name"}
    )
    if df_only_uops.shape[0] > 0:
        cols_to_add = [col for col in df.columns if col not in df_only_uops.columns]
        for col in cols_to_add:
            df_only_uops[col] = np.nan
            if col in ["is_uop", "is_only_uop"]:
                df_only_uops[col] = True

    return df_only_uops
