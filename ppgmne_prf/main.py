from loguru import logger

from ppgmne_prf.load_data import load_accidents, load_stations
from ppgmne_prf.preprocess import (
    preprocess_accidents,
    preprocess_quadrants,
    preprocess_stations,
)


def main():
    ########## Load data  ##########
    logger.info("Load data - Início do carregamento os dados de entrada.")

    logger.info("Load data (accidents) - Carregando os dados históricos dos acidentes.")
    df_accidents = load_accidents()

    logger.info("Load data (stations) - Carregando as coordenadas das UOPs e delegacias.")
    stations_coords = load_stations()

    logger.info("Load data - Fim do carregamento os dados de entrada.")

    ########## Pre-process  ##########
    logger.info("Pre-process - Início do pré-processamento dos dados de entrada.")

    logger.info("Pre-process (accidents) - Início do pré-processamento dos dados dos acidentes.")
    df_accidents = preprocess_accidents(df_accidents)

    logger.info("Pre-process (stations) - Início do pré-processamento dos dados das estações policiais.")
    df_stations = preprocess_stations(stations_coords)

    logger.info("Pre-process (quadrants) - Início do pré-processamento dos dados dos quadrantes.")
    df_quadrants = preprocess_quadrants(df_accidents, df_stations)

    logger.info("Pre-process - Fim do pré-processamento dos dados de entrada.")
    

if __name__ == "__main__":
    main()
