import gzip
import json
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from loguru import logger
from pyomo.opt import TerminationCondition

from ppgmne_prf.config.paths import (PATH_DATA_SOLVER_DETAILS,
                                     PATH_DATA_SOLVER_RESULTS)
from ppgmne_prf.utils import get_distance_matrix


def get_fixed_params(df: pd.DataFrame) -> dict:
    """Função para concentrar os parâmetros fixos do modelo em um dicionário.

    Parameters
    ----------
    df : pd.DataFrame
        Base de quadrantes.

    Returns
    -------
    dict
        Parâmetros fixos do modelo.
    """
    logger.info("Optimizer - Obtendo os parâmetros fixos do modelo.")

    # Matriz de distâncias:
    lat_rows = df[~df["is_only_uop"]]["latitude"]
    lon_rows = df[~df["is_only_uop"]]["longitude"]
    lat_cols = df["latitude"]
    lon_cols = df["longitude"]
    d = get_distance_matrix(lat_rows, lon_rows, lat_cols, lon_cols)

    # Demais parâmetros:
    dmax = np.array(df[~df["is_only_uop"]]["dist_max"])
    h = np.array(df[~df["is_only_uop"]]["n_accidents"])
    a = df[~df["is_uop"]].shape[0]
    u = a + df[(df["is_uop"]) & (~df["is_only_uop"])].shape[0]
    s = u + df[df["is_only_uop"]].shape[0]

    out = {}
    out["dist_matrix"] = d
    out["dist_max"] = dmax
    out["accidents_hist"] = h
    out["a"] = a
    out["u"] = u
    out["s"] = s

    return out


def get_abstract_model(
    a: int,
    u: int,
    s: int,
    dist_max: np.ndarray,
    dist_matrix: np.ndarray,
    accidents_hist: np.ndarray,
) -> pyo.AbstractModel:
    """Função para gerar um modelo de p-medianas parcialmente abstrato.

    Parameters
    ----------
    a : int
        Número de pontos com acidentes e sem UOP atual.
    u : int
        Número de pontos com acidentes e com UOP atual.
    s : int
        Número de pontos sem acidentes e com UOP atual.
    dist_max : np.ndarray
        Vetor com as distâncias máximas.
    dist_matrix : np.ndarray
        Matriz de distâncias m x n.
    accidents_hist : np.ndarray
        Vetor de histórico de acidentes.

    Returns
    -------
    pyo.AbstractModel
        Modelo parcialmente abstrato.
    """

    dmax = dist_max
    d = dist_matrix
    h = accidents_hist

    logger.info("Optimizer - Gerando o modelo de p-medianas.")
    model = pyo.AbstractModel()

    logger.info("Optimizer - Declarando os índices.")
    model.I = pyo.RangeSet(u)
    model.J = pyo.RangeSet(s)
    model.U = pyo.RangeSet(a, u)

    logger.info("Optimizer - Declarando os parâmetros.")
    model.p = pyo.Param()
    model.q = pyo.Param()
    model.d = pyo.Param(
        model.I, model.J, initialize=lambda model, i, j: d[i - 1][j - 1], mutable=True
    )
    model.h = pyo.Param(model.I, initialize=lambda model, i: h[i - 1])
    model.dmax = pyo.Param(model.I, initialize=lambda model, i: dmax[i - 1])

    logger.info("Optimizer - Declarando as variáveis de decisão.")
    model.y = pyo.Var(model.J, within=pyo.Binary)
    model.x = pyo.Var(model.I, model.J, within=pyo.Binary)

    logger.info("Optimizer - Declarando a função objetivo.")

    def f_obj(model):
        return sum(
            model.h[i] * model.d[i, j] * model.x[i, j] for i in model.I for j in model.J
        )

    model.z = pyo.Objective(rule=f_obj, sense=pyo.minimize)

    logger.info("Optimizer - Declarando as restrições.")

    def f_restr1(model, i):
        return sum(model.x[i, j] for j in model.J) == 1

    def f_restr2(model):
        return sum(model.y[j] for j in model.J) == model.p()

    def f_restr3(model):
        return sum(model.y[u] for u in model.U) >= model.q()

    def f_restr4(model, i, j):
        return model.x[i, j] <= model.y[j]

    def f_restr5(model, i, j):
        return (model.d[i, j] * model.x[i, j]) <= model.dmax[i]

    model.restr_1 = pyo.Constraint(model.I, rule=f_restr1)
    model.restr_2 = pyo.Constraint(rule=f_restr2)
    model.restr_3 = pyo.Constraint(rule=f_restr3)
    model.restr_4 = pyo.Constraint(model.I, model.J, rule=f_restr4)
    model.restr_5 = pyo.Constraint(model.I, model.J, rule=f_restr5)

    logger.info("Optimizer - Modelo de p-medianas gerado com sucesso.")

    return model


def get_instance(model: pyo.AbstractModel, p: int, q: int = 0) -> pyo.ConcreteModel:
    """Função para gerar uma instância do modelo de p-Medianas.

    Parameters
    ----------
    model : pyo.AbstractModel
        _description_
    p : int
        Número de medianas
    q : int, optional
        Número mínimo de UOPs atuais a serem mantidas na solução, by default 0

    Returns
    -------
    pyo.ConcreteModel
        Instância do modelo de p-Medianas.
    """
    logger.info(
        f"Optimizer - Obtendo instância p = {p}, q = {q} do modelo de p-medianas."
    )

    params = __format_params({"p": p, "q": q})
    instance = model.create_instance(params)

    return instance


def solve_instance(
    instance: pyo.ConcreteModel, solver: str = "gurobi"
) -> Tuple[pyo.ConcreteModel, bool]:
    logger.info(f"Optimizer - Resolvendo a instância via {solver}.")

    p = instance.p()
    q = instance.q()
    model_name = f"model_p{p}_q{q}_results"
    logger.info(f"Optimizer - Obtendo os resultados da instância p = {p}, q = {q}.")
    result = pyo.SolverFactory(solver).solve(instance)

    is_feasible = (
        not result.solver.termination_condition == TerminationCondition.infeasible
    )
    is_optimal = result.solver.termination_condition == TerminationCondition.optimal

    logger.info(f"Optimizer - Solução factível: {is_feasible}")
    logger.info(f"Optimizer - Solução ótima: {is_optimal}")

    obj_function = 0
    if is_feasible:
        obj_function = instance.z()

    logger.info(f"Optimizer - Função objetivo: {obj_function}")

    output_file = {}
    output_file["name"] = model_name
    output_file["p"] = p
    output_file["q"] = q
    output_file["solver"] = solver
    output_file["is_feasible"] = is_feasible
    output_file["is_optimal"] = is_optimal
    output_file["obj_function"] = obj_function
    with open(PATH_DATA_SOLVER_RESULTS / f"{model_name}.json", "w") as f:
        json.dump(output_file, f, indent=4)

    return instance, is_feasible


def get_solution_data(instance: pyo.ConcreteModel, df: pd.DataFrame) -> pd.DataFrame:
    """Função para extrair os resultados do modelo.

    Parameters
    ----------
    instance : pyo.ConcreteModel
        Modelo.
    df : pd.DataFrame
        Base de quadrantes.

    Returns
    -------
    pd.DataFrame
        Base de quadrantes somente com os pontos de demanda e com os resultados do modelo.
    """

    logger.info("Optimizer - Obtendo os dados da solução.")

    # Adiciona a flag identificadora se o ponto é uma mediana:
    df["is_median"] = [int(instance.y[j]()) == 1 for j in list(instance.y.keys())]

    # Separa em dados de medianas e dados de demanda:
    df_demand = df[~df["is_only_uop"]].copy()

    df_median = df[df["is_median"]][["name", "latitude", "longitude"]].copy()
    df_median.rename(
        columns={
            "name": "median_name",
            "latitude": "median_lat",
            "longitude": "median_lon",
        },
        inplace=True,
    )

    # Adiciona os resultados no dataset de demanda:
    aloc_tuple = [x for x in list(instance.x.keys()) if instance.x[x]() == 1]
    aloc_tuple.sort(key=lambda x: x[0])
    df_demand["median_name"] = [df_demand["name"][tupla[1] - 1] for tupla in aloc_tuple]
    df_demand["distance_q_to_m"] = [instance.d[x[0], x[1]]() for x in aloc_tuple]
    df_demand["obj_function"] = df_demand["n_accidents"] * df_demand["distance_q_to_m"]

    # Adiciona as coordenadas das medianas:
    df_demand = df_demand.merge(df_median, on=["median_name"])
    df_demand.sort_values(by=["municipality", "name"], inplace=True)

    # Persiste os dados:
    p = instance.p()
    q = instance.q()
    model_name = f"model_p{p}_q{q}_details"

    with gzip.open(PATH_DATA_SOLVER_DETAILS / f"{model_name}.gz", "wb") as f:
        pickle.dump(df_demand, f)

    return df_demand


def __format_params(params: dict) -> pyo.DataPortal:
    """Função para formatar um dicionário de parâmetros para criação de instâncias do modelo.

    Parameters
    ----------
    params : dict
        Dicionário de parâmetros.

    Returns
    -------
    pyo.DataPortal
        Parâmetros formatados.
    """

    def format_param(param):
        if isinstance(param, list):
            if isinstance(param[0], list):
                param = [tuple(row) for row in param]
        return {None: param}

    out = pyo.DataPortal()
    for key in params:
        out[key] = format_param(params[key])

    return out
