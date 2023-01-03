from ppgmne_prf.load_data import load_data
from ppgmne_prf.optim import (
    get_abstract_model,
    get_fixed_params,
    get_instance,
    get_solution_data,
    solve_instance,
)
from ppgmne_prf.preprocess import preprocess


def main():
    df_accidents, dict_stations = load_data()
    df_quadrants = preprocess(df_accidents, dict_stations)

    # Modelo
    dict_params = get_fixed_params(df_quadrants)

    # Cria o modelo abstrato:
    model = get_abstract_model(
        a=dict_params["a"],
        u=dict_params["u"],
        s=dict_params["s"],
        dist_max=dict_params["dist_max"],
        dist_matrix=dict_params["dist_matrix"],
        accidents_hist=dict_params["accidents_hist"],
    )

    memory_p = [True for x in range(1,81)]
    for q in range(34):

        # for p in range(1,81):
        for p in range(29,81):

            if p >= q and memory_p[p-1]:

                # Cria a instância:
                instance = get_instance(model, p=p, q=q)

                # Resolve o modelo:
                instance, is_feasible = solve_instance(instance)

                # Atualiza a memória:
                memory_p[p-1] = is_feasible

                # Extrai os dados da soluçao:
                if is_feasible:
                    df_sol = get_solution_data(instance, df_quadrants)
    
if __name__ == "__main__":
    main()
