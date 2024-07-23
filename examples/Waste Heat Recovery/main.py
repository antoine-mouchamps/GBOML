from gboml import GbomlGraph
import json as js
import os


# heat_recovery = False
# no_whr_rpath = r"\data\remote_hub_wacc.txt"
# whr_rpath = r"\data\remote_hub_wacc_whr.txt"
# path_read  =  whr_rpath if heat_recovery else no_whr_rpath


# no_whr_wpath = r"\data\out_RREHA_1y"
# whr_wpath = r"\data\out_RREHA_HRSG_1y"
# path_write =  whr_wpath if heat_recovery else no_whr_wpath 


path_read  =  r"\data\remote_hub_wacc.txt"
path_write =  r"\results\GBOML\RREHA_ref_5y"
path = os.path.abspath(os.path.dirname(__file__))

gboml_model = GbomlGraph(24*365*5)
nodes, edges,global_parameters = gboml_model.import_all_nodes_and_edges(path+path_read)

gboml_model.add_global_parameters(global_parameters)
gboml_model.add_nodes_in_model(*nodes)
gboml_model.add_hyperedges_in_model(*edges)
gboml_model.build_model()
solution, objective, status, solver_info, _, _ = gboml_model.solve_gurobi()
solution_dict = gboml_model.turn_solution_to_dictionary(solver_info, status, solution, objective)


with open(path+path_write, 'w') as fp:
    js.dump(solution_dict, fp)