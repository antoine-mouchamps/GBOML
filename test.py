# Copyright (C) 2020 - 2022
# Bardhyl Miftari, Mathias Berger, Hatim Djelassi, Damien Ernst,
# University of Liege .
# Licensed under the MIT License (see LICENSE file).

"""GBOML compiler test file

Unittesting of different interfaces to GBOML and the implementation

  Typical usage example:

   $ python test.py
"""

from gboml import GbomlGraph, VariableType
from gboml.compiler.classes import Parameter, Expression, Node, Hyperedge,\
    Time, Program
from gboml.compiler import has_quadratic_terms
from gboml.compiler import compile_gboml
from gboml.solver_api import gurobi_solver, cplex_solver, xpress_solver, \
    highs_solver

import unittest
import subprocess
import json
import numpy as np
import os

PATH = os.getcwd()


class CompilerTests(unittest.TestCase):
    """
    CompilerTests

    Tests mainly the gboml/main.py, compile_gboml functions, solver API and
    output types.

    It checks whether a file correctly fails, generates the correct matrices,
    correct solutions and correct output.

    """

    def test_nonexistant_file(self):
        """ test_nonexistant_file

        Tests compiling a file that does not exist.

        """
        process = subprocess.run(['gboml', 'test/nonexistant.txt'],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertNotEqual(return_code, 0)

    def test_empty_file(self):
        """ test_empty_file

        Tests a file which is completely empty.

        """
        process = subprocess.run(['gboml', 'test/empty.txt'],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertNotEqual(return_code, 0)

    def test_timehorizon_only(self):
        """ test_timehorizon_only

        Tests a file with only a timehorizon defined (no nodes, no hyperedges).

        """

        process = subprocess.run(['gboml', 'test/test1.txt'],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertNotEqual(return_code, 0)

    def test_wrong_horizon(self):
        """ test_wrong_horizon

        Tests a file with a wrong name for T the timehorizon.

        """

        process = subprocess.run(['gboml', 'test/test2.txt'],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertNotEqual(return_code, 0)

    def test_negative_horizon(self):
        """ test_negative_horizon

        Tests a file with a negative timehorizon

        """

        process = subprocess.run(['gboml', 'test/test3.txt'],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertNotEqual(return_code, 0)

    def test_impossible_constraint(self):
        """ test_impossible_constraint

        Tests a file with a constraint that is impossible (0*x==10)

        """

        process = subprocess.run(['gboml', 'test/test4.txt'],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertNotEqual(return_code, 0)

    def test_unfeasible_problem_gurobi(self):
        """ test_unfeasible_problem_gurobi

        Tests the gurobi interface for an impossible problem

        """

        process = subprocess.run(['gboml', 'test/test5.txt', "--gurobi"],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

    def test_unfeasible_problem_cplex(self):
        """ test_unfeasible_problem_cplex

        Tests the cplex interface for an impossible problem

        """

        process = subprocess.run(['gboml', 'test/test5.txt', "--cplex"],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

    def test_unfeasible_problem_clp(self):
        """ test_unfeasible_problem_clp

        Tests the clp-cbc interface for an impossible problem

        """

        process = subprocess.run(['gboml', 'test/test5.txt', "--clp"],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

    def test_unfeasible_problem_linprog(self):
        """ test_unfeasible_problem_linprog

        Tests the linprog interface for an impossible problem

        """
        process = subprocess.run(['gboml', 'test/test5.txt', "--linprog"],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

    def test_unfeasible_problem_xpress(self):
        """ test_unfeasible_problem_xpress

        Tests the xpress interface for an impossible problem

        """
        process = subprocess.run(['gboml', 'test/test5.txt', "--xpress"],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

    def test_positive_solution_gurobi(self):
        """ test_positive_solution_gurobi

        Tests a file having a positive solution with gurobi from the command
        line

        """

        process = subprocess.run(
            ['gboml', 'test/test6.txt', "--gurobi", "--json",
             "--output", "test/test6"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

        with open("test/test6.json", 'r') as j:
            contents = json.loads(j.read())
            self.assertIn("objective", contents["solution"])
            self.assertEqual(contents["solution"]["objective"], 40.0)

    def test_positive_solution_cplex(self):
        """ test_positive_solution_cplex

        Tests a file having a positive solution with cplex from the command line

        """

        process = subprocess.run(
            ['gboml', 'test/test6.txt', "--cplex", "--json", "--output",
             "test/test6"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

        with open("test/test6.json", 'r') as j:
            contents = json.loads(j.read())
            self.assertIn("objective", contents["solution"])
            self.assertEqual(contents["solution"]["objective"], 40.0)

    def test_positive_solution_clp(self):
        """ test_positive_solution_clp

        Tests a file having a positive solution with clp from the command line

        """
        process = subprocess.run(
            ['gboml', 'test/test6.txt', "--clp", "--json", "--output",
             "test/test6"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

        with open("test/test6.json", 'r') as j:
            contents = json.loads(j.read())
            self.assertIn("objective", contents["solution"])
            self.assertEqual(contents["solution"]["objective"], 40.0)

    def test_positive_solution_linprog(self):
        """ test_positive_solution_linprog

        Tests a file having a positive solution with linprog from the command
        line

        """
        process = subprocess.run(
            ['gboml', 'test/test6.txt', "--linprog", "--json",
             "--output", "test/test6"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

        with open("test/test6.json", 'r') as j:
            contents = json.loads(j.read())
            self.assertIn("objective", contents["solution"])
            self.assertEqual(contents["solution"]["objective"], 40.0)

    def test_negative_solution_gurobi(self):
        """ test_negative_solution_gurobi

        Tests a file having a negative solution with gurobi from the command
        line

        """
        process = subprocess.run(
            ['gboml', 'test/test7.txt', "--gurobi", "--json",
             "--output", "test/test7"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

        with open("test/test7.json", 'r') as j:
            contents = json.loads(j.read())
            self.assertIn("objective", contents["solution"])
            self.assertEqual(contents["solution"]["objective"], 4.0)

    def test_negative_solution_cplex(self):
        """ test_negative_solution_cplex

        Tests a file having a negative solution with cplex from the command line

        """
        process = subprocess.run(
            ['gboml', 'test/test7.txt', "--cplex", "--json", "--output",
             "test/test7"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

        with open("test/test7.json", 'r') as j:
            contents = json.loads(j.read())
            self.assertIn("objective", contents["solution"])
            self.assertEqual(contents["solution"]["objective"], 4.0)

    def test_negative_solution_clp(self):
        """ test_negative_solution_clp

        Tests a file having a negative solution with clp from the command line

        """
        process = subprocess.run(
            ['gboml', 'test/test7.txt', "--clp", "--json", "--output",
             "test/test7"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

        with open("test/test7.json", 'r') as j:
            contents = json.loads(j.read())
            self.assertIn("objective", contents["solution"])
            self.assertEqual(contents["solution"]["objective"], 4.0)

    def test_negative_solution_linprog(self):
        """ test_negative_solution_linprog

        Tests a file having a negative solution with linprog from the command
        line

        """
        process = subprocess.run(
            ['gboml', 'test/test7.txt', "--linprog", "--json",
             "--output", "test/test7"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

        with open("test/test7.json", 'r') as j:
            contents = json.loads(j.read())
            self.assertIn("objective", contents["solution"])
            self.assertEqual(contents["solution"]["objective"], 4.0)

    def test_non_linearity(self):
        """ test_non_linearity

        Tests a file containing a non-linearity

        """
        process = subprocess.run(['gboml', 'test/test8.txt'],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertNotEqual(return_code, 0)

    def test_non_existant_import(self):
        """ test_non_existant_import

        Tests a file having an import toward a non existing file

        """
        process = subprocess.run(['gboml', 'test/test9.txt'],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertNotEqual(return_code, 0)

    def test_non_linear_assignment_division(self):
        """ test_non_linear_assignment_division

        Tests a file with a non linear assignment

        """
        process = subprocess.run(['gboml', 'test/test10.txt'],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertNotEqual(return_code, 0)

    def test_normal_file(self):
        """ test_normal_file

        Tests a simple file with a non linear assignment and checks if
        additional prints have been added.

        """
        process = subprocess.run(['gboml', 'test/test11.txt'],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)
        if process.stdout.count("\n") != 3:
            print("Please remove the additional prints before you push")

    def test_assignment_T(self):
        """ test_normal_file

        Tests a file with one vector variable with only one value fixed.

        """
        process = subprocess.run(
            ['gboml', 'test/test12.txt', "--gurobi", "--json",
             "--output", "test/test12"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

    def test_an_assignment_reads_the_node_parameters_on_both_sides(self):
        """ test_an_assignment_reads_the_node_parameters_on_both_sides

        Only the left of an assignment had its bare names given the node
        holding it, so a parameter of that node read on the right was refused
        unless it was written qualified. Same declaration, same scope.

        """
        process = subprocess.run(
            ['gboml', 'test/test95_assignment_reads_a_node_parameter.txt',
             "--linprog", "--json", "--output",
             "test/test95_assignment_reads_a_node_parameter"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        self.assertEqual(process.returncode, 0)

        with open("test/test95_assignment_reads_a_node_parameter.json",
                  'r') as j:
            variables = json.loads(
                j.read())["solution"]["elements"]["REPRO"]["variables"]

        self.assertEqual(variables["y"]["values"], [3.0, 3.0])

    def test_single_obj_constr(self):
        """ test_single_obj_constr

        Tests the number of constraints and objectives expanded

        """
        with open("test/test12.json", 'r') as j:
            contents = json.loads(j.read())
            solution = contents['model']
            self.assertIn("nodes", solution)
            nodes = solution["nodes"]
            self.assertIn("A", nodes)
            node_a = nodes["A"]
            self.assertIn("number_expanded_constraints", node_a)
            nb_constr_derived = node_a["number_expanded_constraints"]
            self.assertIn("number_expanded_objectives", node_a)
            nb_obj_derived = node_a["number_expanded_objectives"]
            self.assertEqual(nb_constr_derived, 1)
            self.assertEqual(nb_obj_derived, 1)

    def test_precise_where(self):
        """ test_precise_where

        Tests the where keyword for constraints

        """
        process = subprocess.run(['gboml', 'test/test13.txt', "--matrix"],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        output_split = process.stdout.split("\n")
        self.assertEqual(output_split[4], 'Matrix A_ineq    (0, 9)\t-1.0')
        self.assertEqual(output_split[5], 'Vector b_ineq  [-0.]')
        self.assertEqual(output_split[6],
                         "Vector C  [[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]")
        return_code = process.returncode
        self.assertEqual(return_code, 0)

    # for t in [9:10]
    def test_precise_for(self):
        """ test_precise_for

        Tests a for range for a constraint

        """
        process = subprocess.run(['gboml', 'test/test14.txt', "--matrix"],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        output_split = process.stdout.split("\n")
        self.assertEqual(output_split[4], 'Matrix A_ineq    (0, 9)\t-1.0')
        self.assertEqual(output_split[5], 'Vector b_ineq  [-0.]')
        self.assertEqual(output_split[6],
                         "Vector C  [[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]")
        return_code = process.returncode
        self.assertEqual(return_code, 0)

    def test_links(self):
        """ test_links

        Tests the linking between two nodes

        """
        process = subprocess.run(
            ['gboml', 'test/test15.txt', "--gurobi", "--json",
             "--output", "test/test15"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)
        with open("test/test15.json", 'r') as j:
            contents = json.loads(j.read())
            links = [[["A", "b"], ["R", "f"]],
                     [["A", "b"], ["R", "ff"]],
                     [["R", "h"], ["A", "a"]],
                     [["A", "c"], ["R", "d"]]]
            solution = contents["solution"]
            for [lhs_node, lhs_var], [rhs_node, rhs_var] in links:
                all_nodes = solution["elements"]
                self.assertIn(lhs_node, all_nodes)
                self.assertIn(rhs_node, all_nodes)

                lhs_node_dict = all_nodes[lhs_node]
                rhs_node_dict = all_nodes[rhs_node]

                lhs_node_variables = lhs_node_dict["variables"]
                rhs_node_variables = rhs_node_dict["variables"]

                self.assertIn(lhs_var, lhs_node_variables)
                self.assertIn(rhs_var, rhs_node_variables)

                self.assertTrue(
                    lhs_node_variables[lhs_var] == rhs_node_variables[rhs_var])

    def test_time_dependency_in_obj(self):
        """ test_time_dependancy_in_obj

        Tests the time dependancy of an objective

        """
        process = subprocess.run(['gboml', 'test/test16.txt', "--matrix"],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        output_split = process.stdout.split("\n")
        self.assertEqual(output_split[11], "Vector C  [[6. 0. 0. 1. 0. 0.]]")
        return_code = process.returncode
        self.assertEqual(return_code, 0)

    def test_wrong_variable_def(self):
        """ test_wrong_variable_def

        Tests a simple file with a variable wrongly defined

        """
        process = subprocess.run(['gboml', 'test/test17.txt'],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertNotEqual(return_code, 0)

    def test_array_in_for(self):
        """ test_array_in_for

        Tests a simple file with an array in for

        """
        process = subprocess.run(['gboml', 'test/test_33.txt'],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

    def test_sum_objective(self):
        """ test_sum_objective

        Tests a sum for defining an objective

        """
        process = subprocess.run(['gboml', 'test/test18.txt', "--matrix"],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        output_split = process.stdout.split("\n")
        self.assertEqual(output_split[15],
                         "Vector C  [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. "
                         "0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]")
        return_code = process.returncode
        self.assertEqual(return_code, 0)

    def test_bigger_variable(self):
        """ test_bigger_Variable

        Tests a variable whose size exceeds the timehorizon

        """
        process = subprocess.run(
            ['gboml', 'test/test19.txt', "--gurobi", "--json",
             "--output", "test/test19"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)
        with open("test/test19.json", 'r') as j:
            contents = json.loads(j.read())
            solution = contents["solution"]
            obj = solution["objective"]
            self.assertEqual(obj, 10.0)

    def test_hyperlink(self):
        """ test_hyperlink

        Tests a hyperlink with parameters and that it is correctly converted
        to matrices

        """

        process = subprocess.run(
            ['gboml', 'test/test20.txt', "--gurobi", "--json",
             "--output", "test/test20"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)
        with open("test/test20.json", 'r') as j:
            contents = json.loads(j.read())
            solution = contents["solution"]
            nodes = solution["elements"]
            node_a = nodes["A"]
            node_b = nodes['B']
            ax = node_a["variables"]["x"]["values"]
            by = node_b["variables"]["y"]["values"]
            bw = node_b["variables"]["w"]["values"]

            for i in range(6):
                self.assertEqual(ax[i], by[i] + bw[i])

    def test_globalkey(self):
        """ test_globalkey

        Tests that the global parameters are present through-out the file

        """
        process = subprocess.run(['gboml', 'test/test21.txt', "--gurobi",
                                  "--json", "--output", "test/test21"],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

        with open("test/test21.json", 'r') as j:
            contents = json.loads(j.read())
            model = contents["model"]
            nodes = model["nodes"]

            node_o = nodes["O"]
            parameters_o = node_o["parameters"]
            p1 = parameters_o["p1"][0]
            p2 = parameters_o["p2"][0]
            self.assertEqual(p1, 2)
            self.assertEqual(p2, 3)

            solution = contents["solution"]
            obj = solution["objective"]
            self.assertEqual(obj, 5.0)

    def test_encapsulation(self):
        """ test_encapsulation

        Tests a node defined inside another node

        """
        global PATH
        os.chdir(PATH)

        (_, matrix_eq, vector_b_eq, quadratic_eq, matrix_ineq, vector_b_ineq,
         quadratic_ineq, vector_c, indep_terms_c, quadratic_objective,
         alone_term_c,
         time_horizon, name_tuples) = compile_gboml("test/test23.txt")
        objective_offset = float(indep_terms_c.sum())
        c_sum = np.asarray(vector_c.sum(axis=0), dtype=float)
        self.assertEqual(objective_offset, 0)
        self.assertTrue(np.allclose(c_sum, np.array([1., 1., 0., 0., 0., 0., 0., 0., 0., 0.])))
        self.assertTrue(np.allclose(vector_b_eq,
                                    np.array([120., 0., 0., 0., 6.9, 6.4])))
        expected_matrix_a = [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., -1., 1., -0.75, 0., 1.33333333, 0., 0., 0.],
                             [0., 0., 1., -1., 0., 0., 0., 0., 0., 0.],
                             [-3.14, 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
        self.assertTrue(((matrix_eq.todense() - np.array(expected_matrix_a)) <= 0.1).all())
        x, objective, _, _, _, _ = gurobi_solver(matrix_eq, vector_b_eq, quadratic_eq,
                                                 matrix_ineq, vector_b_ineq, quadratic_ineq,
                                                 c_sum, objective_offset, quadratic_objective,
                                                 name_tuples)
        self.assertTrue((np.array([120., 376.8, 0., 0., 0., 0., 0., 0., 6.9, 6.4]) == x).all())
        self.assertEqual(objective, 496.8)

    def test_encapsulation_with_hyperedge(self):
        """ test_encapsulation_with_hyperedge

        Tests a node defined inside another node with a hyperedge

        """
        global PATH
        os.chdir(PATH)
        (_, matrix_eq, vector_b_eq, quadratic_eq, matrix_ineq, vector_b_ineq,
         quadratic_ineq, vector_c, indep_terms_c, quadratic_objective,
         alone_term_c,
         time_horizon, name_tuples) = compile_gboml("test/test24.txt")
        objective_offset = float(indep_terms_c.sum())
        c_sum = np.asarray(vector_c.sum(axis=0), dtype=float)
        x, objective, _, _, _, _ = gurobi_solver(matrix_eq, vector_b_eq, quadratic_eq,
                                                matrix_ineq, vector_b_ineq, quadratic_ineq,
                                                c_sum, objective_offset, quadratic_objective,
                                                name_tuples)
        self.assertTrue((np.array([120., 1080., 0., 0., 0., 0., 0., 0., 6.9, 6.4, 118]) == x).all())
        self.assertEqual(objective, 1200)

    def test_import_hyperedge(self):
        """ test_import_hyperedge

        Tests a hyperedge import

        """
        global PATH
        os.chdir(PATH)
        (_, matrix_eq, vector_b_eq, quadratic_eq, matrix_ineq, vector_b_ineq,
         quadratic_ineq, vector_c, indep_terms_c, quadratic_objective,
         alone_term_c,
         time_horizon, name_tuples) = compile_gboml("test/test25.txt")
        objective_offset = float(indep_terms_c.sum())+alone_term_c
        c_sum = np.asarray(vector_c.sum(axis=0), dtype=float)
        x, objective, _, _, _, _ = xpress_solver(matrix_eq, vector_b_eq, quadratic_eq,
                                                 matrix_ineq, vector_b_ineq, quadratic_ineq,
                                                 c_sum, objective_offset, quadratic_objective,
                                                 name_tuples)
        self.assertTrue((np.array([6., 5.]) == x).all())
        self.assertEqual(11, objective)

    def test_nested_nodes_accessing_to_one_variable_and_parent_parameters(self):
        """ test_nested_nodes_accessing_to_one_variable_and_parent_parameters

        Tests a series of nested nodes with only one variable defined

        """
        global PATH
        os.chdir(PATH)
        # program, matrix_eq, vector_b_eq, matrix_ineq, vector_b_ineq, vector_c, indep_terms_c, \
        #            alone_term_c, time_horizon, program.get_tuple_name()
        (_, matrix_eq, vector_b_eq, quadratic_eq, matrix_ineq, vector_b_ineq,
         quadratic_ineq, vector_c, indep_terms_c, quadratic_objective,
         alone_term_c,
         _, _) = compile_gboml("test/test26.txt")
        objective_offset = float(indep_terms_c.sum())+alone_term_c
        c_sum = np.asarray(vector_c.sum(axis=0), dtype=float)
        self.assertEqual(objective_offset, 1)
        self.assertTrue((np.array([[1.], [-1.]]) == matrix_ineq.todense()).all())
        self.assertEqual(c_sum.all(), np.array([1]))
        self.assertTrue((vector_b_ineq == np.array([6., -1.])).all())

    def test_nested_nodes_accessing_to_array_variable_and_parent_parameters(
            self):
        """ test_nested_nodes_accessing_to_array_variable_and_parent_parameters

        Tests a series of nested nodes with only one variable defined and
        parameters intertwined

        """
        global PATH
        os.chdir(PATH)
        (_, matrix_eq, vector_b_eq, quadratic_eq, matrix_ineq, vector_b_ineq,
         quadratic_ineq, vector_c, indep_terms_c, quadratic_objective,
         alone_term_c,
         _, name_tuples) = compile_gboml("test/test27.txt")
        objective_offset = float(indep_terms_c.sum())
        c_sum = np.asarray(vector_c.sum(axis=0), dtype=float)
        self.assertEqual(objective_offset, 0)
        solution = cplex_solver(matrix_eq, vector_b_eq, quadratic_eq,
                                matrix_ineq, vector_b_ineq, quadratic_ineq,
                                c_sum, objective_offset, quadratic_objective,
                                name_tuples)
        self.assertTrue((solution[0] == np.array([1., 1., 1., 1., 1.])).all())

    def test_nested_nodes_with_hyperedge_and_parent_parameters(self):
        """ test_nested_nodes_with_hyperedge_and_parent_parameters

        Tests a series of nested nodes with two low level nodes and a hyperedge

        """
        global PATH
        os.chdir(PATH)
        (_, matrix_a_eq, vector_b_eq, quadratic_eq, matrix_a_ineq,
         vector_b_ineq, quadratic_ineq, vector_c, indep_terms_c,
         quadratic_objective, alone_term_c,
         _, name_tuples) = compile_gboml("test/test28.txt")
        objective_offset = float(indep_terms_c.sum())
        c_sum = np.asarray(vector_c.sum(axis=0), dtype=float)
        expected_constraint_matrix_eq = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
        expected_constraint_matrix_ineq = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0],
                                                    [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])

        expected_vector_b_eq = np.array([10, 10])
        expected_vector_b_ineq = np.array([6, 6, -5.5, -5.5, -1, -1])
        expected_vector_c = np.array([[1, 1, 1, 1]])

        self.assertEqual(objective_offset, 0)
        self.assertTrue((expected_vector_c.all() == c_sum).all())
        self.assertTrue((expected_vector_b_eq == vector_b_eq).all())
        self.assertTrue((expected_vector_b_ineq == vector_b_ineq).all())
        self.assertTrue((matrix_a_eq.todense() == expected_constraint_matrix_eq).all())
        self.assertTrue((matrix_a_ineq.todense() == expected_constraint_matrix_ineq).all())

    def test_dual(self):
        """ test_dual

        Tests the dual value of a constraint

        """

        process = subprocess.run(
            ['gboml', 'test/test29_duals.txt', "--gurobi", "--json", "--detailed",
             "--output", "test/test29"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)
        with open("test/test29.json", 'r') as j:
            contents = json.loads(j.read())
            solution = contents["solution"]
            nodes = solution["elements"]
            node_system = nodes["system"]
            constraints = node_system["constraints"]
            test_constraint = constraints["test_constraint"]
            dual_value = test_constraint["Pi"]
            self.assertEqual(dual_value[0], 10.0)

    def test_import_five(self):
        """ test_import_five

        Tests import five times the same node and sum in hyperedge

        """

        process = subprocess.run(
            ['gboml', 'test/test30_import5.txt', "--gurobi", "--json",
             "--output", "test/test30"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)
        with open("test/test30.json", 'r') as j:
            contents = json.loads(j.read())
            solution = contents["solution"]
            nodes = solution["elements"]
            node_system = nodes["Demande"]
            variables = node_system["variables"]
            tot_score = variables["score_total"]["values"]
            self.assertEqual(tot_score[0], 0.6)

    def test_rename_vector_param(self):
        """ test_rename_vector_param

        Tests redefining a vector parameters

        """
        process = subprocess.run(
            ['gboml', 'test/test31_rename.txt'],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

    def test_rename_node(self):
        """ test_rename_node

        Tests redefining several nodes with import

        """
        process = subprocess.run(
            ['gboml', 'test/test_32_rename.txt'],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

    def test_quadratic_constraint(self):
        """ test_quadratic_constraint

        Tests a bilinear constraint written in a #CONSTRAINTS block. The
        problem minimises x*y subject to x+y == 6 and 1 <= x,y <= 5, whose
        optimum is 5 at both ends of the range, hence 15 over three time steps

        """
        process = subprocess.run(
            ['gboml', 'test/test34_quadratic.txt', "--gurobi", "--json",
             "--output", "test/test34_quadratic"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

        with open("test/test34_quadratic.json", 'r') as j:
            contents = json.loads(j.read())
            solution = contents["solution"]
            self.assertAlmostEqual(solution["objective"], 15.0, places=4)
            variables = solution["elements"]["A"]["variables"]
            for x_i, y_i, z_i in zip(variables["x"]["values"],
                                     variables["y"]["values"],
                                     variables["z"]["values"]):
                self.assertAlmostEqual(x_i + y_i, 6.0, places=4)
                self.assertAlmostEqual(z_i, x_i * y_i, places=4)

    def test_quadratic_sum(self):
        """ test_quadratic_sum

        Tests a sum of products of two variables, the range of the sum not
        depending on the index of the constraint

        """
        process = subprocess.run(
            ['gboml', 'test/test35_quadratic_sum.txt', "--gurobi", "--json",
             "--output", "test/test35_quadratic_sum"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

        with open("test/test35_quadratic_sum.json", 'r') as j:
            contents = json.loads(j.read())
            solution = contents["solution"]
            # 2*1*2 + 3*2*2 + 4*3*2 = 40 for each of the two time steps
            self.assertAlmostEqual(solution["objective"], 80.0, places=4)

    def test_quadratic_hyperedge(self):
        """ test_quadratic_hyperedge

        Tests a quadratic constraint defined in a hyperedge, coupling the
        variables of two nodes through a product

        """
        process = subprocess.run(
            ['gboml', 'test/test36_quadratic_hyperedge.txt', "--gurobi",
             "--json", "--output", "test/test36_quadratic_hyperedge"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

        with open("test/test36_quadratic_hyperedge.json", 'r') as j:
            contents = json.loads(j.read())
            solution = contents["solution"]
            # a == 3 and b == 4 give p == 12 for each of the two time steps
            self.assertAlmostEqual(solution["objective"], 30.0, places=4)
            nodes = solution["elements"]
            for p_i in nodes["B"]["variables"]["p"]["values"]:
                self.assertAlmostEqual(p_i, 12.0, places=4)

    def test_quadratic_sum_shadowing(self):
        """ test_quadratic_sum_shadowing

        Tests a sum whose index shadows the index of the constraint, the
        expressions outside of the sum having to keep being expanded over the
        index of the constraint

        """
        process = subprocess.run(
            ['gboml', 'test/test43_quadratic_sum_shadowing.txt', "--gurobi",
             "--json", "--output", "test/test43_quadratic_sum_shadowing"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

        with open("test/test43_quadratic_sum_shadowing.json", 'r') as j:
            contents = json.loads(j.read())
            solution = contents["solution"]
            # x is minimised to one, so both z equal x[0]**2 + x[1]**2 == 2
            for z_i in solution["elements"]["A"]["variables"]["z"]["values"]:
                self.assertAlmostEqual(z_i, 2.0, places=4)
            self.assertAlmostEqual(solution["objective"], 60.0, places=4)

    def test_quadratic_degree_three(self):
        """ test_quadratic_degree_three

        Tests a file with a product of three variables, which is of a degree
        higher than the highest one accepted

        """
        process = subprocess.run(['gboml',
                                  'test/test37_quadratic_degree_three.txt'],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertNotEqual(return_code, 0)

    def test_quadratic_division(self):
        """ test_quadratic_division

        Tests a file with a variable in the denominator of a division

        """
        process = subprocess.run(['gboml',
                                  'test/test38_quadratic_division.txt'],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertNotEqual(return_code, 0)

    def test_quadratic_sum_index_dependent_range(self):
        """ test_quadratic_sum_index_dependent_range

        Tests a quadratic constraint holding a sum whose range depends on the
        index of the constraint. It used to be rejected up front, the
        quadratic factorization not being able to unroll a ragged range, and
        compiles since the factorizer rewrite

        """
        process = subprocess.run(['gboml',
                                  'test/test39_quadratic_sum_range.txt',
                                  "--gurobi"],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

    def test_quadratic_sum_times_sum(self):
        """ test_quadratic_sum_times_sum

        Tests a product of two sums completed independently before being
        multiplied together, rather than a product taken inside one shared
        sum body. Each side folds to a term of width greater than one before
        the multiplication happens, which is the shape that exercises the
        general branch of block.py's _outer_product rather than the width one
        fast path every other quadratic sum test goes through : x sums to
        1+2+3 == 6, y sums to 4+5 == 9, so z, and the objective, is 54

        """
        process = subprocess.run(
            ['gboml', 'test/test52_quadratic_sum_times_sum.txt', "--gurobi",
             "--json", "--output", "test/test52_quadratic_sum_times_sum"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

        with open("test/test52_quadratic_sum_times_sum.json", 'r') as j:
            contents = json.loads(j.read())
            solution = contents["solution"]
            self.assertAlmostEqual(solution["objective"], 54.0, places=4)
            variables = solution["elements"]["A"]["variables"]
            self.assertEqual(variables["x"]["values"], [1.0, 2.0, 3.0])
            self.assertEqual(variables["y"]["values"], [4.0, 5.0])
            self.assertAlmostEqual(variables["z"]["values"][0], 54.0,
                                   places=4)

    def test_a_sum_body_which_ignores_its_index(self):
        """ test_a_sum_body_which_ignores_its_index

        Tests a sum whose body does not read the index of the sum. Every index
        value gives the very same entry, so block.py's Term.widened adds them
        up on the spot rather than writing one entry per index for the matrix
        to add up, which leaves the term one axis fewer than a widened one :
        the product below spreads such a collapsed term against a widened one
        in _outer_product, where getting the axes wrong is a wrong number and
        not a crash. x is [1, 2, 3] and y is [4, 5], so a == 4*x, b over a
        range which stops at the row == (t+1)*x, c under a condition keeping
        two index values == 2*x, e under a coefficient which varies along
        the index while the column does not == (1+2)*x, and d == 2*x * (4+5)

        """
        process = subprocess.run(
            ['gboml', 'test/test78_a_sum_body_which_ignores_its_index.txt', "--gurobi",
             "--json", "--output", "test/test78_a_sum_body_which_ignores_its_index"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

        with open("test/test78_a_sum_body_which_ignores_its_index.json", 'r') as j:
            contents = json.loads(j.read())
            solution = contents["solution"]
            self.assertAlmostEqual(solution["objective"], 108.0, places=4)
            variables = solution["elements"]["A"]["variables"]
            self.assertEqual(variables["x"]["values"], [1.0, 2.0, 3.0])
            self.assertEqual(variables["y"]["values"], [4.0, 5.0])
            self.assertEqual(variables["a"]["values"], [4.0, 8.0, 12.0])
            self.assertEqual(variables["b"]["values"], [1.0, 4.0, 9.0])
            self.assertEqual(variables["c"]["values"], [2.0, 4.0, 6.0])
            self.assertEqual(variables["e"]["values"], [3.0, 6.0, 9.0])
            self.assertEqual(variables["d"]["values"], [18.0, 36.0, 54.0])

    def test_quadratic_ragged_sum(self):
        """ test_quadratic_ragged_sum

        Tests a product whose operands are sums over a range depending on the
        index of the constraint. Each of them folds to a padded term, the rows
        of a ragged range being squared off to the longest one, so the padding
        of each operand has to spread over the entries of the other for the
        products that carry nothing to be dropped. Getting that wrong adds the
        padded products to the row rather than dropping them, which is a wrong
        number and not a crash, so the values are pinned : x is [1, 2, 3] and
        y is [4, 5], making z == [1, 9, 36] and w == [9, 27, 54]

        """
        process = subprocess.run(
            ['gboml', 'test/test57_quadratic_ragged_sum.txt', "--gurobi",
             "--json", "--output", "test/test57_quadratic_ragged_sum"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

        with open("test/test57_quadratic_ragged_sum.json", 'r') as j:
            contents = json.loads(j.read())
            solution = contents["solution"]
            self.assertAlmostEqual(solution["objective"], 46.0, places=4)
            variables = solution["elements"]["A"]["variables"]
            for got, expected in zip(variables["z"]["values"], [1.0, 9.0, 36.0]):
                self.assertAlmostEqual(got, expected, places=4)
            for got, expected in zip(variables["w"]["values"], [9.0, 27.0, 54.0]):
                self.assertAlmostEqual(got, expected, places=4)

    def test_quadratic_unbounded_variable(self):
        """ test_quadratic_unbounded_variable

        Tests a file where a variable involved in a product of two variables
        has no upper bound. Whether a bound is needed at all depends on the
        shape of the quadratic form, which the solver classifies exactly and
        GBOML does not classify at all, so the model compiles and the question
        is left to the solver

        """
        gboml_model = GbomlGraph(2)
        nodes, edges, global_params = \
            gboml_model.import_all_nodes_and_edges(
                "test/test40_quadratic_unbounded.txt")
        gboml_model.add_nodes_in_model(*nodes)
        gboml_model.add_hyperedges_in_model(*edges)
        gboml_model.add_global_parameters(global_params)
        gboml_model.build_model()

        self.assertTrue(has_quadratic_terms(
            gboml_model.quadratic_eq, gboml_model.quadratic_ineq,
            gboml_model.quadratic_objective))

    def test_quadratic_positive_semi_definite_form(self):
        """ test_quadratic_positive_semi_definite_form

        Tests a form which is positive semi-definite but written with cross
        terms, (x - y)**2, over unbounded variables. It needs no bound at all
        and used to be refused by a convexity test of our own which only ever
        recognised a plain sum of squares

        """
        process = subprocess.run(
            ['gboml', 'test/test55_quadratic_psd.txt', "--gurobi"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        self.assertEqual(process.returncode, 0)

    def test_quadratic_bounded_variable(self):
        """ test_quadratic_bounded_variable

        Tests that a product of two variables which are bounded on both sides
        is solved through, the shape a spatial branching needs

        """
        process = subprocess.run(
            ['gboml', 'test/test42_quadratic_bounded.txt', "--gurobi"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

    def test_non_quadratic_in_constraints_block(self):
        """ test_non_quadratic_in_constraints_block

        Tests that a non linearity which is not a product of two variables,
        here a variable in an exponent, is still rejected

        """
        process = subprocess.run(['gboml',
                                  'test/test41_non_quadratic_constraints.txt'],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertNotEqual(return_code, 0)

    def test_quadratic_wrong_solver(self):
        """ test_quadratic_wrong_solver

        Tests that a solver which does not support quadratic constraints
        aborts rather than silently dropping them

        """
        process = subprocess.run(['gboml', 'test/test34_quadratic.txt',
                                  "--clp"],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertNotEqual(return_code, 0)

    def test_quadratic_objective(self):
        """ test_quadratic_objective

        Tests an objective which is a sum of squares. Spreading a total of six
        over three time steps gives three times two squared

        """
        process = subprocess.run(
            ['gboml', 'test/test44_quadratic_objective.txt', "--gurobi",
             "--json", "--output", "test/test44_quadratic_objective"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

        with open("test/test44_quadratic_objective.json", 'r') as j:
            contents = json.loads(j.read())
            solution = contents["solution"]
            self.assertAlmostEqual(solution["objective"], 12.0, places=4)
            node = solution["elements"]["A"]
            for x_i in node["variables"]["x"]["values"]:
                self.assertAlmostEqual(x_i, 2.0, places=4)
            # the value reported for the objective covers its quadratic part
            self.assertAlmostEqual(node["objectives"]["named"]["cost"], 12.0,
                                   places=4)

    def test_quadratic_objective_maximised(self):
        """ test_quadratic_objective_maximised

        Tests that maximising a sum of squares, which is not convex once
        negated into a minimisation, reaches the bound of the variables. The
        solver recognises the form and turns to spatial branching on its own,
        no parameter of ours being set

        """
        process = subprocess.run(
            ['gboml', 'test/test45_quadratic_objective_max.txt', "--gurobi",
             "--json", "--output", "test/test45_quadratic_objective_max"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

        with open("test/test45_quadratic_objective_max.json", 'r') as j:
            contents = json.loads(j.read())
            solution = contents["solution"]
            # both variables reach three in absolute value, giving 2*9. A
            # maximised objective is reported negated at the top level, as it
            # is the value of the minimisation the solver was handed, and put
            # back the right way round for the objective of the node itself
            self.assertAlmostEqual(solution["objective"], -18.0, places=4)
            self.assertAlmostEqual(
                solution["elements"]["A"]["objectives"]["named"]["cost"],
                18.0, places=4)

    def test_quadratic_convex_constraint(self):
        """ test_quadratic_convex_constraint

        Tests a constraint holding a square, x**2 <= 4, under a maximisation.
        No bound of any kind is written on x : the square alone is what pins
        it, and reading that is the solver's business

        """
        process = subprocess.run(
            ['gboml', 'test/test46_quadratic_convex_constraint.txt',
             "--gurobi", "--json",
             "--output", "test/test46_quadratic_convex_constraint"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        return_code = process.returncode
        self.assertEqual(return_code, 0)

        with open("test/test46_quadratic_convex_constraint.json", 'r') as j:
            contents = json.loads(j.read())
            solution = contents["solution"]
            # x**2 <= 4 gives x <= 2, maximised over the two time steps and
            # reported negated at the top level as every maximisation is
            self.assertAlmostEqual(solution["objective"], -4.0, places=4)
            for x_i in solution["elements"]["A"]["variables"]["x"]["values"]:
                self.assertAlmostEqual(x_i, 2.0, places=4)

    def test_quadratic_objective_unbounded(self):
        """ test_quadratic_objective_unbounded

        Tests that an objective holding a product of two variables which have
        no upper bound compiles. Whether the solver needs those bounds depends
        on the shape of the quadratic form, which it classifies itself, so the
        model is handed over rather than refused

        """
        gboml_model = GbomlGraph(2)
        nodes, edges, global_params = \
            gboml_model.import_all_nodes_and_edges(
                "test/test47_quadratic_objective_unbounded.txt")
        gboml_model.add_nodes_in_model(*nodes)
        gboml_model.add_hyperedges_in_model(*edges)
        gboml_model.add_global_parameters(global_params)
        gboml_model.build_model()

        self.assertTrue(has_quadratic_terms(
            gboml_model.quadratic_eq, gboml_model.quadratic_ineq,
            gboml_model.quadratic_objective))

    def test_quadratic_objective_wrong_solver(self):
        """ test_quadratic_objective_wrong_solver

        Tests that a solver which does not support a quadratic objective
        aborts rather than solving the linear part alone

        """
        process = subprocess.run(['gboml',
                                  'test/test44_quadratic_objective.txt',
                                  "--clp"],
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True)
        return_code = process.returncode
        self.assertNotEqual(return_code, 0)


    def solve_sum_semantics(self, model, name):
        """ solve_sum_semantics

        Runs one of the sum semantics models and returns its variables. Every
        variable of those models is pinned by an equality, so the feasible set
        is a single point and a disagreement is a factorization bug rather than
        a tolerance

        """
        process = subprocess.run(
            ['gboml', 'test/' + model + '.txt', "--gurobi", "--json",
             "--output", "test/" + model + "_" + name],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        self.assertEqual(process.returncode, 0)

        with open("test/" + model + "_" + name + ".json", 'r') as j:
            solution = json.loads(j.read())["solution"]

        return solution["elements"]["A"]["variables"]

    def test_sum_ragged_range(self):
        """test_sum_ragged_range

        Tests a sum whose range ends on the index of the constraint, so every
        row holds a different number of terms. The rows which are shorter than
        the longest one used to be dropped silently.
        The same sum is checked once more with a product of two variables in
        it, to cover the case where a sum and a product are combined

        """
        variables = self.solve_sum_semantics("test48_sum_semantics", "ragged")
        self.assertEqual(variables["z_ragged"]["values"],
                         [1.0, 3.0, 6.0, 10.0])

        quadratic = self.solve_sum_semantics(
            "test49_sum_semantics_quadratic", "ragged")
        self.assertEqual(quadratic["z_ragged"]["values"],
                         [1.0, 3.0, 6.0, 10.0])

    def test_sum_where_condition(self):
        """test_sum_where_condition

        Tests a sum filtered by a where condition

        """
        variables = self.solve_sum_semantics("test48_sum_semantics", "cond")
        self.assertEqual(variables["z_cond"]["values"],
                         [8.0, 8.0, 8.0, 8.0])

    def test_sum_ragged_range_with_condition(self):
        """ test_sum_ragged_range_with_condition

        Tests a filtered sum over a range which depends on the index of the
        constraint, the case where the rows have to stay rectangular once the
        condition has dropped entries

        """
        variables = self.solve_sum_semantics("test48_sum_semantics",
                                             "ragged_cond")
        self.assertEqual(variables["z_ragged_cond"]["values"],
                         [1.0, 1.0, 4.0, 8.0])

    def test_nested_sum(self):
        """test_nested_sum

        Tests a sum inside a sum, the inner range depending on the outer index.
        This used to crash the linear factorization outright. Checked once more
        with a product inside

        """
        variables = self.solve_sum_semantics("test48_sum_semantics", "nested")
        self.assertEqual(variables["z_nested"]["values"],
                         [10.0, 10.0, 10.0, 10.0])

        quadratic = self.solve_sum_semantics(
            "test49_sum_semantics_quadratic", "nested")
        self.assertEqual(quadratic["z_nested"]["values"],
                         [10.0, 10.0, 10.0, 10.0])

    def test_nested_sum_ragged_range(self):
        """ test_nested_sum_ragged_range

        Tests a sum inside a sum whose outer range depends on the index of the
        constraint, so both kinds of range are ragged at once

        """
        variables = self.solve_sum_semantics("test48_sum_semantics",
                                             "nested_ragged")
        self.assertEqual(variables["z_nested_ragged"]["values"],
                         [1.0, 4.0, 10.0, 20.0])

    def test_padded_entry_out_of_bounds(self):
        """ test_padded_entry_out_of_bounds

        Tests a ragged sum whose padding carries an index out of bounds. A cell
        the range does not reach holds 0, so `x[i - 1]` reads `x[-1]` there.
        Nothing reads that cell, so it must not invalidate its row : dropping
        the excuse keeps only the row that has no padding at all, 4 equality
        rows instead of 6

        """
        process = subprocess.run(
            ['gboml', 'test/test59_padded_out_of_bounds.txt', "--gurobi",
             "--json", "--output", "test/test59_padded_out_of_bounds"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        self.assertEqual(process.returncode, 0)

        with open("test/test59_padded_out_of_bounds.json", 'r') as j:
            solution = json.loads(j.read())["solution"]
        variables = solution["elements"]["A"]["variables"]
        self.assertEqual(variables["x"]["values"], [1.0, 2.0, 3.0])
        for got, expected in zip(variables["z"]["values"], [1.0, 3.0, 6.0]):
            self.assertAlmostEqual(got, expected, places=4)

    def test_sum_products_and_constants(self):
        """ test_sum_products_and_constants

        Tests the two ways an index of a sum meets a product. An index the sum
        is still running over is read at the same value by both operands, so
        the products between different summands do not exist : the dot product
        is 32 and not the 90 every pair would give. An index the sum has
        already laid out as entries is a summand of its own, so two sums side
        by side do pair every entry with every entry, whether they are spelled
        with the same letter or not, and a sum squared is 36 and not 14.
        The constants pin the other half of that split : a constant inside a
        sum really is summed over the range, unlike a term, which the matrix
        sums later

        """
        process = subprocess.run(
            ['gboml', 'test/test60_sum_products.txt', "--gurobi", "--json",
             "--output", "test/test60_sum_products"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        self.assertEqual(process.returncode, 0)

        with open("test/test60_sum_products.json", 'r') as j:
            solution = json.loads(j.read())["solution"]
        self.assertAlmostEqual(solution["objective"], 96.0, places=4)
        variables = solution["elements"]["A"]["variables"]
        expected = {
            "z_dot": [32.0, 32.0, 32.0],
            "z_sibling": [90.0, 90.0, 90.0],
            "z_two_axes": [150.0, 150.0, 150.0],
            "z_square": [36.0, 36.0, 36.0],
            "z_const_i": [12.0, 12.0, 12.0],
            "z_const_1": [9.0, 9.0, 9.0],
            "z_const_t": [6.0, 9.0, 12.0],
        }
        for name, values in expected.items():
            for got, want in zip(variables[name]["values"], values):
                self.assertAlmostEqual(got, want, places=4, msg=name)

    def test_constant_where_on_a_constraint(self):
        """ test_constant_where_on_a_constraint

        Tests a `where` on a constraint whose condition does not read the index
        of the constraint. It then evaluates to a single bool rather than to
        one per row, and either every row is kept or none is. Filtering used to
        zip the rows with that bool and raise a raw TypeError, on master too

        """
        process = subprocess.run(
            ['gboml', 'test/test58_constant_where.txt', "--gurobi", "--json",
             "--output", "test/test58_constant_where"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        self.assertEqual(process.returncode, 0)

        with open("test/test58_constant_where.json", 'r') as j:
            solution = json.loads(j.read())["solution"]
        self.assertAlmostEqual(solution["objective"], 32.0, places=4)
        variables = solution["elements"]["A"]["variables"]
        # the condition held, so every row was kept and x is pinned
        self.assertEqual(variables["x"]["values"], [1.0, 1.0, 1.0, 1.0])
        # it did not, so the constraint contributed no row and only the bound
        # is left on y
        self.assertEqual(variables["y"]["values"], [7.0, 7.0, 7.0, 7.0])

    def test_sum_body_reads_enclosing_index(self):
        """ test_sum_body_reads_enclosing_index

        Tests a sum whose body reads the index of the constraint, and not only
        its range. An enclosing index is constant along the cells of the sum,
        so it has to be laid out over them before the body can be evaluated;
        the sum only does that for the indexes it actually reads, and this is
        the shape that tells apart an index it reads from one appearing in the
        range alone. Getting it wrong raises on the shapes rather than
        answering a wrong number, but the values are pinned all the same

        """
        variables = self.solve_sum_semantics("test48_sum_semantics", "body_t")
        self.assertEqual(variables["z_body_t"]["values"],
                         [0.0, 3.0, 12.0, 30.0])

    def test_sum_in_objective_ragged_range(self):
        """ test_sum_in_objective_ragged_range

        Tests a sum in an objective whose range depends on the index the
        objective is expanded over. The index appearing only in the range of
        the sum used to leave the objective on a single row

        """
        process = subprocess.run(
            ['gboml', 'test/test50_sum_in_objective.txt', "--gurobi", "--json",
             "--output", "test/test50_sum_in_objective"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        self.assertEqual(process.returncode, 0)

        with open("test/test50_sum_in_objective.json", 'r') as j:
            solution = json.loads(j.read())["solution"]
            # one row per time step, summed : 1 + 3 + 6 + 10
            self.assertAlmostEqual(solution["objective"], 20.0, places=4)

    def test_single_row_sum_range(self):
        """ test_single_row_sum_range

        Tests a sum whose range ends on the index of the constraint over a
        single time step. The index is then bound to an array holding one
        value, which the range has to reduce to a plain integer before it can
        be built

        """
        process = subprocess.run(
            ['gboml', 'test/test51_single_row_sum_range.txt', "--linprog",
             "--json", "--output", "test/test51_single_row_sum_range"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        self.assertEqual(process.returncode, 0)

        with open("test/test51_single_row_sum_range.json", 'r') as j:
            solution = json.loads(j.read())["solution"]
            variables = solution["elements"]["A"]["variables"]
            self.assertEqual(variables["z"]["values"], [1.0])

    def test_constant_objective(self):
        """ test_constant_objective

        Tests an objective built out of parameters only, which holds no row of
        the objective matrix and only shifts the optimum by its own constant.
        Its independent term is one value per row of the objective it was
        expanded over, which used to be truth tested as a whole

        """
        process = subprocess.run(
            ['gboml', 'test/test53_constant_objective.txt', "--linprog",
             "--json", "--output", "test/test53_constant_objective"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        self.assertEqual(process.returncode, 0)

        with open("test/test53_constant_objective.json", 'r') as j:
            solution = json.loads(j.read())["solution"]
            # 3 for the objective holding a variable, 0 + 1 + 2 for the one
            # built out of the index only
            self.assertAlmostEqual(solution["objective"], 6.0, places=4)
            objectives = solution["elements"]["A"]["objectives"]["named"]
            self.assertAlmostEqual(objectives["c"], 3.0, places=4)
            self.assertAlmostEqual(objectives["d"], 3.0, places=4)

    def test_empty_objective(self):
        """ test_empty_objective

        Tests an objective whose condition keeps no index at all. It expands
        to no row and contributes nothing, rather than reporting an
        independent term it does not have

        """
        process = subprocess.run(
            ['gboml', 'test/test54_empty_objective.txt', "--linprog",
             "--json", "--output", "test/test54_empty_objective"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        self.assertEqual(process.returncode, 0)

        with open("test/test54_empty_objective.json", 'r') as j:
            solution = json.loads(j.read())["solution"]
            self.assertAlmostEqual(solution["objective"], 3.0, places=4)
            objectives = solution["elements"]["A"]["objectives"]["named"]
            self.assertAlmostEqual(objectives["c"], 3.0, places=4)
            self.assertAlmostEqual(objectives["d"], 0.0, places=4)

    def test_power_operator(self):
        """ test_power_operator

        Tests the ** operator applied to an expression holding a variable.
        Only the exponent is a constant, and it is read from the expression of
        the exponent alone : the whole power expression can not be evaluated,
        its base holding a variable, and asking it for a constant used to
        refuse every power outright. Each exponent is written in a different
        shape, a bare parameter, a literal, an expression over the horizon,
        and the degenerate 0 and 1 which produce no product at all

        """
        process = subprocess.run(
            ['gboml', 'test/test56_power_operator.txt', "--gurobi",
             "--json", "--output", "test/test56_power_operator"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        self.assertEqual(process.returncode, 0)

        with open("test/test56_power_operator.json", 'r') as j:
            solution = json.loads(j.read())["solution"]
            self.assertAlmostEqual(solution["objective"], 13.0, places=4)
            variables = solution["elements"]["A"]["variables"]
            self.assertEqual(variables["x"]["values"], [2.0, 3.0])
            for name in ["by_parameter", "by_literal"]:
                self.assertEqual(variables[name]["values"], [4.0, 9.0])
            for name in ["by_horizon", "by_one"]:
                self.assertEqual(variables[name]["values"], [2.0, 3.0])
            self.assertEqual(variables["by_zero"]["values"], [1.0, 1.0])


    def test_a_hyperedge_range_and_where_read_the_edge_parameters(self):
        """ test_a_hyperedge_range_and_where_read_the_edge_parameters

        The naming pass visited the two sides of an edge constraint and
        nothing else, so a name in its range or in its condition was left
        bare and reached the evaluator undefined. Every part is pinned by a
        value here : the range keeps 1, 3 and 5, the condition drops 3, and
        w follows y on those rows alone

        """
        process = subprocess.run(
            ['gboml', 'test/test69_hyperedge_range_and_where.txt', "--linprog",
             "--json", "--output", "test/test69_hyperedge_range_and_where"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        self.assertEqual(process.returncode, 0)

        with open("test/test69_hyperedge_range_and_where.json", 'r') as j:
            solution = json.loads(j.read())["solution"]
            elements = solution["elements"]
            self.assertEqual(elements["A"]["variables"]["y"]["values"],
                             [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            self.assertEqual(elements["B"]["variables"]["w"]["values"],
                             [0.0, 2.0, 0.0, 0.0, 0.0, 6.0])

    def test_a_range_is_checked_like_the_rest_of_the_constraint(self):
        """ test_a_range_is_checked_like_the_rest_of_the_constraint

        A bound reading the index which numbers the rows names nothing, the
        range being evaluated before that index is bound. The bounds were the
        only expressions the definition pass never looked at, so this used to
        raise a NameError out of the evaluator

        """
        process = subprocess.run(
            ['gboml', 'test/test70_range_reads_unbound_name.txt', "--linprog"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("Unknown identifier", process.stdout)


    def test_a_where_is_checked_like_the_rest_of_the_constraint(self):
        """ test_a_where_is_checked_like_the_rest_of_the_constraint

        A condition holds expressions the way the two sides do, and was the
        last piece of a constraint the definition pass never read. A bare
        name there used to raise a NameError out of the evaluator

        """
        process = subprocess.run(
            ['gboml', 'test/test71_condition_reads_unbound_name.txt',
             "--linprog"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("Unknown identifier", process.stdout)

    def test_the_range_of_a_sum_is_checked_like_the_rest_of_it(self):
        """ test_the_range_of_a_sum_is_checked_like_the_rest_of_it

        Unlike the range of a `for`, the range of a sum may name an index the
        rows run over, which is what makes it ragged, but it still has to name
        something

        """
        process = subprocess.run(
            ['gboml', 'test/test72_sum_range_reads_unbound_name.txt',
             "--linprog"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("Unknown identifier", process.stdout)


    def test_a_sum_reads_a_parent_parameter(self):
        """ test_a_sum_reads_a_parent_parameter

        The children of a sum were the one expression checked without the
        list of parent nodes, so a reference a node is allowed to make was
        refused as soon as it stood inside a sum. Three times cap, read from
        the parent, against the two the same reference gives outside one

        """
        process = subprocess.run(
            ['gboml', 'test/test73_sum_reads_a_parent_parameter.txt',
             "--linprog", "--json", "--output",
             "test/test73_sum_reads_a_parent_parameter"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        self.assertEqual(process.returncode, 0)

        with open("test/test73_sum_reads_a_parent_parameter.json", 'r') as j:
            solution = json.loads(j.read())["solution"]
            self.assertEqual(
                solution["elements"]["A"]["variables"]["y"]["values"],
                [6.0, 6.0, 6.0])

    def test_the_where_of_a_sum_is_checked_like_the_rest_of_it(self):
        """ test_the_where_of_a_sum_is_checked_like_the_rest_of_it

        The condition of a sum was the last expression the definition pass
        never read, one level below the condition of a constraint. A bare
        name there used to raise a NameError out of the evaluator

        """
        process = subprocess.run(
            ['gboml', 'test/test75_sum_condition_reads_unbound_name.txt',
             "--linprog"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("Unknown identifier", process.stdout)

    def test_the_where_of_a_sum_refuses_a_variable(self):
        """ test_the_where_of_a_sum_refuses_a_variable

        A `where` is decided before the problem is built, so a variable in
        one names nothing to decide it with

        """
        process = subprocess.run(
            ['gboml', 'test/test76_sum_condition_reads_a_variable.txt',
             "--linprog"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("variables are not allowed", process.stdout)

    def test_the_where_of_a_sum_reads_a_parameter(self):
        """ test_the_where_of_a_sum_reads_a_parameter

        The condition of a sum was never given the name of the node holding
        it, so a parameter it reads was refused as a bare name while the same
        reference works in the range and in the body. keep drops i == 1, so
        every row holds 1 + 3 + 4

        """
        process = subprocess.run(
            ['gboml', 'test/test77_sum_condition_reads_a_parameter.txt',
             "--linprog", "--json", "--output",
             "test/test77_sum_condition_reads_a_parameter"],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        self.assertEqual(process.returncode, 0)

        with open("test/test77_sum_condition_reads_a_parameter.json", 'r') as j:
            solution = json.loads(j.read())["solution"]
            self.assertEqual(
                solution["elements"]["A"]["variables"]["z"]["values"],
                [8.0, 8.0, 8.0])


class GBOMLpyTest(unittest.TestCase):
    """
    GBOMLpyTest

    Tests the GBOML python interface

    It checks whether a file correctly fails, generates the correct matrices,
    correct solutions and correct output.

    """

    def test_horizon_initialization(self):
        """ test_horizon_intialization

        Tests initializing the timehorizon with a value

        """
        timehorizon_considered = 10
        gboml_model = GbomlGraph(timehorizon_considered)
        self.assertEqual(gboml_model.get_timehorizon(), timehorizon_considered)

    def test_horizon_set(self):
        """ test_horizon_set

        Tests setting the timehorizon after object initialization

        """
        gboml_model = GbomlGraph()
        gboml_model.set_timehorizon(12)
        self.assertEqual(gboml_model.get_timehorizon(), 12)

    def test_import_same_name(self):
        """ test_import_same_name

        Tests importing a node without renaming it

        """
        gboml_model = GbomlGraph()
        node_pv = gboml_model.import_node("examples/microgrid/microgrid.txt",
                                          "SOLAR_PV", copy=True)
        self.assertEqual(type(node_pv), Node)
        self.assertEqual(node_pv.get_name(), "SOLAR_PV")

    def test_renaming_in_import(self):
        """ test_renaming_in_import

        Test importing a node and renaming it

        """
        gboml_model = GbomlGraph()
        node_b = gboml_model.import_node("examples/microgrid/microgrid.txt",
                                         "DEMAND", new_node_name="B", copy=True)
        self.assertEqual(node_b.get_name(), "B")
        self.assertEqual(type(node_b), Node)

    def test_redefine_parameters_from_list(self):
        """ test_redefine_parameters_from_list

        Tests importing a node, redefining its parameters and checking changes

        """
        gboml_model = GbomlGraph()
        node_pv = gboml_model.import_node("examples/microgrid/microgrid.txt",
                                          "SOLAR_PV", copy=True)
        all_parameters_names = []
        all_parameters_values = []
        for i, parameter in enumerate(node_pv.get_parameters()):
            all_parameters_names.append(parameter.get_name())
            all_parameters_values.append(i)

        gboml_model.redefine_parameters_from_list(node_pv, all_parameters_names,
                                                  all_parameters_values)
        for i, parameter in enumerate(node_pv.get_parameters_changes()):
            name = parameter.get_name()
            value = parameter.get_expression().get_name()
            self.assertEqual(name, all_parameters_names[i])
            self.assertEqual(value, all_parameters_values[i])

        node_pv_2 = gboml_model.import_node("examples/microgrid/microgrid.txt",
                                            "SOLAR_PV", copy=True)
        self.assertEqual(node_pv_2.get_parameters_changes(), [])

    def test_remove_objective(self):
        """ test_remove_objective

        Tests importing a node and removing its objective

        """
        gboml_model = GbomlGraph()
        node_pv = gboml_model.import_node("examples/microgrid/microgrid.txt",
                                          "SOLAR_PV", copy=True)
        gboml_model.remove_objective_in_node(node_pv, "investment")
        for objective in node_pv.get_objectives():
            self.assertTrue(objective.get_name() != "investment")

    def test_change_type_variable(self):
        """ test_change_type_variable

        Tests importing a node and changing the type of one of its variables

        """
        gboml_model = GbomlGraph()
        node_pv = gboml_model.import_node("examples/microgrid/microgrid.txt",
                                          "SOLAR_PV", copy=True)
        gboml_model.change_type_variable_in_node(node_pv, "investment",
                                                 VariableType.EXTERNAL)
        self.assertEqual(len(node_pv.get_variables_changes()), 1)
        self.assertEqual(node_pv.get_variables_changes(),
                         [["investment", "external", 0]])

    def test_import_everything_and_solve(self):
        """ test_import_everything_and_solve

        Tests to import a full file, renaming its nodes and solve it

        """
        import os
        import sys
        f = open(os.devnull, 'w')
        temp = sys.stdout
        sys.stdout = f

        gboml_model = GbomlGraph()
        gboml_model_with_1 = GbomlGraph()
        nodes, edges, global_params = gboml_model.import_all_nodes_and_edges(
            "examples/microgrid/microgrid.txt")

        old_names = []
        for node in nodes:
            old_names.append(node.get_name())
            gboml_model_with_1.rename(node, node.get_name() + "_1")

        for hyperedge in edges:
            gboml_model_with_1.rename(hyperedge, hyperedge.get_name() + "_1")
            for i, node in enumerate(nodes):
                gboml_model_with_1.change_node_name_in_hyperedge(hyperedge,
                                                                 old_names[i],
                                                                 node.get_name()
                                                                 )

        gboml_model_with_1.add_nodes_in_model(*nodes)
        gboml_model_with_1.add_hyperedges_in_model(*edges)
        gboml_model_with_1.add_global_parameters_objects(global_params)
        gboml_model_with_1.build_model()
        solution_with_1 = gboml_model_with_1.solve_gurobi()
        f.close()
        sys.stdout = temp
        self.assertEqual(solution_with_1[0].all(), np.array(
            [0., 0., 6.9, 0., 0., 0., 0., 6.9, 0., 0., 0.]).all())

    def test_encapsulating(self):
        """ test_encapsulating

        Tests to import the microgrid problem and put everything inside a toy
        node from tests and solve

        """
        import os
        import sys
        f = open(os.devnull, 'w')
        temp = sys.stdout
        sys.stdout = f

        gboml_model = GbomlGraph()
        nodes, edges, global_param = gboml_model.import_all_nodes_and_edges(
            "examples/microgrid/microgrid.txt")
        parent = gboml_model.import_node("test/test6.txt", "H", copy=True)
        for node in nodes:
            gboml_model.add_sub_node(node, parent)

        for edge in edges:
            gboml_model.add_sub_hyperedge(edge, parent)

        gboml_model.add_nodes_in_model(parent)
        gboml_model.add_global_parameters_objects(global_param)
        gboml_model.build_model()
        solution = gboml_model.solve_cplex("src/gboml/solver_api/cplex.opt", False, {"lpmethod": 1})
        f.close()
        sys.stdout = temp
        self.assertEqual(solution[0].all(), np.array(
            [0., 0., 6.9, 0., 0., 0., 0., 6.9, 0., 0., 0., 4.]).all())

    def test_quadratic_build_and_solve(self):
        """ test_quadratic_build_and_solve

        Tests building a graph holding a quadratic constraint through the
        python interface and solving it with Gurobi

        """
        import os
        import sys
        f = open(os.devnull, 'w')
        temp = sys.stdout
        sys.stdout = f

        gboml_model = GbomlGraph(3)
        nodes, edges, global_params = gboml_model.import_all_nodes_and_edges(
            "test/test34_quadratic.txt")
        gboml_model.add_nodes_in_model(*nodes)
        gboml_model.add_hyperedges_in_model(*edges)
        gboml_model.add_global_parameters(global_params)
        gboml_model.build_model()
        quadratic_terms = (gboml_model.quadratic_eq,
                           gboml_model.quadratic_ineq,
                           gboml_model.quadratic_objective)
        nb_eq_rows = gboml_model.matrix_a_eq.shape[0]
        _, objective, _, _, _, _ = gboml_model.solve_gurobi()
        f.close()
        sys.stdout = temp

        self.assertTrue(has_quadratic_terms(*quadratic_terms))
        # the degree two part shares the rows of the degree one one
        self.assertEqual(gboml_model.quadratic_eq.shape[0], nb_eq_rows)
        self.assertAlmostEqual(objective, 15.0, places=4)

    def test_quadratic_objective_reaches_the_matrix(self):
        """ test_quadratic_objective_reaches_the_matrix

        Tests that an objective which is a bare product of two variables, and
        therefore holds no degree one term of its own, still expands to the
        rows of the objective matrix and still carries its degree two part.
        Reading the degree two part off the rows of the degree one one is what
        makes those rows have to exist

        """
        import os
        import sys
        f = open(os.devnull, 'w')
        temp = sys.stdout
        sys.stdout = f

        gboml_model = GbomlGraph(3)
        nodes, edges, global_params = gboml_model.import_all_nodes_and_edges(
            "test/test44_quadratic_objective.txt")
        gboml_model.add_nodes_in_model(*nodes)
        gboml_model.add_hyperedges_in_model(*edges)
        gboml_model.add_global_parameters(global_params)
        gboml_model.build_model()
        quadratic_objective = gboml_model.quadratic_objective
        vector_c = gboml_model.vector_c
        _, objective, _, _, _, _ = gboml_model.solve_gurobi()
        f.close()
        sys.stdout = temp

        # the objective still holds one row per time step, even though not one
        # of them carries a degree one entry
        self.assertEqual(vector_c.shape[0], 3)
        self.assertEqual(vector_c.nnz, 0)
        # and its degree two part lands on those very same rows
        self.assertEqual(quadratic_objective.nnz, 3)
        self.assertEqual(sorted(quadratic_objective.coords[0].tolist()),
                         [0, 1, 2])
        self.assertAlmostEqual(objective, 12.0, places=4)

    def test_quadratic_unsupported_solver(self):
        """ test_quadratic_unsupported_solver

        Tests that solving a graph holding a quadratic constraint with a
        solver that does not support quadratic constraints aborts

        """
        import os
        import sys
        f = open(os.devnull, 'w')
        temp = sys.stdout
        sys.stdout = f

        gboml_model = GbomlGraph(3)
        nodes, edges, global_params = gboml_model.import_all_nodes_and_edges(
            "test/test34_quadratic.txt")
        gboml_model.add_nodes_in_model(*nodes)
        gboml_model.add_hyperedges_in_model(*edges)
        gboml_model.add_global_parameters(global_params)
        gboml_model.build_model()
        with self.assertRaises(SystemExit):
            gboml_model.solve_clp()
        f.close()
        sys.stdout = temp

    def test_toy_example(self):
        """ test_toy_example

        Tests combining importing, renaming, parameter_redefinition all
        together and solving

        """
        import os
        import sys
        f = open(os.devnull, 'w')
        temp = sys.stdout
        sys.stdout = f
        timehorizon = 3
        gboml_model = GbomlGraph(timehorizon)
        nodes, edges, global_params = gboml_model.import_all_nodes_and_edges(
            "examples/microgrid/microgrid.txt")
        old_names = []
        for node in nodes:
            old_names.append(node.get_name())
            gboml_model.rename(node, "new_" + node.get_name())

        for hyperedge in edges:
            gboml_model.rename(hyperedge, "new_" + hyperedge.get_name())
            for i, node in enumerate(nodes):
                gboml_model.change_node_name_in_hyperedge(hyperedge,
                                                          old_names[i],
                                                          node.get_name())

        parent = gboml_model.import_node("test/test6.txt", "H", copy=True)
        for node in nodes:
            gboml_model.add_sub_node(node, parent)

        for edge in edges:
            gboml_model.add_sub_hyperedge(edge, parent)

        gboml_model.redefine_parameters_from_keywords(parent, b=6)
        gboml_model.add_nodes_in_model(parent)
        gboml_model.add_global_parameters(global_params)
        gboml_model.add_global_parameters([("electricity_price", 0.05)])
        gboml_model.build_model()
        x, objective, status, solver_info = gboml_model.solve_cbc(opt_dict={"gap": [float, 10]})
        f.close()
        sys.stdout = temp
        self.assertTrue(np.allclose(x, np.array(
            [0.345, 0.32, 0., 6.9, 6.4, 0., 6.9, 6.4, 6.1, -0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 6.1,
             -0., 0., 0., 0., 0., 6., 6., 6.])))


if __name__ == '__main__':
    """ __main__

    main initializing the unittests. 

    """
    unittest.main()
