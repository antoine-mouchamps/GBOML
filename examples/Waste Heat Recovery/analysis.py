# Copyright (C) 2023
# Liam Dauchat
# University of Liege .
# All rights reserved.

#parsing
import os
import json as js
import re
#treatment
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
#reporting
import pprint
from PyPDF2 import PdfMerger
from tabulate import tabulate
import subprocess




def save_pretty_dictionnary(data,file_name,save=False):
    """
    PRE: data is a dictionary, save is a boolean indicating if we want to save the dictionary in a file by default file_name+_pretty
    POST: print the dictionary in a pretty way
    """

    pretty = js.dumps(data, indent=4, sort_keys=True)
    # print(data) # to be used with causion as supply vectors are very long
    if save:
        with open(file_name+"_pretty", 'w') as f:
            pprint.pprint(data, f)

def data_description(data):
    print("data description ================")
    print("keys: ",data.keys())
    print("version: ",data['version'])
    # print("model: ",data['model'])
    print("solver: ",data['solver'])
    print("end of data description ================ \n")

def model_description(model):
    print("model description ================")
    print("model keys: ",model.keys())
    print("model nodes: ",model['nodes'].keys())
    print("model hyperedges: ",model['hyperedges'].keys())
    print("end of model description ================ \n")

def solution_description(solution):
    print("solution description ================")
    print("solution keys: ",solution.keys())
    print("solution status: ",solution['status'])
    print("solution objective: ",solution['objective'])
    print("solution elements: ",solution['elements'].keys())
    for key in solution["elements"].keys():
        print("solution elements ",key," len : ",len(solution['elements'][key]))
    print("end of solution description ================ \n")


def make_node_report_old(node_name, node_data,destination):
    """
    OUTDATED
    PRE: 
        - node_data is the node data stored in a dictionary
        - node_name is a string representing the name of the node
        - destination is a string representing the destination folder for all node reports
    POST: save all nodes' variables on a single PDF file
    """
    # Calculate the number of rows and columns based on the number of plots
    num_plots = len(node_data.keys())
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols

    output_filename = node_name + "_DATA.pdf"
    output_path = os.path.join(destination, output_filename)
    pdf_pages = PdfPages(output_path)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    axes = axes.flatten()

    for idx, key in enumerate(node_data.keys()):
        ax = axes[idx]
        ax.plot(node_data[str(key)]["values"])
        ax.set_ylabel(str(key))
        ax.set_xlabel("time")
        ax.set_title(node_name + " " + str(key))

    # Hide any empty subplots
    for idx in range(num_plots, num_rows * num_cols):
        fig.delaxes(axes[idx])

    fig.suptitle(node_name+"_DATA", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Add some space for the title
    pdf_pages.savefig(fig)
    plt.close(fig)
    pdf_pages.close()



def make_node_report(node_name, node_data, destination):
    """
    PRE: 
        - node_data is the node data stored in a dictionary
        - node_name is a string representing the name of the node
        - destination is a string representing the destination folder for all node reports
    POST: save all nodes' variables and objectives on a single PDF file
    """
    # Separate the keys into two groups
    data_tech = node_data["variables"]
    data_obj = node_data["objectives"]
    capacity_keys = []
    other_keys = []
    objective_keys = ["unnamed"] #list(data_obj.keys())

    for key in data_tech.keys():
        if "capacity" in key:
            capacity_keys.append(key)
        else:
            other_keys.append(key)

    # Calculate the number of rows and columns based on the number of plots
    num_plots = len(other_keys) + 2 # +2 for capacities and objectives
    num_cols = 1
    num_rows = (num_plots + num_cols - 1) // num_cols

    output_filename = node_name + "_DATA.pdf"
    output_path = os.path.join(destination, output_filename)
    pdf_pages = PdfPages(output_path)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    axes = axes.flatten()

    # Plot the capacities
    if capacity_keys:
        capacity_ax = axes[0]
        capacity_values = []
        capacity_labels = []
        for key in capacity_keys:
            values = data_tech[str(key)]["values"]
            capacity_values.extend(values)
            capacity_labels.append(key)

        capacity_ax.stem(capacity_labels, capacity_values, basefmt=" ",markerfmt=" ", orientation="horizontal")
        capacity_ax.set_title("Capacities")
        for label, value in zip(capacity_labels, capacity_values):
            capacity_ax.text(value , label, str(round(value, 2)), va='center', ha="left")

    
    # Plot the objective
    if objective_keys:
        objective_ax = axes[1]
        objective_values = []
        objective_labels = ["CAPEX","OPEX"]
        for key in objective_keys:
            values = data_obj[str(key)]
            objective_values.extend(values)

        objective_ax.stem(objective_labels, objective_values, basefmt=" ",markerfmt=" ", orientation="horizontal")
        objective_ax.set_title("Objectives")
        for label, value in zip(objective_labels, objective_values):
            objective_ax.text(value, label, str(round(value,2)), va='center')

    # Plot the remaining keys
    for idx, key in enumerate(other_keys):
        ax = axes[idx+2] if capacity_keys else axes[idx]
        values = data_tech[str(key)]["values"]
        non_zero_values = [val for val in values if val != 0]
        ax.plot(non_zero_values)
        ax.set_ylabel(str(key))
        ax.set_xlabel("time")
        ax.set_title(node_name + " " + str(key))

    # Hide any empty subplots
    for idx in range(num_plots, num_rows * num_cols):
        fig.delaxes(axes[idx])

    fig.suptitle(node_name+"_DATA", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Add some space for the title

    pdf_pages.savefig(fig)
    plt.close(fig)
    pdf_pages.close()

def make_node_statistics(node_name, node_variables_data,stats_df):
    """
    PRE : 
        - node_variables_data is the dictionnary containings all the items variables - values 
        - node_name is a string representing the name of the node
        - stats_df is a panda dataframe containing the statistics of all nodes
    POST : write/update stats dataframe for each variable the following statistics : mean, min, max, std, 25%, 50%, 75% 
    """
    for key in node_variables_data.keys():
        values = node_variables_data[str(key)]["values"]
        stats_df.loc[node_name+"_"+key] = [np.mean(values), np.min(values), np.max(values), np.std(values), 
                                           np.percentile(values, 25), np.percentile(values, 50), np.percentile(values, 75)]

def make_system_statistics(data):
    """
    PRE : 
        - data is a dictionary containing the solution of the model
    POST :
        - returns a panda dataframe containing for each variable of each node of the system the following 
        statistics : mean, min, max, std, 25%, 50%, 75%
    """
    stats = pd.DataFrame(columns=["mean", "min", "max", "std", "25%", "50%", "75%"])
    for key in data["solution"]["elements"].keys():
        make_node_statistics(key,data["solution"]["elements"][key]["variables"],stats)
    return stats


def make_system_report(data,destination,report_name, merge):
    """
    PRE: 
        - data is a dictionary containing the solution of the model
        - destination is a string representing the destination folder name in the results folder for the report
    POST: save a pdf file at destination agregating all the nodes' plots
    """

    if not os.path.exists(destination):
        os.makedirs(destination)
        # os.makedirs(os.path.join(destination, r"\report"))

    #Node report generation
    for key,val in data.items():
        if key == "elements":
            for k,v in val.items():
                make_node_report(k,v,destination)
        else:
            print(key,val)
    if merge : merge_pdfs(destination, report_name)

def merge_pdfs(destination, report_name):
    #PDF aggregation
    merger = PdfMerger()
    for filename in os.listdir(destination):
        if filename.endswith(".pdf"):
            filepath = os.path.join(destination, filename)
            merger.append(filepath)
    output_path = os.path.join(destination, report_name+".pdf")
    merger.write(output_path)
    merger.close()

def compile_tex_to_pdf(file_name):
    # Run LaTeX compiler to generate the PDF
    process = subprocess.Popen(['pdflatex', file_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    if process.returncode == 0:
        print('PDF generated successfully.')
    else:
        print('Error occurred during PDF generation:')
        print(error.decode())


def make_acronym(string):
    parts = string.split('_')
    new_name=""
    for part in parts:
        if part.islower():
            new_name +='_'+part
        else : new_name +=part[0]
    return new_name

def make_statistics_report(data, destination, report_name):
    """
    This function is not yet finished, tex file is generated but is too large and table is too long to fit on an A4 page
    PRE: 
        - data is a dictionary containing the solution of the model
        - destination is a string representing the destination folder name in the results folder for the report
        - report_name is a string representing the name of the report
    POST: save a tex file at destination aggregating all the nodes' statistics
    """

   
    statistics_data=make_system_statistics(data)
    df = pd.DataFrame.from_dict(statistics_data)
    df.index = [make_acronym(index) for index in df.index]

    #save the statistics in a csv file
    csv_stats_path = os.path.join(destination, "statistics" + ".csv")
    statistics_data.to_csv(csv_stats_path)
    # Save the DataFrame to a LaTeX table with custom styling
    table = tabulate(df, headers='keys', tablefmt='latex')

    # Define a LaTeX template with custom styling for the document and save it to a .tex file
    latex_template = r'''
    \documentclass[10pt]{article}
    \usepackage{booktabs}
    \usepackage{array}
    \usepackage{lscape}
    \begin{document}
    \pagenumbering{gobble}
    \centering
    \renewcommand{\arraystretch}{1.5}
    
    \begin{landscape}
    \begin{tabular}{>{\raggedright\arraybackslash}p{1cm}cc}
    %s
    \end{tabular}
    \end{landscape}
    \end{document}
    '''
    tex_path = os.path.join(destination, report_name + ".tex")
    with open(tex_path, 'w') as f:
        f.write(latex_template % table)

def make_system_capacity_report(data,destination,report_name):
    """
    PRE:
        - data is a dictionary containing the solution of the model
        - destination is a string representing the destination folder name in the results folder for the report
        - report_name is a string representing the name of the report
    POST:
        - saves a table containing node_name, capacity_name, value in a csv file at destination
    """
    capacity_data = pd.DataFrame(columns=["node_name","capacity_name","value"])
    for key in data["solution"]["elements"].keys():
        for k,v in data["solution"]["elements"][key]["variables"].items():
            if "capacity" in k:
                capacity_data.loc[len(capacity_data)] = [key,k,v["values"][-1]]
    capacity_data.to_csv(os.path.join(destination,"Capacity_"+report_name+".csv"))

def make_system_cost_report(data,destination,report_name):
    """
    PRE:
        - data is a dictionary containing the solution of the model
        - destination is a string representing the destination folder name in the results folder for the report
        - report_name is a string representing the name of the report
    POST:
        - saves a table containing node_name, capex, opex and sorted by greatest capex in a csv file at destination
    """
    cost_data = pd.DataFrame(columns=["node_name","capex","opex"])
    for key in data["solution"]["elements"].keys():
        cost_data.loc[len(cost_data)] = [key,data["solution"]["elements"][key]["objectives"]["unnamed"][0],data["solution"]["elements"][key]["objectives"]["unnamed"][1]]
    cost_data.to_csv(os.path.join(destination,"Cost_"+report_name+".csv"))
          
def main():

    #=============================ANALYSIS PARAMETERS====================================
    gboml_file =  "RREHA_ref_5y"
    time_horizon = 5 #years
    
    #=============================load data====================================
    gboml_path = r"\results\GBOML"
    path = gboml_path + "\\" + gboml_file 
    report_folder = r"\results"+"\\"+gboml_file
    report_name = "Report_"+gboml_file
    
    root_path = os.path.abspath(os.path.dirname(__file__))
    file_name = root_path+path
    with open(file_name) as f:
        data = js.load(f)
    # save_pretty_dictionnary(data,file_name,True)

    #=============================Variables=============================
    installed_capacity= 1e7 #MWh
    yearly_cost = data.get("solution").get("objective") #M€
    print("Methane cost {} [€/MWh]".format(1e6*(yearly_cost/installed_capacity/time_horizon)))
    units = {"Power":"[GW]",
             "Energy":"[GWh]",
             "Cost":"[M€]",
             "Mass":"[kt]",
             "MassFlow":"[kt/h]"}

    #=============================data_description=============================

    # data_description(data)
    # model_description(data['model'])
    # save_pretty_dictionnary(data,True)
    # solution_description(data['solution'])
    # make_technology_plots("DIRECT_AIR_CAPTURE_PLANTS",data["solution"]["elements"]["DIRECT_AIR_CAPTURE_PLANTS"]["variables"]) #build a node report
    make_system_report(data["solution"],root_path+report_folder,report_name, True)
    # merge_pdfs(root_path+report_folder, report_name)
    make_statistics_report(data,root_path+report_folder,"statistics_report")
    make_system_capacity_report(data,root_path+report_folder,report_name)
    make_system_cost_report(data,root_path+report_folder,report_name)
if __name__ == "__main__":
    main()



