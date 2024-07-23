This subproject is an attempt to reproduce M. Berger's results and extract key statistics from the model.

##How to use analysis.py file:

- make_system_report builds a pdf report for the modelized system
- make_statistics_report builds a dataframe containing min,max,mean,25%,50%,75% of each node's variable and export it as a tex and csv file

Original Readme by M. Berger and al. : 

This linear programming model can be used to analyse the economics of producing carbon-neutral synthetic methane from solar and wind energy in North Africa and delivering it to Northwestern European markets. The model sizes and operations different technologies converting, storing and transporting energy carriers so as to minimise total system cost and satisfy a pre-specified methane demand level. The full supply chain is modelled in an integrated fashion, the model is both spatially and temporally-resolved and essential operating constraints are taken into account, which makes it possible to accurately capture the interaction between various technologies in space and time. A time horizon of five years with hourly resolution is considered.

The following nodes are included in the model: solar photovoltaic power plants, wind power plants, battery storage system (Li-ion), HVDC interconnector, electrolysis plants, hydrogen storage system (compressed), methanation plants, direct air capture plants, carbon dioxide storage system (liquefied), desalination plants, water storage system, methane liquefaction plants, liquefied methane storage system, liquefied methane carrier vessels, liquefied methane re-gasification plants.

Details of the model can be found in the following paper:

"Remote Renewable Hubs for Carbon-Neutral Synthetic Fuel Production", Berger et al., Frontiers in Energy Research, 9:200, 2021. doi: https://doi.org/10.3389/fenrg.2021.671279
