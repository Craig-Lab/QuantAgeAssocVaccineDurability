# QuantAgeAssocVaccineDurability
 Code related to Quantifying age-associated differences in the durability of Shingrix-induced immunity using mathematical modelling 

 ### System requirements
All of the code was written and tested using the Julia programming language. The following packages are required:  
- CSV, DataFrames, Statistics, StatsBase, Distributions, Random, Dates, Printf, PlotlyJS

### Installation guide
Installation guide for Julia is available at https://docs.julialang.org/en/v1/manual/installation/ 
Install the necessary packages can be done in a REPL session. Cf. https://docs.julialang.org/en/v1.7-dev/stdlib/Pkg/

### Data
The data is available in the Data folder
	There are two CSV fiels which contain the summary statistics from GSK. One for CD4+ T cells and another for the antibody concentrations.

### Instructions for use
Download the data
After installing Julia and the necssary packages all code can be ran in a REPL session
For the code related to CD4+ T cells
Paste the correct data CSV path into VPOP and run
Afterwards run Immunity_Advantage_plot.jl to get the plot for the Immunity Advanatage
Running then CD4_traj_plot.jl will give allow for a sampling of the CD4+ T cell trajectories for the virtual patients
Finally additional_booster.jl is the simulated second booster for the oldest virtual population

