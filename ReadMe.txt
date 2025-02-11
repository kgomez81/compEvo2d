#####################################
#####################################
Overview of Python Code for "Improved population fitness in a larger habitat is reduced or even reversed by clonal interference from a zero-sum trait"
#####################################
#####################################

Author: Kevin Gomez

#####################################
Instructions for generating figures
#####################################
1. Install Anaconda (64-bit) 22.11.1 with Python 3.9.12 64-bit and Spyder IDE 5.1.5. If not available, use Anaconda 2022.10 with Python 3.9.13 and Spyder IDE 5.2.2
	https://docs.anaconda.com/anaconda/release-notes/

2. Download or clone a copy of the repository https://github.com/kgomez81/compEvo2d

3. In Spyder, navigate to compEvo2d/figureScripts/

4. Create an "outputs" within  compEvo2d/figureScripts/

5. Run any of the scripts listed below to create figures provided in "Improved population fitness in a larger habitat is reduced or even reversed by clonal interference from a zero-sum trait."

#####################################
List of Scripts for Figures
#####################################

Manuscript
⦁ Figure 1: compEvo2d/figureScripts/fig_bEvo_DRE_MC_vIntersections.py
⦁ Figure 2: compEvo2d/figureScripts/fig_bEvo_DRE_Decr_Incr_AbsFit_varyT_yAxis.py
⦁ Figure 3: compEvo2d/figureScripts/fig_bEvo_DRE_Rho_vs_ScSa_UaUc_VaryUaCp.py
⦁ Figure 4 (Appendix C):  compEvo2d/figureScripts/fig_bEvo_DRE_Appendix_quasiEquilibriumDensity.py

Supplement
⦁ Supplementary Figure 1A: compEvo2d/figureScripts/fig_bEvo_DRE_Rho_vs_ScSa_UaUc_VaryUaCp_SmallT.py
⦁ Supplementary Figure 1B: compEvo2d/figureScripts/fig_bEvo_DRE_Rho_vs_ScSa_UaUc_VaryUaCp_LargeT.py
⦁ Supplementary Figure 2A: fig_bEvo_DRE_Rho_vs_ScSa_UaUc_VaryUaCp_HighAlpha.py
⦁ Supplementary Figure 2B: fig_bEvo_DRE_Rho_vs_ScSa_UaUc_VaryUaCp_UnbndBVals.py
⦁ Supplementary Figure 3A: compEvo2d/figureScripts/fig_bEvo_DRE_Rho_vs_ScSa_UaUc_VaryUcSb0.py
⦁ Supplementary Figure 3B: compEvo2d/figureScripts/fig_bEvo_DRE_Rho_vs_ScSa_UaUc_VaryUcAlpha.py

#####################################
Custom Libraries and Supporting Files
#####################################
⦁ compEvo2d/evoLibraries/LotteryModel/
Includes functions specific to Bertram & Masel Lotter model calculations.

⦁ compEvo2d/evoLibraries/MarkovChain/
Includes classes and supporting functions to generate Markov Chain models from parameters provided in input file for both Diminshing Returns Epistasis and Running Out of Mutations cases. 

⦁ compEvo2d/evoLibraries/RateOfAdapt/
Includes functions for evaluation of rates of adaption. 

⦁ compEvo2d/evoLibraries/constants.py
Includes constants used across various libraries.

⦁ compEvo2d/evoLibraries/evoObjects.py
Includes class definition to package DRE and RM model parameters.

#####################################
List of Inputs Files
#####################################
See file parameterKey.ods for additional details on model parameters included in each input file; file can be view with the free to download LibreOffice 7.1 Calculator application.

⦁ Figure 1A: compEvo2d/figureScripts/inputs/evoExp_DRE_bEvo_01_parameters.csv
⦁ Figure 1B: compEvo2d/figureScripts/inputs/evoExp_DRE_bEvo_02_parameters.csv
⦁ Figure 2A: compEvo2d/figureScripts/inputs/evoExp_DRE_bEvo_03_parameters.csv
⦁ Figure 2B: compEvo2d/figureScripts/inputs/evoExp_DRE_bEvo_04_parameters.csv
⦁ Supp. Figure 2B: compEvo2d/figureScripts/inputs/evoExp_DRE_bEvo_05_parameters.csv
⦁ Figure 3: compEvo2d/figureScripts/inputs/evoExp_DRE_bEvo_06_parameters.csv
⦁ Supp. Figure 1A: compEvo2d/figureScripts/inputs/evoExp_DRE_bEvo_07_parameters.csv
⦁ Supp. Figure 1B: compEvo2d/figureScripts/inputs/evoExp_DRE_bEvo_08_parameters.csv
⦁ (Not Used) compEvo2d/figureScripts/inputs/evoExp_DRE_bEvo_09_parameters.csv
⦁ Supp. Figure 2A: compEvo2d/figureScripts/inputs/evoExp_DRE_bEvo_10_parameters.csv
⦁ Supp. Figure 3A: compEvo2d/figureScripts/inputs/evoExp_DRE_bEvo_11_parameters.csv
⦁ Supp. Figure 3B: compEvo2d/figureScripts/inputs/evoExp_DRE_bEvo_12_parameters.csv


