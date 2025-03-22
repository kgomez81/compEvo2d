# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 01:56:23 2025

@author: Owner
"""
import matplotlib.pyplot as plt

# new class for debugging based on final

evoSnapshotFile = 'D:\\Documents\\GitHub\\compEvo2d\\figureScripts\\outputs\\sim_bEvo_DRE_Fig2B\\sim_Fig2B_T1E9_snpsht_param_03A_DRE_bEvo_20250321_1516.pickle'
with open(evoSnapshotFile, 'rb') as file:
    # Serialize and write the variable to the file
    loaded_data = pickle.load(file)
    
evoSimTest1 = simDre.simDREClass(loaded_data)

# factorUT = (1-np.sum(evoSimTest.nij)/evoSimTest.mcModel.params['T'])
# evoSimTest.mcModel.params['Ua']*evoSimTest.nij*evoSimTest.get_bij()*factorUT

idxEq = evoSimTest1.mcModel.get_mc_stable_state_idx()

# Ni  = evoSimTest1.mcModel.eq_Ni[idxEq]
# Uai = evoSimTest1.mcModel.params['Ua']
# sai = evoSimTest1.mcModel.sc_i[idxEq]
# Uci = evoSimTest1.mcModel.params['Uc']
# sci = evoSimTest1.mcModel.sc_i[idxEq]

# Test = 1/(Ni*si*Ui)
# Tswp = (1/si)*np.log(Ni*si)

# va = sai**2*(2*np.log(Ni*sai)-np.log(sai/Uai))/(np.log(sai/Uai))**2
# vc = sci**2*(2*np.log(Ni*sci)-np.log(sci/Uci))/(np.log(sci/Uci))**2

# qa = 2*np.log(Ni*sai)/np.log(sai/Uai)
# qc = 2*np.log(Ni*sci)/np.log(sci/Uci)

fig,ax = plt.subplots(1,1)
ax.plot(evoSimTest1.mcModel.state_i,evoSimTest1.mcModel.va_i,c='blue')
ax.plot(evoSimTest1.mcModel.state_i,evoSimTest1.mcModel.vc_i,c='red')
ax.plot(evoSimTest1.mcModel.state_i,evoSimTest1.mcModel.ve_i,c='black')


#%% 
evoSnapshotFile = 'D:\\Documents\\GitHub\\compEvo2d\\figureScripts\\outputs\\sim_bEvo_DRE_Fig2B\\sim_Fig2B_T1E12_snpsht_param_03B_DRE_bEvo_20250321_1517.pickle'
with open(evoSnapshotFile, 'rb') as file:
    # Serialize and write the variable to the file
    loaded_data = pickle.load(file)
    
evoSimTest2 = simDre.simDREClass(loaded_data)

# factorUT = (1-np.sum(evoSimTest.nij)/evoSimTest.mcModel.params['T'])
# evoSimTest.mcModel.params['Ua']*evoSimTest.nij*evoSimTest.get_bij()*factorUT
idxEq = evoSimTest2.mcModel.get_mc_stable_state_idx()

# Ni = evoSimTest.mcModel.eq_Ni[idxEq]
# Uai = evoSimTest.mcModel.params['Ua']
# sai = evoSimTest.mcModel.sc_i[idxEq]
# Uci = evoSimTest.mcModel.params['Uc']
# sci = evoSimTest.mcModel.sc_i[idxEq]

# Test = 1/(Ni*si*Ui)
# Tswp = (1/si)*np.log(Ni*si)

# va = sai**2*(2*np.log(Ni*sai)-np.log(sai/Uai))/(np.log(sai/Uai))**2
# vc = sci**2*(2*np.log(Ni*sci)-np.log(sci/Uci))/(np.log(sci/Uci))**2

# qa = 2*np.log(Ni*sai)/np.log(sai/Uai)
# qc = 2*np.log(Ni*sci)/np.log(sci/Uci)

fig,ax = plt.subplots(1,1)
ax.plot(evoSimTest1.mcModel.state_i,evoSimTest1.mcModel.va_i,c='blue')
ax.plot(evoSimTest1.mcModel.state_i,evoSimTest1.mcModel.vc_i,c='red')
ax.plot(evoSimTest2.mcModel.state_i,evoSimTest2.mcModel.va_i,c='cyan')
ax.plot(evoSimTest2.mcModel.state_i,evoSimTest2.mcModel.vc_i,c='magenta')

evoSimTest1.mcModel.get_mc_stable_state_idx()
evoSimTest2.mcModel.get_mc_stable_state_idx()
