import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import corner
import pandas as pd
import sys

def Distrib(x):
   '''Finds median and 68% interval of array x.'''
   y    = sorted(x)
   up   = y[int(0.8413*len(y))]
   down = y[int(0.1587*len(y))]
   med  = y[int(0.5*len(y))]
   
   return med,up,down   

def Uncertainties(x):
   '''Finds median and 68% interval of array x.'''
   y    = sorted(x)
   up   = y[int(0.8413*len(y))]
   down = y[int(0.1587*len(y))]
   med  = y[int(0.5*len(y))]
   
   return med,up-med,med-down   

# Select which posterior distributions to use.
letter = 'O'

# Load the MCMC data
chain1 = np.load('../chains/chain_'+letter+'_1.npz')
chain2 = np.load('../chains/chain_'+letter+'_2.npz')
chain3 = np.load('../chains/chain_'+letter+'_3.npz')
chain4 = np.load('../chains/chain_'+letter+'_4.npz')
chain5 = np.load('../chains/chain_'+letter+'_5.npz')
chain6 = np.load('../chains/chain_'+letter+'_6.npz')
chain7 = np.load('../chains/chain_'+letter+'_7.npz')
chain8 = np.load('../chains/chain_'+letter+'_8.npz')
chain9 = np.load('../chains/chain_'+letter+'_9.npz')
chain10 = np.load('../chains/chain_'+letter+'_10.npz')
chain11 = np.load('../chains/chain_'+letter+'_11.npz')
chain12 = np.load('../chains/chain_'+letter+'_12.npz')
chain13 = np.load('../chains/chain_'+letter+'_13.npz')
chain14 = np.load('../chains/chain_'+letter+'_14.npz')
chain15 = np.load('../chains/chain_'+letter+'_15.npz')
chain16 = np.load('../chains/chain_'+letter+'_16.npz')
chain17 = np.load('../chains/chain_'+letter+'_17.npz')
chain18 = np.load('../chains/chain_'+letter+'_18.npz')
chain19 = np.load('../chains/chain_'+letter+'_19.npz')
chain20 = np.load('../chains/chain_'+letter+'_20.npz')
chain21 = np.load('../chains/chain_'+letter+'_21.npz')
chain22 = np.load('../chains/chain_'+letter+'_22.npz')
chain23 = np.load('../chains/chain_'+letter+'_23.npz')
chain24 = np.load('../chains/chain_'+letter+'_24.npz')

nh_bp   =   np.concatenate((chain1['nh_bp'],chain2['nh_bp'],chain3['nh_bp'],\
chain4['nh_bp'],chain5['nh_bp'],chain6['nh_bp'],chain7['nh_bp'],\
chain8['nh_bp'],chain9['nh_bp'],chain10['nh_bp'],chain11['nh_bp'],\
chain12['nh_bp'],chain13['nh_bp'],chain14['nh_bp'],chain15['nh_bp'],\
chain16['nh_bp'],chain17['nh_bp'],chain18['nh_bp'],chain19['nh_bp'],\
chain20['nh_bp'],chain21['nh_bp'],chain22['nh_bp'],chain23['nh_bp'],\
chain24['nh_bp']))

max_f   =   np.concatenate((chain1['max_f'],chain2['max_f'],chain3['max_f'],\
chain4['max_f'],chain5['max_f'],chain6['max_f'],chain7['max_f'],\
chain8['max_f'],chain9['max_f'],chain10['max_f'],chain11['max_f'],\
chain12['max_f'],chain13['max_f'],chain14['max_f'],chain15['max_f'],\
chain16['max_f'],chain17['max_f'],chain18['max_f'],chain19['max_f'],\
chain20['max_f'],chain21['max_f'],chain22['max_f'],chain23['max_f'],\
chain24['max_f']))

uf   =   np.concatenate((chain1['uf'],chain2['uf'],chain3['uf'],\
chain4['uf'],chain5['uf'],chain6['uf'],chain7['uf'],\
chain8['uf'],chain9['uf'],chain10['uf'],chain11['uf'],\
chain12['uf'],chain13['uf'],chain14['uf'],chain15['uf'],\
chain16['uf'],chain17['uf'],chain18['uf'],chain19['uf'],\
chain20['uf'],chain21['uf'],chain22['uf'],chain23['uf'],\
chain24['uf']))

av   =   np.concatenate((chain1['av'],chain2['av'],chain3['av'],\
chain4['av'],chain5['av'],chain6['av'],chain7['av'],\
chain8['av'],chain9['av'],chain10['av'],chain11['av'],\
chain12['av'],chain13['av'],chain14['av'],chain15['av'],\
chain16['av'],chain17['av'],chain18['av'],chain19['av'],\
chain20['av'],chain21['av'],chain22['av'],chain23['av'],\
chain24['av']))

#'''
v_H   =   np.concatenate((chain1['v_H'],chain2['v_H'],chain3['v_H'],\
chain4['v_H'],chain5['v_H'],chain6['v_H'],chain7['v_H'],\
chain8['v_H'],chain9['v_H'],chain10['v_H'],chain11['v_H'],\
chain12['v_H'],chain13['v_H'],chain14['v_H'],chain15['v_H'],\
chain16['v_H'],chain17['v_H'],chain18['v_H'],chain19['v_H'],\
chain20['v_H'],chain21['v_H'],chain22['v_H'],chain23['v_H'],\
chain24['v_H']))
#'''

'''
v_X   =   np.concatenate((chain1['v_X'],chain2['v_X'],chain3['v_X'],\
chain4['v_X'],chain5['v_X'],chain6['v_X'],chain7['v_X'],\
chain8['v_X'],chain9['v_X'],chain10['v_X'],chain11['v_X'],\
chain12['v_X'],chain13['v_X'],chain14['v_X'],chain15['v_X'],\
chain16['v_X'],chain17['v_X'],chain18['v_X'],chain19['v_X'],\
chain20['v_X'],chain21['v_X'],chain22['v_X'],chain23['v_X'],\
chain24['v_X']))

nh_X   =   np.concatenate((chain1['nh_X'],chain2['nh_X'],chain3['nh_X'],\
chain4['nh_X'],chain5['nh_X'],chain6['nh_X'],chain7['nh_X'],\
chain8['nh_X'],chain9['nh_X'],chain10['nh_X'],chain11['nh_X'],\
chain12['nh_X'],chain13['nh_X'],chain14['nh_X'],chain15['nh_X'],\
chain16['nh_X'],chain17['nh_X'],chain18['nh_X'],chain19['nh_X'],\
chain20['nh_X'],chain21['nh_X'],chain22['nh_X'],chain23['nh_X'],\
chain24['nh_X']))
'''
nh_ISM   =   np.concatenate((chain1['nh_ISM'],chain2['nh_ISM'],chain3['nh_ISM'],\
chain4['nh_ISM'],chain5['nh_ISM'],chain6['nh_ISM'],chain7['nh_ISM'],\
chain8['nh_ISM'],chain9['nh_ISM'],chain10['nh_ISM'],chain11['nh_ISM'],\
chain12['nh_ISM'],chain13['nh_ISM'],chain14['nh_ISM'],chain15['nh_ISM'],\
chain16['nh_ISM'],chain17['nh_ISM'],chain18['nh_ISM'],chain19['nh_ISM'],\
chain20['nh_ISM'],chain21['nh_ISM'],chain22['nh_ISM'],chain23['nh_ISM'],\
chain24['nh_ISM']))

# Arange the data into pandas format to be compatible with corner.py
print Uncertainties(nh_bp)
print Uncertainties(max_f)
#print Uncertainties(uf)
print Uncertainties(1./uf)
#print Uncertainties(av)
print Uncertainties(av*(1./uf)*np.sqrt(2))
print Uncertainties(v_H)
print Uncertainties(nh_ISM)
#sys.exit()
data = np.array([nh_bp,max_f/1e-10,1./uf,av*(1./uf)*np.sqrt(2),v_H,nh_ISM]).T
#data = np.array([nh_bp,max_f/1e-9,uf,av,v_H,nh_ISM]).T
#columns = ['N','M','uf','av','v_H','nh_ISM']

data = np.array([nh_bp,max_f/1e-10,v_H-20.5,nh_ISM]).T
columns = ['N','M','v_H','nh_ISM']
df = pd.DataFrame(data,columns=columns)

# Plot the posterior distributions.

fontlabel_size  = 14
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
params = {'backend': 'wxAgg', 'lines.markersize' : 2, 'axes.labelsize': fontlabel_size, 'font.size': fontlabel_size, 'legend.fontsize': fontlabel_size, 'font.family': 'Computer Modern'}
plt.rcParams.update(params)

# I'd like to have TeX font on the axis. Sadly the above line does not work.
#plt.rc('text', usetex=True)


figure = corner.corner(data,labels=[r"$\log(N_{\mathrm{H}})_{\mathrm{disk}}$", r"$F_{\mathrm{max}}/1\times10^{-10}$",r"$v_{\mathrm{disk}}$",r"$\log(N_{\mathrm{H}})_{\mathrm{ISM}}$"],#$\mathrm{v_X}/1\times10^{-4}$
                                     quantiles=[0.16, 0.5,0.8413],
                                     levels=(1-np.exp(-0.5),),
                                     #truths=[18.522,4.358,0.0031516,0.01880,64.629,18.1306],
                                     range=[(18.20,18.92),(3.2,5),(20.0,65.0),(17.8,18.4)],
                                     #fill_contours=True,
                                     #ret=True,
                                     bins=40,
                                     smooth=0.8,
                                     #show_titles=True, title_kwargs={"fontsize": 13},
                                     label_kwargs = {"fontsize": 22},
                                     plot_contours=True,
                                     verbose=False,
                                     use_math_text=True)
#plt.show()


'''
figure = corner.corner(data,labels=[r"$\log(N_{\mathrm{H}})_{\mathrm{disk}}$", r"$F_{\mathrm{max}}/1\times10^{-10}$", r"$\sigma_{\mathrm{Ly}\alpha}$",
                                     r"$\gamma_{\mathrm{Ly}\alpha}$",r"$v_{\mathrm{disk}}$",r"$\log(N_{\mathrm{H}})_{\mathrm{ISM}}$"],#$\mathrm{v_X}/1\times10^{-4}$
                                     quantiles=[0.16, 0.5,0.8413],
                                     levels=(1-np.exp(-0.5),),
                                     #truths=[18.522,4.358,0.0031516,0.01880,64.629,18.1306],
                                     range=[(18.20,18.92),(3.2,5),(0.003184,0.003207),(0.0219,0.0235),(40.0,90.0),(17.8,18.4)],
                                     #fill_contours=True,
                                     #ret=True,
                                     bins=40,
                                     smooth=0.8,
                                     #show_titles=True, title_kwargs={"fontsize": 13},
                                     label_kwargs = {"fontsize": 18},
                                     plot_contours=True,
                                     verbose=False,
                                     use_math_text=True)
plt.show()
'''
figure.savefig("../plots/mcmc_"+letter+".pdf")
