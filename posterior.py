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

def PlotChain(X,Y,l1,l2,x1,x2,y1,y2,name):
    sns.set_context("paper")
    sns.set(style="white",font_scale=2.5)
    sns.set_style("ticks", {"xtick.major.size": 4, "ytick.major.size": 4})
    
    g = sns.JointGrid(X, Y, space=0,xlim=(x1, x2), ylim=(y1, y2))
    #g = sns.JointGrid(X, Y, space=0)
    g = g.plot_joint(sns.kdeplot, cmap="Blues_d")
    sns.axlabel(l1, l2)
    g = g.plot_marginals(sns.kdeplot, shade=True)
    
    plt.savefig(name+'.pdf', bbox_inches='tight', pad_inches=0.1,dpi=300)



# 4 chain
'''
# Load the MCMC data
chain1 = np.load('../chains/chain_c_1.npz')
chain2 = np.load('../chains/chain_c_2.npz')
chain3 = np.load('../chains/chain_c_3.npz')
chain4 = np.load('../chains/chain_c_4.npz')

nh_bp   =   np.concatenate((chain1['nh_bp'],chain2['nh_bp'],chain3['nh_bp'],chain4['nh_bp']))
max_f   =   np.concatenate((chain1['max_f'],chain2['max_f'],chain3['max_f'],chain4['max_f']))
uf      =   np.concatenate((chain1['uf'],chain2['uf'],chain3['uf'],chain4['uf']))
av      =   np.concatenate((chain1['av'],chain2['av'],chain3['av'],chain4['av']))
v_H     =   np.concatenate((chain1['v_H'],chain2['v_H'],chain3['v_H'],chain4['v_H']))
'''

# 12 chain
letter = 'c'


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

nh_bp   =   np.concatenate((chain1['nh_bp'],chain2['nh_bp'],chain3['nh_bp'],chain4['nh_bp'],chain5['nh_bp'],chain6['nh_bp'],chain7['nh_bp'],chain8['nh_bp'],chain9['nh_bp'],chain10['nh_bp'],chain11['nh_bp'],chain12['nh_bp']))
max_f   =   np.concatenate((chain1['max_f'],chain2['max_f'],chain3['max_f'],chain4['max_f'],chain5['max_f'],chain6['max_f'],chain7['max_f'],chain8['max_f'],chain9['max_f'],chain10['max_f'],chain11['max_f'],chain12['max_f']))
uf      =   np.concatenate((chain1['uf'],chain2['uf'],chain3['uf'],chain4['uf'],chain5['uf'],chain6['uf'],chain7['uf'],chain8['uf'],chain9['uf'],chain10['uf'],chain11['uf'],chain12['uf']))
av      =   np.concatenate((chain1['av'],chain2['av'],chain3['av'],chain4['av'],chain5['av'],chain6['av'],chain7['av'],chain8['av'],chain9['av'],chain10['av'],chain11['av'],chain12['av']))
v_H     =   np.concatenate((chain1['v_H'],chain2['v_H'],chain3['v_H'],chain4['v_H'],chain5['v_H'],chain6['v_H'],chain7['v_H'],chain8['v_H'],chain9['v_H'],chain10['v_H'],chain11['v_H'],chain12['v_H']))

# Arange the data into pandas format to be compatible with corner.py
data = np.array([nh_bp,max_f/1e-12,uf,av,v_H]).T
columns = ['N','M','uf','av','v_H']
df = pd.DataFrame(data,columns=columns)

# Plot the posterior distributions.

fontlabel_size  = 13
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
params = {'backend': 'wxAgg', 'lines.markersize' : 2, 'axes.labelsize': fontlabel_size, 'font.size': fontlabel_size, 'legend.fontsize': fontlabel_size, 'font.family': 'Computer Modern'}
plt.rcParams.update(params)

# I'd like to have TeX font on the axis. Sadly the above line does not work.
#plt.rc('text', usetex=True)

figure = corner.corner(data,labels=[r"$\logN_{\mathrm{H}}$", r"$F_{\mathrm{max}}/1\times10^{-12}$", r"$\mathrm{uf}$",
                                     r"$\mathrm{av}$",r"$v\mathrm{_H}$"],#$\mathrm{v_X}/1\times10^{-4}$
                                     quantiles=[0.16, 0.8413],
                                     levels=(1-np.exp(-0.5),),
                                     #truths=[19.42,9.2,2.915,0.03937,33.06,12.],
                                     range=[(19.24,19.49),(2,24),(2.7,3.38),(0.01,0.09),(26.8,38)],
                                     #fill_contours=True,
                                     #ret=True,
                                     bins=40,
                                     smooth=0.8,
                                     show_titles=True, title_kwargs={"fontsize": 13},
                                     label_kwargs = {"fontsize": 18},
                                     plot_contours=True,
                                     verbose=False,
                                     use_math_text=True)

figure.savefig("../plots/mcmc_"+letter+".pdf")

# Uncomment to print uncertainties
'''
print Uncertainties(nh_bp)
print Uncertainties(max_f)
print Uncertainties(uf)
print Uncertainties(av)
print Uncertainties(v_X)
'''
