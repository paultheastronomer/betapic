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

# Load the MCMC data
chain1 = np.load('../chains/chain_c_1.npz')
chain2 = np.load('../chains/chain_c_2.npz')
chain3 = np.load('../chains/chain_c_3.npz')
chain4 = np.load('../chains/chain_c_4.npz')

nh_bp   =   np.concatenate((chain1['nh_bp'],chain2['nh_bp'],chain3['nh_bp'],chain4['nh_bp']))
max_f   =   np.concatenate((chain1['max_f'],chain2['max_f'],chain3['max_f'],chain4['max_f']))
uf      =   np.concatenate((chain1['uf'],chain2['uf'],chain3['uf'],chain4['uf']))
av      =   np.concatenate((chain1['av'],chain2['av'],chain3['av'],chain4['av']))
v_X     =   np.concatenate((chain1['v_X'],chain2['v_X'],chain3['v_X'],chain4['v_X']))
nh_X    =   np.concatenate((chain1['nh_X'],chain2['nh_X'],chain3['nh_X'],chain4['nh_X']))

# Arange the data into pandas format to be compatible with corner.py
data = np.array([nh_bp,max_f/1e-11,uf,av,v_X,nh_X]).T
columns = ['N','M','uf','av','v_X','nh_X']
df = pd.DataFrame(data,columns=columns)

# Plot the posterior distributions.

fontlabel_size  = 13
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
params = {'backend': 'wxAgg', 'lines.markersize' : 2, 'axes.labelsize': fontlabel_size, 'font.size': fontlabel_size, 'legend.fontsize': fontlabel_size, 'font.family': 'Computer Modern'}
plt.rcParams.update(params)

# I'd like to have TeX font on the axis. Sadly the above line does not work.
#plt.rc('text', usetex=True)

summ = (10**(nh_X) + 10**(nh_bp))
lsum = np.log10(summ)

data = np.array([v_X,lsum]).T
columns = ['N','nh_X']
df = pd.DataFrame(data,columns=columns)


figure = corner.corner(data,labels=[r"$\mathrm{v_X}$", r"$\logN_{\mathrm{H\beta}} + \logN_{\mathrm{Hx}}$"],
                                     quantiles=[0.16, 0.8413],
                                     levels=(1-np.exp(-0.5),),
                                     #truths=[19.42,9.2,2.915,0.03937,33.06,12.],
                                     range=[(10,70),(19.2,19.6)],
                                     #fill_contours=True,
                                     #ret=True,
                                     bins=30,
                                     smooth=0.8,
                                     #show_titles=True, title_kwargs={"fontsize": 13},
                                     label_kwargs = {"fontsize": 18},
                                     plot_contours=True,
                                     verbose=False,
                                     use_math_text=True)

figure.savefig("../plots/ColDens.pdf")
'''
plt.hist2d(v_X,lsum,bins=50)
#H = np.ones((4, 4)).cumsum().reshape(4, 4)
#plt.imshow(H,interpolation='nearest')
plt.xlabel(r'Radial Velocity [km/s] (Heliocentric)')
plt.ylabel(r'$(N_{+20.5} + N_{+X})$')
plt.savefig('rel.pdf')
plt.show()
'''

# Uncomment to print uncertainties
'''
print Uncertainties(nh_bp)
print Uncertainties(max_f)
print Uncertainties(uf)
print Uncertainties(av)
print Uncertainties(v_X)
'''
