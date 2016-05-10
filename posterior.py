import numpy as np
import matplotlib.pyplot as plt

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

def Plot_Chain(P1, P2,name):
  fig = plt.figure()
  plt.clf()
  
  x_max = P1.max()
  x_min = P1.min()

  y_max = P2.max()
  y_min = P2.min()
  
  # Top plot
  top = plt.subplot2grid((3,3), (0, 0), colspan=2)
  top.hist(P1,bins=30)
  plt.axvline(Distrib(P1)[0],color="red",lw=2)
  plt.axvline(Distrib(P1)[1],color="red",lw=2,linestyle='--')
  plt.axvline(Distrib(P1)[2],color="red",lw=2,linestyle='--')
  top.get_xaxis().set_ticklabels([])
  plt.xlim(x_min,x_max)
  plt.minorticks_on()
 
  # Right hand side plot
  right = plt.subplot2grid((3,3), (1, 2), rowspan=2)
  right.hist(P2,orientation='horizontal',bins=30)
  plt.axhline(Distrib(P2)[0],color="red",lw=2)
  plt.axhline(Distrib(P2)[1],color="red",lw=2,linestyle='--')
  plt.axhline(Distrib(P2)[2],color="red",lw=2,linestyle='--')
  right.get_yaxis().set_ticklabels([])
  #right.xaxis.set_major_locator(LinearLocator(5))
  plt.ylim(y_min,y_max)
  plt.minorticks_on()
  
  # Center plot
  center = plt.subplot2grid((3,3), (1, 0), rowspan=2, colspan=2)
  center.hist2d(P1,P2,bins=30)
  plt.xlim(x_min,x_max)
  plt.minorticks_on()
  
  # Corner plot
  corner = plt.subplot2grid((3,3), (0, 2))
  corner.get_xaxis().set_ticklabels([])
  corner.get_yaxis().set_ticklabels([])
  corner.plot(P1,P2,'-k')
  plt.minorticks_on()
  plt.ylim(y_min,y_max)
  
  plt.savefig(name+'_param.png',paper='a4',orientation='landscape',bbox_inches='tight', pad_inches=0.1)

#chaint = np.load('chaint.npz')

chain1 = np.load('chain1.npz')
chain2 = np.load('chain2.npz')
chain3 = np.load('chain3.npz')
chain4 = np.load('chain4.npz')

#nh_bp   =   np.concatenate((chain1['nh_bp'],chain2['nh_bp'],chain3['nh_bp']))
#max_f   =   np.concatenate((chain1['max_f'],chain2['max_f'],chain3['max_f']))

nh_bp   =   np.concatenate((chain1['nh_bp'],chain2['nh_bp'],chain3['nh_bp'],chain4['nh_bp']))
max_f   =   np.concatenate((chain1['max_f'],chain2['max_f'],chain3['max_f'],chain4['max_f']))

uf   =   np.concatenate((chain1['uf'],chain2['uf'],chain3['uf'],chain4['uf']))
av   =   np.concatenate((chain1['av'],chain2['av'],chain3['av'],chain4['av']))

slope   =   np.concatenate((chain1['slope'],chain2['slope'],chain3['slope'],chain4['slope']))

print Uncertainties(nh_bp)
print Uncertainties(max_f)
print Uncertainties(uf)
print Uncertainties(av)
print Uncertainties(slope)

Plot_Chain(nh_bp, max_f,'1')
Plot_Chain(uf, av,'2')
Plot_Chain(nh_bp, slope,'3')

#plt.show()
