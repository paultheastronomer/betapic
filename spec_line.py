import matplotlib.pyplot as plt
import numpy as np

def Bin_data(x,y1,y2,y3,y4,bin_pnts):
    bin_size = int(len(x)/bin_pnts)
    bins = np.linspace(x[0], x[-1], bin_size)
    digitized = np.digitize(x, bins)
    bin_y1 = np.array([y1[digitized == i].mean() for i in range(0, len(bins))])
    bin_y2 = np.array([y2[digitized == i].mean() for i in range(0, len(bins))])
    bin_y3 = np.array([y3[digitized == i].mean() for i in range(0, len(bins))])
    bin_y4 = np.array([y4[digitized == i].mean() for i in range(0, len(bins))])
    return bins, bin_y1, bin_y2, bin_y3, bin_y4

def wave2RV(Wave,rest_wavelength,RV_BP):
    c = 299792458
    rest_wavelength = rest_wavelength*(RV_BP*1.e3)/c + rest_wavelength # Convert to beta pic reference frame
    delta_wavelength = Wave-rest_wavelength
    RV = ((delta_wavelength/rest_wavelength)*c)/1.e3	# km/s
    return RV

def findCenter(w,l):
    for i in range(len(w)):
      if w[i] > l:
        if abs(l - w[i-1]) < abs(l - w[i]):
          ans = i-1
          break
        else:
          ans = i
          break
    return ans

def main():

    # Parameters you can change
    #====================================================
    species             = 'OV'     # Name of species
    line_of_interest    = 1218.3440 # Wavelength of line
    RV_BP               = 20.5      # RV of Beta Pic
    width               = 1500       # [-2*width:2*width]
    bin_pnts            = 5         # Number of points to bin
    norm1               = 160        # Start norm region
    norm2               = 170        # End norm region
    #====================================================
    
    #W, F0, F1, F2, F3 = np.loadtxt('/home/paw/science/betapic/data/HST/dat/B.dat',unpack=True)
    W, F0, F1, F2, F3, E0, E1, E2, E3 = np.loadtxt('/home/paw/science/betapic/data/HST/dat/B.dat',unpack=True)

    mid_pnt = findCenter(W,line_of_interest)

    W   = W[mid_pnt-width:mid_pnt+width]
    F0  = F0[mid_pnt-width:mid_pnt+width]
    F1  = F1[mid_pnt-width:mid_pnt+width]
    F2  = F2[mid_pnt-width:mid_pnt+width]
    F3  = F3[mid_pnt-width:mid_pnt+width]
    
    
    RV = wave2RV(W,line_of_interest,RV_BP)
  
    RV_bin, F0_bin, F1_bin, F2_bin, F3_bin	=	Bin_data(RV,F0,F1,F2,F3,bin_pnts)

    fig = plt.figure(figsize=(14,10))
    
    # Fancy customisation to make the plot look nice
    #================================================
    fontlabel_size = 18
    tick_size = 18
    params = {'backend': 'wxAgg', 'lines.markersize' : 2, 'axes.labelsize': fontlabel_size, 'font.size': fontlabel_size, 'legend.fontsize': fontlabel_size, 'xtick.labelsize': tick_size, 'ytick.labelsize': tick_size, 'text.usetex': True}
    plt.rcParams.update(params)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.unicode'] = True 
    #================================================

    #plt.text(-1100,0.5,'Lots of OI contamination',ha="center")
    
    # Show normalised region
    plt.text((RV_bin[norm1]+RV_bin[norm2])/2.,0.78,'Normalised region',ha='center')
    plt.plot([RV_bin[norm1],RV_bin[norm2]], [0.83,0.83],lw=1.2,color="black")
    plt.plot([RV_bin[norm1],RV_bin[norm1]], [0.83,0.85],lw=1.2,color="black")
    plt.plot([RV_bin[norm2],RV_bin[norm2]], [0.83,0.85],lw=1.2,color="black")

    # Show the name of line
    plt.text(0,1.1,species+' at '+str(line_of_interest)+' \AA')
    plt.plot([0,0],[1.0,1.09],color='black',lw=1.2)
    
    # Plot the spectra
    plt.step(RV_bin, F0_bin/np.median(F0_bin[norm1:norm2]),color='#FF281C',lw=1.5,label='2014')
    #plt.step(RV_bin*-1, F0_bin/np.median(F0_bin[norm1:norm2]),color='blue',lw=1.5,label='flipped')
    
    plt.step(RV_bin, F1_bin/np.median(F1_bin[norm1:norm2]),color='#FF9303',lw=1.5,label='2015v1')
    #plt.step(RV_bin*-1, F1_bin/np.median(F1_bin[norm1:norm2]),color='blue',lw=1.5,label='flipped')
    
    plt.step(RV_bin, F2_bin/np.median(F2_bin[norm1:norm2]),color='#0386FF',lw=1.5,label='2015v2')
    #plt.step(RV_bin*-1, F2_bin/np.median(F2_bin[norm1:norm2]),color='blue',lw=1.5,label='flipped')
    
    plt.step(RV_bin, F3_bin/np.median(F3_bin[norm1:norm2]),color='#00B233',lw=1.5,label='2016v3')
    #plt.step(RV_bin*-1, F3_bin/np.median(F3_bin[norm1:norm2]),color='blue',lw=1.5,label='flipped')
    
    # Place a legend in the lower right
    plt.legend(loc='lower right', numpoints=1)
    
    # Add labels to the axis
    plt.xlabel('RV [km/s]')
    plt.ylabel('Normalised Flux')
    
    plt.xlim(-500,500)
    plt.ylim(0,5)
    
    # Produce a .pdf
    fig.tight_layout()
    plt.show()
    #plt.savefig(species+'_'+str(line_of_interest)+'.pdf', bbox_inches='tight', pad_inches=0.1,dpi=300)      

if __name__ == '__main__':
    main()
