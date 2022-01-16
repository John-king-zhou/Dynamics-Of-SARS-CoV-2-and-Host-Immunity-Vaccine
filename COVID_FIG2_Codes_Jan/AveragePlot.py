#Type_Specific: function for plotting the illustration of mode 1~4 with mean+std (figure 2A)
import numpy as np
from matplotlib.ticker import MultipleLocator

xmajorLocator = MultipleLocator(10)

#axes=plotting subplots; time=integration time point array;
# Mean=list containing the trajectory of the 30 variables; #color=str of rgb or name for a color;
# n=int mode-1
def Type_Specific(ax,time,Mean,Std,color,n):
    labels=['Mode 1','Mode 2','Mode 3','Mode 4']
    ulim=[52,4.5,1.4,1.3,3.5,3600]
    llim=[0,-2.1,-0.1,-1.5,0,-100]
    if n==0:
        ax[0].set_title('Epithelial Cell',fontsize=13)
        ax[1].set_title('log$_{10}$ Viral Load',fontsize=13)
        ax[2].set_title('Ag Presentation',fontsize=13)
        ax[3].set_title('log$_{10}$ CD8$^+$T',fontsize=13)
        ax[4].set_title('log$_{10}$ Abs',fontsize=13)
        ax[5].set_title('IL-6',fontsize=15)
    indices=[[2,],[0,],[4,],[32,],[33,],[26,]]
    for i in range(6):
        mean=np.sum(Mean[indices[i],:],axis=0)
        std=np.sum(Std[indices[i],:],axis=0)
        ax[i].plot(time, mean, linewidth=3, c=color)
        ax[i].fill_between(time, mean-std, mean+std, facecolor=color, alpha=0.2)
        ax[i].set_ylim(llim[i],ulim[i])
        yticks=np.arange(1,4,1)*(ulim[i]-llim[i])/4+llim[i]
        ax[i].set_yticks(yticks)
        ax[i].set_yticklabels([])
        ax[i].set_xlim([0, 28])
        ax[i].set_xticks([0, 7, 14, 21, 28])
        ax[i].set_xticklabels([])
    ax[0].set_ylabel(labels[n], fontsize=13)

    if n==3:
        for j in range(6):
            ax[j].set_xticks([0,7,14,21,28])
            ax[j].set_xticklabels([0,1,2,3,4])
            ax[j].tick_params(labelsize=14)
