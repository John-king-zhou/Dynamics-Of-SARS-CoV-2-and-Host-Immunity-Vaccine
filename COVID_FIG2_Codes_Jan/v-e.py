#calculate the relationship between final state viral load and maximum immune efficacy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.spatial import ConvexHull
from scipy.integrate import odeint
from Equation import func
from E_Calculation import E,E1,E2,E12
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('error')


import matplotlib.mathtext as mathtext
mathtext.FontConstantsBase.sup1 = 0.5
mathtext.FontConstantsBase.sub1 = 0.2
mathtext.FontConstantsBase.sub2 = 0.3

time=np.arange(0,50,0.1)
Types = ['Mode 1', 'Mode 2', 'Mode 3', 'Mode 4']
try:
    dfs=pd.read_csv('v-e.csv')
except:
    dfs=[]
    for mode in range(1,5,1):
        Paras=np.loadtxt('Spl_Para%i.txt'%mode)
        Mem=[]
        Traj=[[] for j in range(38)]
        Aver_Results=[]
        N=Paras.shape[0]
        points=[]
        print('mode %i' % mode)
        for k in range(Paras.shape[0]):
            ratio=k/N
            rat_str=['>']*int(ratio*50)+['-']*(50-int(ratio*50))
            rat_str=''.join(rat_str)
            print('\r'+rat_str+'%.2f %%' %(ratio*100), end='')
            Para=Paras[k]
            initial = [0.01, 0, Para[52] / Para[62], 0, 0, Para[53] / Para[65], 0, 0, Para[159], 0, 0, 0, 0, 0, 0,
                       Para[54] / Para[70], 0, Para[160], 0, 0, 0, 0, 0, 0, Para[77] / Para[101], Para[82] / Para[102],
                       Para[84] / Para[103], (Para[88] + Para[54] / Para[70] * Para[90]) / Para[104],
                       Para[91] / Para[105],
                       Para[95] / Para[106], 0, 0]
            results=odeint(func, initial, time, args=(Para,))
            v=results[:,0]
            v[v<1e-4]=1e-4
            logv=np.log10(v)
            e=E(results, Para)
            e_a=E2(results, Para)+E12(results,Para)
            IL6=results[:, 26]
            points.append([e[0],e[70],logv[-1],e[-1],np.max(e),np.max(IL6),
                           np.max(logv),np.mean(e),np.max(e_a)])
        points=np.array(points)
        df=pd.DataFrame({'e0':points[:,0],'e7':points[:,1],'v_final':points[:,2],'e_final':points[:,3],
                         'emax':points[:,4],'IL6':points[:,5],'vmax':points[:,6],'e_aver':points[:,7],
                         'e_a':points[:,8],'Mode':[Types[mode-1] for i in range(points.shape[0])]})
        #e0=initial immune efficacy; e7=day 7 immune efficacy; v_final=day 50 viral load
        #e_final=day 50 immune efficacy;emax=maximum immune efficacy; IL6=maximum IL6;
        #vmax=maximum viral load; e_aver=average immun efficacy; e_a=maximum adaptive immune efficacy
        dfs.append(df)
    dfs=pd.concat(dfs)
    dfs.to_csv('v-e.csv')
dfs.loc[(dfs['v_final']<=-1),'v_final']=-1
dfs['vf']=10**dfs['v_final']
ggcolors=['#2CA02C','#1F77B4','#FF7F0E','#D62728',]
ho= ['Mode 1', 'Mode 2', 'Mode 3', 'Mode4',]
markers=['v','o','^','s']
sizes=[80,80,60,80]
fig,ax=plt.subplots(1,1,figsize=(5,2))
for i in range(4):
    inds=(dfs['Mode']==Types[i])
    x=dfs.loc[inds,'emax']
    y=dfs.loc[inds,'vf']
    ax.scatter(x,y,marker=markers[i],facecolor=ggcolors[i],edgecolor='w',lw=0.6,s=sizes[i],alpha=0.9,)
ax.set_xlabel('$\epsilon_{max}$',fontsize=13,labelpad=-2)
ax.set_xscale('log')
ax.set_yscale('log')
xlim=ax.get_xlim()

ylim=ax.get_ylim()
y=np.linspace(ylim[0],ylim[1],100)
x=np.ones(100)*3.6
ax.plot(x,y,c='k',zorder=10,linewidth=1)
ax.annotate(xy=(3.6,0.1),xytext=(1,1),text=r'$\epsilon=\gamma$',color='k',ha='left',
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3", color='k',lw=1),)
ax.set_yticks([0.1,1,100])
ax.set_yticklabels(['$<10^{-1}$','$10^0$','$10^2$'])
ax.set_ylabel('[nCoV]$_{final}$',labelpad=-12)
ax.set_ylim(ylim)
fig.subplots_adjust(bottom=0.2,top=0.9,left=0.18,right=0.9,hspace=0.3)
#fig.savefig('v-e-2.svg')
plt.show()
