#plotting the ensemble-averaged curves of mode 1, 2, 3 and asymptomatic. (figure S3)
#call AveragePlot
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

def encircle(x,y, ax, **kw):

    p = np.c_[x,y]

    hull = ConvexHull(p)

    poly = plt.Polygon(p[hull.vertices,:], **kw)

    ax.add_patch(poly)


import matplotlib.mathtext as mathtext
mathtext.FontConstantsBase.sup1 = 0.5
mathtext.FontConstantsBase.sub1 = 0.2
mathtext.FontConstantsBase.sub2 = 0.3

time=np.arange(0,80,0.1)
#time=np.arange(0,28,0.1)
Types = ['Mode 1', 'Mode 2', 'Mode 3', 'Mode 4']
try:
    dfs=pd.read_csv('IL6-e.csv')
except:
    dfs=[]
    for mode in range(1,5,1):
        Paras=np.loadtxt('Spl_Para%i.txt'%mode)
        Mem=[]
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
            v_decay=v[np.argmax(v):]
            t_rec_series1=np.where(v_decay<1e-2)[0]
            t_rec_series2=np.where(v_decay<np.max(v)/2)[0]
            if len(t_rec_series1)==0:
                t_rec1=50
            else:
                t_rec1=t_rec_series1[0]*0.1
            if len(t_rec_series2)==0:
                t_rec2=50
            else:
                t_rec2=t_rec_series2[0]*0.1
            points.append([e[0],e[70],logv[-1],e[-1],np.max(e),np.max(IL6),
                           np.max(logv),np.mean(e),np.max(e_a)])
        points=np.array(points)
        df=pd.DataFrame({'e0':points[:,0],'e7':points[:,1],'v_final':points[:,2],'e_final':points[:,3],
                         'emax':points[:,4],'IL6':points[:,5],'vmax':points[:,6],'e_aver':points[:,7],
                         'e_a':points[:,8],'Mode':[Types[mode-1] for i in range(points.shape[0])]})
        dfs.append(df)
    dfs=pd.concat(dfs)
    dfs.to_csv('IL6-e.csv')
dfs.loc[(dfs['v_final']<=-1),'v_final']=-1
dfs['vf']=10**dfs['v_final']
# fig,axes=plt.subplots(2,1,figsize=(4,4))
# ax,ax2=axes
fig1,ax=plt.subplots(1,1,figsize=(2.4,2.4))
fig2,ax2=plt.subplots(1,1,figsize=(5,2))
ggcolors=['#2CA02C','#1F77B4','#FF7F0E','#D62728',]

markers= {'Mode 1':'o','Mode 2':'^','Mode 3':'s','Mode 4':'v',}
sns.scatterplot(data=dfs,x='e7',y='IL6',hue='Mode', style='Mode',
                hue_order=['Mode 1', 'Mode 2', 'Mode 3', 'Mode 4', ], markers=markers,
                palette=sns.set_palette(ggcolors),ax=ax,s=50,alpha=0.9)
z1 = np.polyfit(np.log10(dfs.e7),np.log10(dfs.IL6), 1)
p1 = np.poly1d(z1)
x2=np.arange(0.1,10,0.1)
y2=10**z1[1]*x2**z1[0]
r=pearsonr(np.log10(dfs.e7),np.log10(dfs.IL6))
print(p1,'pearson=',r[0])
ax.plot(x2,y2,color='k',zorder=-10)
ax.text(x=3,y=4300,s='p=%1.2f'%r[0])
ax.set_xlabel('$\epsilon$ at day 7',fontsize=12,labelpad=0)
ax.set_ylabel('IL-6$_{max}$(pg/mL)',labelpad=0)
labels= ['Mode 1', '2', '3', '4', ]
marker=['v','o', '^', 's',]
s=[plt.scatter([],[],c=ggcolors[i],label=labels[i],marker=marker[i],s=50) for i in range(0,4,1)]
ax.legend(handles=s,frameon=False,ncol=4,loc='lower right',bbox_to_anchor=(1.08,0.98),
          columnspacing=0.8,handlelength=0.4)
ax.set_xscale('log')
ax.set_yscale('log')
inds=dfs.index[dfs.Mode=='Mode 3']
dfs2=dfs.iloc[inds,:]
ggcolors=['#2CA02C','#1F77B4','#FF7F0E','#D62728',]
ho= ['Mode 1', 'Mode 2', 'Mode 3', 'Mode 4', ]
markers=['v','o','^','s']
sizes=[80,80,60,80]
for i in range(4):
    inds=(dfs['Mode']==ho[i])
    x=dfs.loc[inds,'emax']
    y=dfs.loc[inds,'vf']
    ax2.scatter(x,y,marker=markers[i],facecolor=ggcolors[i],edgecolor='w',lw=0.6,s=sizes[i],alpha=0.9,)
# sns.scatterplot(data=dfs,x='emax',y='vf',hue='Mode', style='Mode', lw=0.1,
#                 hue_order=['Asymptomatic', 'Mode 1', 'Mode 2', 'Mode 3', ], markers=markers,
#                 palette=sns.set_palette(ggcolors),ax=ax2,legend=False,s=80,alpha=0.9)
ax2.set_xlabel('$\epsilon_{max}$',fontsize=13,labelpad=-2)
ax2.set_xscale('log')
ax2.set_yscale('log')
xlim=ax2.get_xlim()

ylim=ax2.get_ylim()
y=np.linspace(ylim[0],ylim[1],100)
x=np.ones(100)*3.6
ax2.plot(x,y,c='k',zorder=10,linewidth=1)
#ax2.plot(x2,y2,c='k',zorder=10,linewidth=1)
ax2.annotate(xy=(3.6,0.1),xytext=(1,1),text=r'$\epsilon=\gamma$',color='k',ha='left',
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3", color='k',lw=1),)
ax2.set_yticks([0.1,1,100,10000])
ax2.set_yticklabels(['$<10^{-1}$','$10^0$','$10^2$','$10^4$'])
ax2.set_ylabel('[nCoV]$_{final}$',labelpad=-12)
ax2.set_ylim(ylim)
fig1.subplots_adjust(bottom=0.18,top=0.85,left=0.23,right=0.9,hspace=0.3)
fig1.savefig('IL6-e-1.png')
fig1.savefig('IL6-e-1.svg')
fig2.subplots_adjust(bottom=0.2,top=0.9,left=0.18,right=0.9,hspace=0.3)
fig2.savefig('v-e-2.svg')
fig2.savefig('v-e-2.png')
fig2,axes2=plt.subplots(2,2,figsize=(6,3.5))
axes2=axes2.flat
bins=np.arange(0,20,1)
Types2 = ['Mode 1', 'Mode 2', 'Mode 3', 'Mode 4', ]
ggcolors=['#2CA02C','#1F77B4','#FF7F0E','#D62728',]
for i in range(4):
    ax=axes2[i]
    inds1 = dfs.index[dfs.Mode == Types2[i]]
    dfs1 = dfs.iloc[inds1, :]
    inds2 = dfs.index[dfs.Mode != Types2[i]]
    dfs2 = dfs.iloc[inds2, :]
    sns.histplot(data=dfs1,bins=bins,x='emax',stat='density',color=ggcolors[i],ax=ax,alpha=1)
    #ho=Types2.remove(Types2[i])
    sns.kdeplot(data=dfs2,x='emax',ax=ax,hue='Mode',
                hue_order=['Mode 1', 'Mode 2', 'Mode 3', 'Mode 4', ],
                palette=ggcolors,legend=False,common_norm=False,alpha=0.2,zorder=-10)
    for line in ax.get_lines():
        line.set_alpha(0.5)
    if i in [2,3]:
        ax.set_xlabel('$\epsilon_{max}$',fontsize=12,labelpad=0)
    else:
        ax.set_xlabel('')
    if i in [1,3]:
        ax.set_ylabel('')
    ax.set_ylim([0,0.4])
    ax.set_xlim([0,20])
import matplotlib.patches as mpatches
patches=[mpatches.Patch(color=ggcolors[i], label=labels[i]) for i in range(4)]
fig2.legend(handles=patches,loc='center',ncol=4,bbox_to_anchor=(0.5,0.94),handlelength=2,
              columnspacing=1,frameon=False)
fig2.savefig('e_distribution.svg')
fig2.savefig('e_distribution.png')
plt.show()
