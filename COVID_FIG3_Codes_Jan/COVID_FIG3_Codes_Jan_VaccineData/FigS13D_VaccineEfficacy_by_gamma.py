'''
Predicting how gamma affects vaccine protection rates against diffent variants.

Author: Dianjie (Peter) Li
Built Date: 2021/12/28
Edited: 2022/1/16 Dianjie Li, John King Zhou
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.integrate import quad

path = os.path.split(os.path.realpath(__file__))[0]
path_data = path
set2colors = ['#66c2a5', '#fc8d62', '#a6d854',
              '#e78ac3', '#ffd92f', '#e5c494', '#b3b3b3', 'gray', 'red']
variant_shape = ['o', 'D', '^', ]
# -------------- Load Data -----------------------------
# norm T -> normalized specific T level = (IFNg - IFNg0_mean)/IFNg0_mean
# norm Ab -> Neutralising Antibody/(Antibody of convalescent)
dt = {'normT': [], 'normAb': [], 'name': [],
      'logAb': [], 'logT': [], 'limit': []}
# -----CoronaVac, 0/14-----
# Here, the individual data from Sinovac Biotech are not provided.
# We give out the fitting results of CoronaVac in following codes.
dt['normT'].append([])
dt['normAb'].append([])
dt['name'].append('CoronaVac')
dt['limit'].append([0.02443494196701283, 0.27810658450334813])
# -----rAd26-----
df_rAd26 = pd.read_csv(path_data+'/rAd26.csv')
dt['normT'].append(np.array(df_rAd26['IFNg-fold'])-1)
dt['normAb'].append(np.array(df_rAd26["Ig"]))
dt['name'].append('Sputnik V')
dt['limit'].append([4/33, 1e-3])
# -----ChAdOx1, day 35-----
df_ChAdOx1 = pd.read_csv(path_data+'\ChAdOx1-python.csv')
dt['normT'].append(np.array(df_ChAdOx1['T-35']) /
                   np.nanmean(142.77) - 1)
dt['normAb'].append(np.array(df_ChAdOx1['Ab-35'])/56.1)
dt['name'].append('ChAdOx1')
dt['limit'].append([0, 1e-3])
# -----Pfizer, day 29-----
df_Pfizer = pd.read_csv(path_data+'\Pfizer-python.csv')
mean_coronavac = 0.7824073611111112
dt['normT'].append(np.array(df_Pfizer['CD8T-29']) /
                   (10*mean_coronavac))
dt['normAb'].append(np.array(df_Pfizer['Ab-29'])/94)
dt['name'].append('BNT162b2')
dt['limit'].append([0, 1e-3])

limit = dt["limit"]
for i in range(len(dt['name'])):
    if i == 0:
        continue
    T_tmp = dt['normT'][i]
    limit[i][1] = np.min(T_tmp[T_tmp > 0])
# pfizer limit = coronvac limit
limit[-1][1] = limit[0][1]


# ---------------- Fitting Distribution ------------------


def negative_log_likelihood_func(theta, x, limit):
    mu, sigma = theta
    lkhfunc = 1
    for val in x:
        if val < limit:
            lkhfunc = lkhfunc*norm.cdf(val, loc=mu, scale=sigma)
        else:
            lkhfunc = lkhfunc*norm.pdf(val, loc=mu, scale=sigma)
    return -np.log10(lkhfunc)


mu = [[-1.4181462985925921, 0.6135169024462654]]
sigma = [[0.31736946190249415, 0.6250472517036898]]

for i in range(len(dt['name'])):
    if i == 0:
        dt['logAb'].append([])
        dt['logT'].append([])
        continue
    x = dt['normAb'][i]
    x = x[~np.isnan(x)]
    xlimit = np.copy(limit[i][0])
    x[x <= 0] = np.copy(xlimit)
    logx = np.log10(x)
    resx = minimize(negative_log_likelihood_func, [[np.mean(logx), 1]], method='Nelder-Mead',
                    args=(logx, np.log10(xlimit)),
                    options={'disp': True})

    y = dt['normT'][i]
    y = y[~np.isnan(y)]
    ylimit = np.copy(limit[i][1])
    y[y <= 0] = np.copy(ylimit)
    logy = np.log10(y)
    resy = minimize(negative_log_likelihood_func, [[0, 0.5]], method='Nelder-Mead',
                    args=(logy, np.log10(ylimit)),
                    options={'disp': True})

    mu.append([resx.x[0], resy.x[0]])
    sigma.append([resx.x[1], resy.x[1]])
    dt['logAb'].append(logx)
    dt['logT'].append(logy)

# =============================================================
# ---------- Efficacy depends on the gamma/kv -----------------
# =============================================================


def two_norm_sample(mu, sigma, size):
    cov = [[sigma[0]**2, 0], [0, sigma[1]**2]]
    x, y = np.random.multivariate_normal(mu, cov, size).T
    return (x, y)


def f_eff(y, muy, sy, mux, sx, eth):
    return norm.pdf(y, loc=muy, scale=sy)*(1-norm.cdf(eth-y, loc=mux, scale=sx))


def efficacy(mu, sigma, eth):
    mux, muy = mu
    sx, sy = sigma
    return quad(f_eff, -np.inf, np.inf, args=(muy, sy, mux, sx, eth))[0]

figeff,axeff=plt.subplots(1,1,figsize=(3,3))
#variant_shape = [r'$W$', r'$\alpha$', r'$\delta$', r'$o$']
variant_shape = ['o', '^', 'D', 'X']
clr_vax = set2colors[:4]
# eth of variants
eth_voc = 10**np.array([-0.814,-0.296,-0.140])
Omicron_eth = 10**(-0.814)*np.array([20,20,21,37])
voc_name=['WT','Alpha','Delta','Omicron']
labels = ['Sputnik V', 'CoronaVac', 'ChAdOx1', 'BNT162b2', ]
for i in range(4):#different voc
    eff_voc = []
    for j in range(4):#different vaccines
        if i==3:
            eff_voc.append(efficacy(mu[j], sigma[j], np.log10(Omicron_eth[j])))
            if j==0:
                axeff.plot(Omicron_eth[j], eff_voc[-1] * 100, label=voc_name[i],
                           marker=variant_shape[i], markersize=8, c='w',
                           markeredgecolor='gray', ls='', markeredgewidth=1.3,
                           zorder=100)
            else:
                axeff.plot(Omicron_eth[j], eff_voc[-1] * 100,
                           marker=variant_shape[i], markersize=8, c='w',
                           markeredgecolor='gray', ls='', markeredgewidth=1.3,
                           zorder=100)
            print(labels[j],eff_voc[-1])
        else:
            eff_voc.append(efficacy(mu[j], sigma[j], np.log10(eth_voc[i])))
    if i!=3:
        axeff.plot([eth_voc[i]]*4,np.array(eff_voc)*100,label=voc_name[i],
                    marker=variant_shape[i],markersize=8,c='w',
                    markeredgecolor='gray',ls='',markeredgewidth=1.3,
                    zorder=100)

# eth sequences
eth = 10**np.linspace(-1,1,20)
eff = []
for i in range(len(eth)):
    eff_tmp=[]
    for j in range(4):
        eff_tmp.append(efficacy(mu[j], sigma[j], np.log10(eth[i])))
    eff.append(eff_tmp)
    print(i)
eff = np.array(eff)

for j in range(4):
    axeff.plot(eth, eff[:,j]*100,lw=2,c=clr_vax[j])
axeff.legend(loc='upper center',bbox_to_anchor=(0.48,1.27),fontsize=11,frameon=False,ncol=2)
axeff.set_xticks([0.1,1,10])
axeff.set_xlim([0.1,10])
axeff.set_yticks([0,20,40,60,80,100])
axeff.set_xscale('log')
axeff.set_xlabel(r'$\gamma/k_v$',labelpad=0)
axeff.set_ylabel('Prediction (%)',labelpad=0)
axeff.grid()
figeff.subplots_adjust(bottom=0.18,left=0.18,top=0.85,right=0.85)

figeff.savefig('FigSI_Variants_efficacy_by_gamma.png')
figeff.savefig('FigSI_Variants_efficacy_by_gamma.svg')

plt.show()
