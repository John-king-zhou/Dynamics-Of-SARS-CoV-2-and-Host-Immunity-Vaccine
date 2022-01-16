'''
Fitting protection rates of Sputinik V, CoronaVac, ChAdOx-1, BNT142b2 against wildtype SARS-CoV-2, delta, alpha variants.

Author: Dianjie (Peter) Li
Built Date: 2021/12/28
Edited: 2022/1/16
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from scipy.optimize import minimize
import seaborn as sns
from scipy.integrate import quad
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit

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

# ---- efficacies of protection against WT SARS-CoV-2 --------------
eff_phase3 = np.array([0.5065, 0.916, 0.621, 0.95])
ci_p3 = np.array([[0.3301, 0.6136], [0.856, 0.952],
                  [0.410, 0.757], [0.903, 0.976]])

# ---- efficacies of protection against variants --------------
eff_phase3_delta = np.array([0.81, 0.67, 0.88])
ci_delta = np.array([[0.68, 0.88], [0.613, 0.718], [0.853, 0.901]])
eff_phase3_alpha = np.array([0.745, 0.937])
ci_alpha = np.array([[0.684, 0.794], [0.916, 0.953]])

df_rAd26 = pd.read_csv(path_data+'/rAd26.csv')
dt['normT'].append(np.array(df_rAd26['IFNg-fold'])-1)
dt['normAb'].append(np.array(df_rAd26["Ig"]))
dt['name'].append('Sputnik V-delta')
dt['limit'].append([4/33, 1e-3])

dt['normT'].append((np.array(df_ChAdOx1['T-35']) /
                    np.nanmean(142.77) - 1))
dt['normAb'].append(np.array(df_ChAdOx1['Ab-35'])/56.1)
dt['name'].append('ChAdOx1-delta')
dt['limit'].append([0, 1e-3])

dt['normT'].append(np.array(df_Pfizer['CD8T-29']) /
                   (10*mean_coronavac))
dt['normAb'].append(np.array(df_Pfizer['Ab-29'])/94)
dt['name'].append('BNT162b2-delta')
dt['limit'].append([0, 1e-3])

dt['normT'].append((np.array(df_ChAdOx1['T-35']) /
                    np.nanmean(142.77) - 1))
dt['normAb'].append(np.array(df_ChAdOx1['Ab-35'])/56.1)
dt['name'].append('ChAdOx1-alpha')
dt['limit'].append([0, 1e-3])

dt['normT'].append(np.array(df_Pfizer['CD8T-29']) /
                   (10*mean_coronavac))
dt['normAb'].append(np.array(df_Pfizer['Ab-29'])/94)
dt['name'].append('BNT162b2-alpha')
dt['limit'].append([0, 1e-3])

limit = dt["limit"]
for i in range(len(dt['name'])):
    if i == 0:
        continue
    T_tmp = dt['normT'][i]
    limit[i][1] = np.min(T_tmp[T_tmp > 0])
# pfizer limit = coronvac limit
limit[-1][1] = limit[0][1]
limit[-3][1] = limit[0][1]
limit[3][1] = limit[0][1]

# --------------------------------------------------------
# ---------------- Fitting Distribution ------------------
# --------------------------------------------------------


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
# ---------- Fitting Efficacy -----------------
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


def fit_eff(xdata, eth):
    mux = xdata[:, 0]
    muy = xdata[:, 1]
    sx = xdata[:, 2]
    sy = xdata[:, 3]
    E = np.array([quad(f_eff, -np.inf, np.inf, args=(muy[j], sy[j], mux[j], sx[j], eth))[0]
                  for j in range(len(mux))])
    return E


def lsquared(y, ci, yp):
    return np.sum(((yp-y)/(ci[:, 1]-ci[:, 0]))**2)


def root_mean_square_error(y, yp):
    return np.sqrt(np.mean((y-yp)**2))


# fit WildType
print('Fitting WildType data')
x_for_fit = np.array(
    [np.array([mu[i][0], mu[i][1], sigma[i][0], sigma[i][1]]) for i in range(2)])
eff_phase3_for_fit = eff_phase3[0:2]
eth_original, _ = curve_fit(fit_eff, xdata=x_for_fit,
                            ydata=eff_phase3_for_fit, p0=0.1)
print('(gamma/kv)_WT = %.3f' % (10**eth_original))

# fit delta
print('Fitting Delta data')
x_for_fit = np.array(
    [np.array([mu[i][0], mu[i][1], sigma[i][0], sigma[i][1]]) for i in [4, 5, 6]])
eff_phase3_for_fit = eff_phase3_delta
eth_delta, _ = curve_fit(fit_eff, xdata=x_for_fit,
                         ydata=eff_phase3_for_fit, p0=0.1)
print('(gamma/kv)_Delta = %.3f' % (10**eth_delta))

# fit alpha
print('Fitting Alpha data')
x_for_fit = np.array(
    [np.array([mu[i][0], mu[i][1], sigma[i][0], sigma[i][1]]) for i in [7, 8]])
eff_phase3_for_fit = eff_phase3_alpha
eth_alpha, _ = curve_fit(fit_eff, xdata=x_for_fit,
                         ydata=eff_phase3_for_fit, p0=0.1)
print('(gamma/kv)_Alpha = %.3f' % (10**eth_alpha))

# =================================================================================
# -------- Fitting results on prediciton-observation space -------------------
# =================================================================================
eff_tot = np.hstack((eff_phase3, eff_phase3_delta, eff_phase3_alpha))
ci_tot = np.vstack((ci_p3, ci_delta, ci_alpha))
eth_tot = [eth_original]*len(eff_phase3)+[eth_delta] * \
    len(eff_phase3_delta)+[eth_alpha]*len(eff_phase3_alpha)
eff_pred = []
fig_pred, ax_pred = plt.subplots(figsize=(5, 3))

mkrstl = [variant_shape[i] for i in [0] *
          len(eff_phase3)+[1]*len(eff_phase3_delta)+[2]*len(eff_phase3_alpha)]
clr_vax = set2colors[:4]+[set2colors[0]]+set2colors[2:4]*2
for eth, mu_t, sig_t, i in zip(eth_tot, mu, sigma, range(len(mu))):
    eff_pred.append(efficacy(mu_t, sig_t, eth))
    if i==2:
        continue
    ax_pred.plot(eff_tot[i]*100, eff_pred[-1]*100, label=dt['name'][i], ls='',
                 marker=mkrstl[i], color=clr_vax[i],
                 markeredgecolor='gray', markersize=8)
    ax_pred.plot(ci_tot[i]*100, [eff_pred[-1]*100]*2,
                 marker='|', color='gray', zorder=-1,
                 markersize=10)

eff_pred = np.array(eff_pred)
ax_pred.plot(np.linspace(0, 110), np.linspace(
    0, 110), color='gray', ls='--', zorder=-2)
ax_pred.set_ylabel('Prediction (%)')
ax_pred.set_xlabel('Reported Efficacy (%)')
ax_pred.legend(loc='lower left', bbox_to_anchor=([1, 0]))
ax_pred.set_xlim([45, 105])
ax_pred.set_xticks([50, 60, 70, 80, 90, 100])
ax_pred.set_ylim([45, 105])
ax_pred.set_yticks([50, 60, 70, 80, 90, 100])
fig_pred.tight_layout()

# ------- compute RMSE ------------------
tot_pears = np.corrcoef(np.array([eff_tot, eff_pred]))[0, 1]
orig_pears = np.corrcoef(np.array([eff_phase3, eff_pred[0:4]]))[0, 1]
delta_pears = np.corrcoef(np.array([eff_phase3_delta, eff_pred[4:7]]))[0, 1]
alpha_pears = np.corrcoef(np.array([eff_phase3_alpha, eff_pred[7:]]))[0, 1]

print('Root mean square errors:'+'\n'+'Total:%.3f%%, ' % (root_mean_square_error(eff_tot, eff_pred)*100)
      + 'WT nCoV:%.3f%%, ' % (root_mean_square_error(eff_phase3, eff_pred[0:4])*100)
      + 'WT nCoV (CoronaVac, Sputnik, BNT162):%.3f%%, ' % (root_mean_square_error(
          eff_phase3[[0, 1, 3]], eff_pred[[0, 1, 3]])*100)
      + 'Delta:%.3f%%, ' % (root_mean_square_error(eff_phase3_delta, eff_pred[4:7])*100)
      + 'Alpha:%.3f%% ' % (root_mean_square_error(eff_phase3_alpha, eff_pred[7:])*100)
      + '\n')

# ============================================================
# -------- Figures of fitted protection borders -------------------
# =============================================================
splsize = 10000
eff_tmp = eff_pred
# sampling data by fitted distribution
labels = ['CoronaVac', 'Sputnik V', 'ChAdOx1', 'BNT162b2', ]
for group in [[0,1],[2,3]]:
    dt_joint = {'x': [], 'y': [], 'Vaccine (Efficacy)': []}
    for i in group:
        xtmp, ytmp = two_norm_sample(mu[i], sigma[i], splsize)
        dt_joint['x'] = dt_joint['x']+xtmp.tolist()
        dt_joint['y'] = dt_joint['y']+ytmp.tolist()
        dt_joint['Vaccine (Efficacy)'] = \
            dt_joint['Vaccine (Efficacy)'] \
            + [dt['name'][i]+' (%.2f%%)' % (eff_tmp[i]*100)]*splsize
        if i == 0:
            eff1 = eff_tmp[i] * 100
        if i == 1:
            eff2 = eff_tmp[i] * 100

    # protection line
    g = sns.jointplot(data=dt_joint, x='x', y='y',
                    hue='Vaccine (Efficacy)', kind="kde", height=3.6, space=0,
                    alpha=0.5,
                    levels=6, fill=True, palette=sns.set_palette(set2colors[group[0]:]))
    ax = g.ax_joint
    g.plot_joint(sns.kdeplot,
                hue='Vaccine (Efficacy)', kind="kde", height=3.6, space=0,
                alpha=0.5, lw=1,
                levels=6, fill=False, palette=sns.set_palette(set2colors[group[0]:]),
                ax=ax)

    # protection lines
    patches_line = []
    ls_variant = ['-', '-', '-']
    for eth_tmp, name, i in zip([eth_original, eth_delta, eth_alpha, ], ['WT', 'Delta', 'Alpha'], range(3)):
        xtmp = np.linspace(-3-0.4*i, 3-0.4*i, 10)
        ytmp = eth_tmp-xtmp
        ax.plot(xtmp, ytmp, ls=ls_variant[i], lw=2, alpha=1, zorder=100,
                color='gray', marker=variant_shape[i], markevery=1, ms=6,
                fillstyle='full', mfc='w',
                label=name+':'+'%.2f' % 10**eth_tmp)
        patches_line.append(plt.plot([], [], ls=ls_variant[i], lw=2.5, alpha=1, zorder=100,
                                    color='gray', marker=variant_shape[i], fillstyle='full',
                                    mfc='w', label=name)[0])

    patches = [mpatches.Patch(
        color=set2colors[i], label=labels[i], alpha=0.5) for i in range(4)]
    ax.legend(handles=patches+patches_line, loc='upper right', bbox_to_anchor=(1, 1),
            columnspacing=1, frameon=False, fontsize=11, ncol=1)
    ax.set_xlim([-2.5, 1.5])
    ax.set_ylim([-2, 4])
    ax.set_xlabel('$log_{10}$Ab (Normalized)')
    ax.set_ylabel('$log_{10}$T (Normalized)')

    if group[0]==0:
        figname = 'FigSI_vaccine_distribution_%i' %(1)
    else:
        figname = 'FigSI_vaccine_distribution_%i' %(2)

    plt.savefig(figname+'.png', dpi=300)
    plt.savefig(figname+'.svg', dpi=300)
fig_pred.savefig('FigSI_efficacy_predict.png', dpi=300)
fig_pred.savefig('FigSI_efficacy_predict.svg', dpi=300)
plt.show()
