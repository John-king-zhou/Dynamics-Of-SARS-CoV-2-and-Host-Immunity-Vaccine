'''
Fitting the immunogenicity distribution and the wildtype SARS-CoV-2 protection efficacy of Sputinik V and CoronaVac.

Author: Dianjie (Peter) Li
Built Date: 2021/12/11
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
from scipy.optimize import curve_fit

path = os.path.split(os.path.realpath(__file__))[0]
path_data = path
set2colors = ['#66c2a5', '#fc8d62', '#a6d854',
              '#e78ac3', '#ffd92f', '#e5c494', '#b3b3b3']
# -------------- Load Data -----------------------------
# norm T -> normalized specific T level = (IFNg - IFNg0_mean)/IFNg0_mean
# norm Ab -> Neutralising Antibody/(Antibody of convalescent)
dt = {'normT': [], 'normAb': [], 'name': [], 'logAb': [], 'logT': []}
# -----CoronaVac, 0/14-----
# Here, the individual data from Sinovac Biotech are not provided. 
# We give out the fitting results of CoronaVac in following codes.
dt['normT'].append([])
dt['normAb'].append([])
dt['name'].append('CoronaVac')
# -----Sputnik V-----
df_rAd26 = pd.read_csv(path_data+'/rAd26.csv')
dt['normT'].append(np.array(df_rAd26['IFNg-fold'])-1)
dt['normAb'].append(np.array(df_rAd26["Ig"]))
dt['name'].append('Sputnik V')

# Limitations of detection for antibody titers and IFN-gamma levels
limit = [[0.02443494196701283, 0.27810658450334813], [4/33, 1e-3]]
for i in range(len(dt['name'])):
    if i == 0:
        continue
    T_tmp = dt['normT'][i]
    limit[i][1] = np.min(T_tmp[T_tmp > 0])
# Vaccine efficacy of CoronaVac, Sputnik V from clinical trials
eff_phase3 = np.array([0.5065, 0.916])

# ---------------- Fitting Distribution of Ab and T  ------------------


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

for i in range(0, len(dt['name'])):
    if i==0:
        dt['logAb'].append([])
        dt['logT'].append([])
        continue
    x = dt['normAb'][i]
    x = x[~np.isnan(x)]
    xlimit = limit[i][0]
    x[x <= 0] = xlimit
    logx = np.log10(x)
    resx = minimize(negative_log_likelihood_func, [[np.mean(logx), 1]], method='Nelder-Mead',
                    args=(logx, np.log10(xlimit)),
                    options={'disp': True})

    y = dt['normT'][i]
    y = y[~np.isnan(y)]
    ylimit = limit[i][1]
    y[y <= 0] = ylimit
    logy = np.log10(y)
    resy = minimize(negative_log_likelihood_func, [[0, 0.5]], method='Nelder-Mead',
                    args=(logy, np.log10(ylimit)),
                    options={'disp': True})

    mu.append([resx.x[0], resy.x[0]])
    sigma.append([resx.x[1], resy.x[1]])
    dt['logAb'].append(logx)
    dt['logT'].append(logy)


# --------- Fitting Vaccine Efficacy -----------------------------------------------------

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


def root_mean_square_error(y, yp):
    return np.sqrt(np.mean((y-yp)**2))


# fitting data
x_for_fit = np.array(
    [np.array([mu[i][0], mu[i][1], sigma[i][0], sigma[i][1]]) for i in range(2)])
eff_phase3_for_fit = eff_phase3[0:2]

eth_fit, _ = curve_fit(fit_eff, xdata=x_for_fit,
                       ydata=eff_phase3_for_fit, p0=0.1)

# compute root mean square errors
predict_eff = []
for i in range(2):
    predict_eff.append(efficacy(mu[i], sigma[i], eth_fit))
rmse = root_mean_square_error(eff_phase3, np.array(predict_eff))
print('Fitted threshold(gamma/kv)=%.3f, with RMSE=%.2f%%' % (10**eth_fit, rmse*100))


# -------- Figures of vaccine efficacy fitting -------------------------------------------
def two_norm_sample(mu, sigma, size):
    cov = [[sigma[0]**2, 0], [0, sigma[1]**2]]
    x, y = np.random.multivariate_normal(mu, cov, size).T
    return (x, y)


dt_joint = {'x': [], 'y': [], 'Vaccine (Efficacy)': []}
splsize = 10000
eth_tmp = eth_fit
eff_tmp = [efficacy(mu[i], sigma[i], eth_tmp) for i in range(len(mu))]

# sampling data of fitted distribution
for i in range(len(dt['name'])):
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
                  levels=6, fill=True, palette=sns.set_palette(set2colors))
ax = g.ax_joint
xtmp = np.linspace(-5, 5, 100)
ytmp = eth_tmp-xtmp
ax.plot(xtmp, ytmp, 'k--', lw=2.5, alpha=0.8, zorder=100, label=eff_tmp)
ax.text(x=0.5, y=eth_tmp-0.5+0.2,
        s='Protection'+'\n'+'Border',
        fontsize=14,
        rotation=-25, rotation_mode="anchor")

# Plot original individual-level data
xtmp = np.copy(dt['logAb'][1])
ytmp = np.copy(dt['logT'][1])
np.random.shuffle(xtmp)
for xx, yy in zip(xtmp, ytmp):
    if xx+yy >= eth_tmp:
        ax.scatter(xx, yy, s=40, marker='o',
                    facecolor=set2colors[1], edgecolor='w')
    else:
        ax.scatter(xx, yy, s=80, marker='X', facecolor='w')
        ax.scatter(xx, yy, s=40, marker='x', facecolor=set2colors[1])
ax.set_xlim([-2.5, 1.8])
ax.set_xticks([-2, -1, 0, 1])
ax.set_ylim([-3.2, 5.8])
ax.set_xlabel('$log_{10}$Ab (Normalized)')
ax.set_ylabel('$log_{10}$T (Normalized)')
fig = g.fig
fig.set_size_inches(4, 4)

# to save figures
# fig.savefig('efficacy.png')
# fig.savefig('efficacy.svg')

plt.show()
