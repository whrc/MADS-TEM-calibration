#!/usr/bin/env python
# coding: utf-8

import utils as ut
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sns
import scipy
get_ipython().run_line_magic('matplotlib', 'inline')


#Load paths
# path_VM = ../../dvm-dos-tem/mads_calibration/'
samples='US-Prr/SA/STEP1-cmax-GPP/sample_matrix.csv'
results='US-Prr/SA/STEP1-cmax-GPP/results.csv'


# Cmax targeting GPP

df_param = pd.read_csv(samples)
df_model = pd.read_csv(results,header=None)

output_name = ['GPP0','GPP1','GPP2','GPP3', 'GPP4']
target_vars_nopft = ['GPP']
df_model.columns = output_name
units = 'gC/m2/year'
title_1='One to One Match Plot fot Cmax targeting GPP'
title_2='Spaghetti Match Plot fot Cmax targeting GPP'
title_3='Log Scale Spaghetti Match Plot fot Cmax targeting GPP'


df_metrics = ut.get_unified_r2_rmse(df_model, target_vars_nopft, plot=True)


df_model.head()


ut.one_to_one_match_plot(df_param)
plt.title(title_1)
ut.spaghetti_match_plot(df_param, df_model)
plt.ylabel(units)
plt.title(title_2)
ut.spaghetti_match_plot(df_param, df_model, logy=True)
plt.ylabel(units)
plt.title(title_3)


rmse=ut.plot_r2_rmse(df_model)


corr_mp=ut.get_output_param_corr(df_param,df_model)


xparams, ymodel =  ut.get_params_r2_rmse(df_param,df_model,r2lim=0.97)
xparams


best=xparams[xparams.MAPE<1]
best_model=ymodel[xparams.MAPE<1]
best


#print bounds of param from best scores
x_max = best.max()
x_min = best.min()

# Create a list of pairs [x_min, x_max] where x_min and x_max correspond to the same index in xparams
result = [[x_min[i], x_max[i]] for i in range(len(x_max))]

for pair in result:
    print("Uniform({:.5f}, {:.5f})".format(pair[0], pair[1]))


# Nmax + Krb Targeting NPP and VEGC

df_param = pd.read_csv('US-Prr/SA/STEP2-nmax_krb-NPP_vegc/sample_matrix.csv')
df_model = pd.read_csv('US-Prr/SA/STEP2-nmax_krb-NPP_vegc/results.csv',header=None)

output_name =['NPPAll1', 'NPPAll2', 'NPPAll3', 'NPPAll4', 'NPPAll5',
                   'VegCarbonLeaf1', 'VegCarbonStem1', 'VegCarbonRoot1', 'VegCarbonLeaf2', 'VegCarbonLeaf3',
                   'VegCarbonStem3', 'VegCarbonRoot3', 'VegCarbonLeaf4',
                   'VegCarbonRoot4', 'VegCarbonLeaf5']
target_vars_nopft =['NPPAll', 'VegCarbonLeaf', 'VegCarbonStem', 'VegCarbonRoot']
df_model.columns = output_name
units_NPP = 'gC/m2/year'
units_VEGC = 'gC/m2'
title_1='One to One Match Plot for Nmax targeting NPP'
title_2='Spaghetti Match Plot for Nmax targeting NPP'
title_3='Log Scale Spaghetti Match Plot for Nmax targeting NPP'

title_4='One to One Match Plot for krb targeting VEGC'
title_5='Spaghetti Match Plot for krb targeting VEGC'
title_6='Log Scale Spaghetti Match Plot for krb targeting VEGC'


df_metrics = ut.get_unified_r2_rmse(df_model, target_vars_nopft, plot=True)


ut.one_to_one_match_plot(df_param.iloc[:,0:5])
plt.title(title_1)
ut.spaghetti_match_plot(df_param.iloc[:,0:5], df_model.iloc[:,0:5])
plt.ylabel(units_NPP)
plt.title(title_2)
ut.spaghetti_match_plot(df_param.iloc[:,0:5], df_model.iloc[:,0:5], logy=True)
plt.ylabel(units_NPP)
plt.title(title_3)


ut.one_to_one_match_plot(df_param.iloc[:,5:])
plt.title(title_4)
ut.spaghetti_match_plot(df_param.iloc[:,5:], df_model.iloc[:,5:])
plt.ylabel(units_VEGC)
plt.title(title_5)
ut.spaghetti_match_plot(df_param.iloc[:,5:], df_model.iloc[:,5:], logy=True)
plt.ylabel(units_VEGC)
plt.title(title_6)


xparams, ymodel =  ut.get_params_r2_rmse(df_param,df_model,r2lim=0.97)
xparams


rmse=ut.plot_r2_rmse(df_model)


corr_mp=ut.get_output_param_corr(df_param,df_model)


e_rmse=ut.find_important_features_err(df_param,df_model,error='rmse')


ut.plot_relationships(corr_mp,df_param,df_model,corr_thresh=0.50)


# Nmax + krb + cfall + nfall targeting NPP, VEGC, and VEGN

df_param = pd.read_csv('US-Prr/SA/STEP3-nmax_krb_cfall_nfall-NPP_vegc_vegn/sample_matrix.csv')
df_model = pd.read_csv('US-Prr/SA/STEP3-nmax_krb_cfall_nfall-NPP_vegc_vegn/results.csv')

output_name = ['NPPAll1', 'NPPAll2', 'NPPAll3', 'NPPAll4', 'NPPAll5',
                'VegCarbonLeaf1', 'VegCarbonStem1', 'VegCarbonRoot1', 'VegCarbonLeaf2', 'VegCarbonLeaf3',
                   'VegCarbonStem3', 'VegCarbonRoot3', 'VegCarbonLeaf4',
                   'VegCarbonRoot4', 'VegCarbonLeaf5',
                   'VegNitrogenLeaf1', 'VegNitrogenStem1', 'VegNitrogenRoot1', 'VegNitrogenLeaf2', 'VegNitrogenLeaf3',
                   'VegNitrogenStem3', 'VegNitrogenRoot3', 'VegNitrogenLeaf4',
                   'VegNitrogenRoot4', 'VegNitrogenLeaf5']

target_vars_nopft = ['NPPAll', 'VegCarbonLeaf', 'VegCarbonStem', 'VegCarbonRoot', 'VegNitrogenLeaf', 'VegNitrogenStem', 'VegNitrogenRoot']

df_model.columns = output_name

units_NPP = 'gC/m2/year'
units_VEGC = 'gC/m2' #same for VEGN
title_1='One to One Match Plot for Nmax/krb/cfall/nfall targeting NPP'
title_2='Spaghetti Match Plot for Nmax/krb/cfall/nfall targeting NPP'
title_3='Log Scale Spaghetti Match Plot for Nmax/krb/cfall/nfall targeting NPP'

title_4='One to One Match Plot for krb/cfall/nfall targeting VEGC'
title_5='Spaghetti Match Plot for krb/cfall/nfall targeting VEGC'
title_6='Log Scale Spaghetti Match Plot for krb/cfall/nfall targeting VEGC'

title_7='One to One Match Plot for krb/cfall/nfall targeting VEGN'
title_8='Spaghetti Match Plot for krb/cfall/nfall targeting VEGN'
title_9='Log Scale Spaghetti Match Plot for krb/cfall/nfall targeting VEGN'


df_metrics = ut.get_unified_r2_rmse(df_model, target_vars_nopft, plot=True)


df_metrics


ut.one_to_one_match_plot(df_param.iloc[:,0:5])
plt.title(title_1)
ut.spaghetti_match_plot(df_param.iloc[:,0:5], df_model.iloc[:,0:5])
plt.ylabel(units_NPP)
plt.title(title_2)
ut.spaghetti_match_plot(df_param.iloc[:,0:5], df_model.iloc[:,0:5], logy=True)
plt.ylabel(units_NPP)
plt.title(title_3)


ut.one_to_one_match_plot(df_param.iloc[:,5:15])
plt.title(title_4)
ut.spaghetti_match_plot(df_param.iloc[:,5:15], df_model.iloc[:,5:15])
plt.ylabel(units_VEGC)
plt.title(title_5)
ut.spaghetti_match_plot(df_param.iloc[:,5:15], df_model.iloc[:,5:15], logy=True)
plt.ylabel(units_VEGC)
plt.title(title_6)


ut.one_to_one_match_plot(df_param.iloc[:,15:])
plt.title(title_7)
ut.spaghetti_match_plot(df_param.iloc[:,15:], df_model.iloc[:,15:25])
plt.ylabel(units_VEGC)
plt.title(title_8)
ut.spaghetti_match_plot(df_param.iloc[:,15:25], df_model.iloc[:,15:25], logy=True)
plt.ylabel(units_VEGC)
plt.title(title_9)




