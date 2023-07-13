#!/usr/bin/env python
# coding: utf-8

import utils as ut
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


#Load paths
# path_VM = ../../dvm-dos-tem/mads_calibration/'
samples='sample_matrix.csv'
results='results.txt'
results_new='results.csv'
path_EML='EML21/'
path_MD3='MD3/'
path_MD1='MD1/'
path_TK='TK/'
path_USPrr='US-Prr/SA/'


#Load Functions

def z_score(y_short,y_long,outnames):
    '''
    NOTE: this function assumes that last row in y_long is target
    
    y_short: model dataframe restricted by R2
    y_long : full model dataframe 
    outnames : dataframe header
    '''

    zscore=[]
    for iname in outnames:
        zscore.append((y_short[iname].mean()-y_long[iname].iloc[-1])/y_short[iname].std())
    
    return zscore


from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
# MAE is computed by taking the absolute difference between each target value and its corresponding 
# model output value (over all runs), and then taking the average of these absolute differences. n=num simulations
# MAE = (1 / n) * Σ|target_i - model_i|
# Calculate the mean of the absolute values of the target values.
# Mean Absolute Target Value = (1 / n) * Σ|target_i|
# NMAE = (MAE / Mean Absolute Target Value) * 100

def calculate_nmae(df_model,ymodel):
    '''    
    df_model: model dataframe
    ymodel: full dataframe constrained by error
    '''
    targets = df_model.iloc[-1, :]
    [n,m]=np.shape(ymodel)
    df=(abs(ymodel.iloc[:-1,:] - df_model.iloc[-1, :]))
    column_sums = df.sum(axis=0)
    mae = column_sums /(n-1) 
    mean_abs_target = np.abs(targets).mean()
    nmae = (mae / mean_abs_target) * 100
    return nmae


def site_metric_matrix(metric_matrix, metric):
    '''    
    metric_matrix: matrix (dataframe) of metric to plot. cols = sites, rows = targets
    metric: string of the name of the metric, ex; 'NMAE'
    '''
    #if metric=='Zscore':
        
    n_rows, n_cols = metric_matrix.shape
    fig, ax = plt.subplots(figsize=(n_cols, n_rows))
    sns.heatmap(metric_matrix, cmap="coolwarm", annot=True, fmt=".2f")
    ax.set_xlabel('Sites')
    ax.set_ylabel('Targets')
    ax.set_title(metric + ' Matrix')
    plt.show()
    return


# # cmax -> GPP

# Load EML21
sa_folder='sa_cmax_EML21_090_AK/'
df_param = pd.read_csv(path_EML+sa_folder+samples)
df_model = pd.read_csv(path_EML+sa_folder+results,header=None)

output_name = ['GPP0','GPP1','GPP2','GPP3','GPP4','GPP5','GPP6','NA','NA']
df_model.columns = output_name
units = 'gC/m2/year'

xparams, ymodel =  ut.get_params_r2_rmse(df_param,df_model,n_top_runs=50)


# Load MD3
sa_folder = 'sa_cmax_MD3_050_AK/'
df_param_MD3 = pd.read_csv(path_MD3+sa_folder+samples)
df_model_MD3 = pd.read_csv(path_MD3+sa_folder+results,header=None)

output_name_MD3 = ['GPP0','GPP1','GPP2','GPP3']
df_model_MD3.columns = output_name_MD3

xparams_MD3, ymodel_MD3 =  ut.get_params_r2_rmse(df_param_MD3,df_model_MD3,n_top_runs=50)


# Load MD1
sa_folder = 'SA-MD1-STEP1/'
print(path_MD1+sa_folder+samples)
df_param_MD1 = pd.read_csv(path_MD1+sa_folder+samples)
df_model_MD1 = pd.read_csv(path_MD1+sa_folder+results,header=None)

output_name_MD1 = ['GPP0','GPP1','GPP2','GPP3']
df_model_MD1.columns = output_name_MD1
xparams_MD1, ymodel_MD1 =  ut.get_params_r2_rmse(df_param_MD1,df_model_MD1,n_top_runs=50)


# Load TK
sa_folder = 'sa-cmax-TK-075-EJ/'
print(path_TK+sa_folder+samples)
df_param_TK = pd.read_csv(path_TK+sa_folder+samples)
df_model_TK = pd.read_csv(path_TK+sa_folder+'results.csv',header=None)

output_name_TK = ['GPP0','GPP1','GPP2','GPP3','GPP4','GPP5','GPP6','GPP7']
df_model_TK.columns = output_name_TK
xparams_TK, ymodel_TK =  ut.get_params_r2_rmse(df_param_TK,df_model_TK,n_top_runs=50)


# Load US-Prr
sa_folder = 'STEP1-cmax-GPP/'
print(path_USPrr+sa_folder+samples)
df_param_USPrr = pd.read_csv(path_USPrr+sa_folder+samples)
df_model_USPrr = pd.read_csv(path_USPrr+sa_folder+'results.csv',header=None)

output_name_USPrr = ['GPP0','GPP1','GPP2','GPP3','GPP4']
df_model_USPrr.columns = output_name_USPrr
xparams_USPrr, ymodel_USPrr =  ut.get_params_r2_rmse(df_param_USPrr,df_model_USPrr,n_top_runs=50)


#PLOT NMAE
nmae = calculate_nmae(df_model,ymodel)
nmae_tk = calculate_nmae(df_model_TK,ymodel_TK)
nmae_md3 = calculate_nmae(df_model_MD3,ymodel_MD3)
nmae_md1 = calculate_nmae(df_model_MD1,ymodel_MD1)
nmae_usprr = calculate_nmae(df_model_USPrr,ymodel_USPrr)
nmae_matrix = pd.DataFrame({'EML21': nmae[:7], 'TK': nmae_tk, 'MD3': nmae_md3, 'MD1': nmae_md1, 'US-Prr': nmae_usprr})
site_metric_matrix(nmae_matrix, 'NMAE')

#PLOT ZSCORE:
z = z_score(ymodel,df_model,output_name)
z = pd.Series(z[:7], dtype=float)
z_tk = z_score(ymodel_TK,df_model_TK,output_name_TK)
z_tk = pd.Series(z_tk, dtype=float)
z_md3 = z_score(df_model_MD3,ymodel_MD3,output_name_MD3)
z_md3 = pd.Series(z_md3, dtype=float)
z_md1 = z_score(df_model_MD1,ymodel_MD1,output_name_MD1)
z_md1 = pd.Series(z_md1, dtype=float)
z_usprr = z_score(df_model_USPrr,ymodel_USPrr, output_name_USPrr)
z_usprr = pd.Series(z_usprr, dtype=float)
z_matrix = pd.DataFrame({'EML21': z, 'TK': z_tk, 'MD3': z_md3, 'MD1': z_md1, 'US-Prr': z_usprr})
site_metric_matrix(z_matrix.iloc[:8,:], 'Z-score')


# # nmax+krb -> NPP+VEGC

#Load EML21
per_number=0.5
sa_folder = 'sa_nmaxkrb_EML21_090_AK-2/'
df_param = pd.read_csv(path_EML+sa_folder+samples)
df_model = pd.read_csv(path_EML+sa_folder+results,header=None)

#output_name = ['NPP0','NPP1','NPP2','NPP3','NPP4','NPP5','NPP6',\
#               'VEGC00','VEGC01','VEGC02','VEGC03','VEGC04','VEGC05','VEGC06',\
#                'VEGC10','VEGC11',\
#                'VEGC20','VEGC21','VEGC22','VEGC23']

output_name = ['NPP0','NPP1','NPP2','NPP3','NPP4','NPP5','NPP6',\
               'VEGC00','VEGC10','VEGC20',\
               'VEGC01','VEGC11','VEGC21',\
               'VEGC02','VEGC22',\
               'VEGC03','VEGC23',\
               'VEGC04','VEGC05','VEGC06']
df_model.columns = output_name
units_NPP = 'gC/m2/year'
units_VEGC = 'gC/m2'

xparams, ymodel =  ut.get_params_r2_rmse(df_param,df_model,n_top_runs=5)


#Load MD3
sa_folder = 'sa_nmaxkrb_MD3_090_AK/'
df_param_MD3 = pd.read_csv(path_MD3+sa_folder+samples)
df_model_MD3 = pd.read_csv(path_MD3+sa_folder+results,header=None)

#output_name_MD3 = ['NPP0','NPP1','NPP2','NPP3',\
#                'VEGC00','VEGC01','VEGC02','VEGC03','VEGC10','VEGC11','VEGC12',\
#               'VEGC20','VEGC21','VEGC22']

output_name_MD3 = ['NPP0','NPP1','NPP2','NPP3',\
                   'VEGC00','VEGC10','VEGC20',\
                   'VEGC01','VEGC11','VEGC21',\
                   'VEGC02','VEGC12','VEGC22',\
                   'VEGC03'
                   ]
df_model_MD3.columns = output_name_MD3

xparams_MD3, ymodel_MD3 =  ut.get_params_r2_rmse(df_param_MD3,df_model_MD3,n_top_runs=5)


#MD1
sa_folder = 'SA-MD1-STEP2/'
df_param_MD1 = pd.read_csv(path_MD1+sa_folder+samples)
df_model_MD1 = pd.read_csv(path_MD1+sa_folder+'results.csv',header=None)
output_name_MD1 = ['NPP0','NPP1','NPP2','NPP3',\
                   'VEGC00','VEGC10','VEGC20',\
                   'VEGC01','VEGC11','VEGC21',\
                   'VEGC02','VEGC12','VEGC22',
                   'VEGC03'\
                   ]
df_model_MD1.columns = output_name_MD1

xparams_MD1, ymodel_MD1 =  ut.get_params_r2_rmse(df_param_MD1,df_model_MD1,n_top_runs=50)


#TK
sa_folder = 'sa-krb-NPPVEGC-TK-EJ-075/'
df_param_TK = pd.read_csv(path_TK+sa_folder+samples)
df_model_TK = pd.read_csv(path_TK+sa_folder+'results.csv',header=None)

output_name_TK = ['NPP0','NPP1','NPP2','NPP3','NPP4','NPP5','NPP6','NPP7',\
                  'VEGC00','VEGC10','VEGC20',\
                  'VEGC01','VEGC11','VEGC21',\
                  'VEGC02','VEGC12','VEGC22',\
                  'VEGC03','VEGC23',\
                  'VEGC04','VEGC24',\
                  'VEGC05','VEGC06','VEGC07'
                  ]

df_model_TK.columns = output_name_TK

xparams_TK, ymodel_TK =  ut.get_params_r2_rmse(df_param_TK,df_model_TK,n_top_runs=5)


#US-Prr
sa_folder = 'STEP2-nmax_krb-NPP_vegc/'
df_param_USPrr = pd.read_csv(path_USPrr+sa_folder+samples)
df_model_USPrr = pd.read_csv(path_USPrr+sa_folder+'results.csv',header=None)

output_name_USPrr = ['NPP0','NPP1','NPP2','NPP3','NPP4',\
                     'VEGC00','VEGC10','VEGC20',\
                     'VEGC01',\
                     'VEGC02','VEGC12','VEGC22',\
                     'VEGC03','VEGC23',\
                     'VEGC04'
                     ]

df_model_USPrr.columns = output_name_USPrr

xparams_USPrr, ymodel_USPrr =  ut.get_params_r2_rmse(df_param_USPrr,df_model_USPrr,n_top_runs=5)


nmae = calculate_nmae(df_model,ymodel)
nmae_tk = calculate_nmae(df_model_TK,ymodel_TK)
nmae_3 = calculate_nmae(df_model_MD3,ymodel_MD3)
nmae_md1 = calculate_nmae(df_model_MD1,ymodel_MD1)
nmae_usprr = calculate_nmae(df_model_USPrr,ymodel_USPrr)

nmae_matrix = pd.DataFrame({'EML21': nmae, 'TK': nmae_tk, 'MD3': nmae_3, 'MD1': nmae_md1, 'US-Prr': nmae_usprr})

site_metric_matrix(nmae_matrix.iloc[:8,:], 'NMAE')

#PLOT ZSCORE:
z = z_score(ymodel,df_model,output_name)
z = pd.Series(z, dtype=float)
z_tk = z_score(ymodel_TK,df_model_TK,output_name_TK)
z_tk = pd.Series(z_tk, dtype=float)
z_md3 = z_score(df_model_MD3,ymodel_MD3,output_name_MD3)
z_md3 = pd.Series(z_md3, dtype=float)
z_md1 = z_score(df_model_MD1,ymodel_MD1,output_name_MD1)
z_md1 = pd.Series(z_md1, dtype=float)
z_usprr = z_score(df_model_USPrr,ymodel_USPrr, output_name_USPrr)
z_usprr = pd.Series(z_usprr, dtype=float)

z_matrix = pd.DataFrame({'EML21': z[:7], 'TK': z_tk[:8], 'MD3': z_md3[:4], 'MD1': z_md1[:4], 'US-Prr': z_usprr[:5]})
z_matrix.index=output_name_TK[:8]

site_metric_matrix(z_matrix.iloc[:8,:], 'Z-score')


# # nmax+krb+cfall+nfall -> NPP + VEGC + VEGN

#Load EML21
per_number=0.5
sa_folder = 'sa_nmaxkrbcfallnfall_EML21_090_AK-2/'
df_param = pd.read_csv(path_EML+sa_folder+samples)
df_model = pd.read_csv(path_EML+sa_folder+results,header=None)

output_name = ['NPP0','NPP1','NPP2','NPP3','NPP4','NPP5','NPP6',\
               'VEGC00','VEGC10','VEGC20','VEGC30','VEGC40','VEGC50','VEGC60',\
                'VEGC01','VEGC11',\
                'VEGC02','VEGC12','VEGC22','VEG32',\
                'VEGN00','VEGN10','VEGN20','VEGN30','VEGN40','VEGN50','VEGN60',\
                'VEGN01','VEGN11',\
                'VEGN02','VEGN12','VEGN22','VEGN32']
df_model.columns = output_name
units_NPP = 'gC/m2/year'
units_VEGC = 'gC/m2'

xparams, ymodel =  ut.get_params_r2_rmse(df_param,df_model,n_top_runs=5)


#Load MD3
sa_folder = 'sa_nmaxkrbcfallnfall_MD3_090_AK/'
df_param_MD3 = pd.read_csv(path_MD3+sa_folder+samples)
df_model_MD3 = pd.read_csv(path_MD3+sa_folder+results_new,header=None)

output_name_MD3 = ['NPP0','NPP1','NPP2','NPP3',\
                'VEGC00','VEGC10','VEGC20','VEGC30',\
                'VEGC01','VEGC11','VEGC21',\
                'VEGC02','VEGC12','VEGC22',\
                'VEGN00','VEGN10','VEGN20','VEGN30',\
                'VEGN01','VEGN11','VEGN21',\
                'VEGN02','VEGN12','VEGN22']
df_model_MD3.columns = output_name_MD3

xparams_MD3, ymodel_MD3 =  ut.get_params_r2_rmse(df_param_MD3,df_model_MD3,n_top_runs=5)


#Load US-Prr
sa_folder = 'STEP3-nmax_krb_cfall_nfall-NPP_vegc_vegn/'
df_param_USPrr = pd.read_csv(path_USPrr+sa_folder+samples)
df_model_USPrr = pd.read_csv(path_USPrr+sa_folder+results_new,header=None)

output_name_USPrr = ['NPP0','NPP1','NPP2','NPP3','NPP4',\
                     'VEGC00','VEGC01','VEGC02',\
                     'VEGC10',
                     'VEGC20','VEGC21','VEGC22',\
                     'VEGC30','VEGC32',
                     'VEGC40',\
                     'VEGN00','VEGN01','VEGN02',\
                     'VEGN10',
                     'VEGN20','VEGN21','VEGN22',\
                     'VEGN30','VEGN32',
                     'VEGN40']
df_model_USPrr.columns = output_name_USPrr
xparams_USPrr, ymodel_USPrr =  ut.get_params_r2_rmse(df_param_USPrr,df_model_USPrr,n_top_runs=5)


nmae = calculate_nmae(df_model,ymodel)
# nmae_tk = calculate_nmae(df_model_TK,ymodel_TK)
nmae_3 = calculate_nmae(df_model_MD3,ymodel_MD3)
# nmae_md1 = calculate_nmae(df_model_MD1,ymodel_MD1)
nmae_usprr = calculate_nmae(df_model_USPrr,ymodel_USPrr)

nmae_matrix = pd.DataFrame({'EML21': nmae, 'MD3': nmae_3, 'US-Prr': nmae_usprr})

site_metric_matrix(nmae_matrix.iloc[8:,:], 'NMAE')

#PLOT ZSCORE:
z = z_score(ymodel,df_model,output_name)
z = pd.Series(z, dtype=float)
# z_tk = z_score(ymodel_TK,df_model_TK,output_name_TK)
# z_tk = pd.Series(z_tk, dtype=float)
z_md3 = z_score(df_model_MD3,ymodel_MD3,output_name_MD3)
z_md3 = pd.Series(z_md3, dtype=float)
# z_md1 = z_score(df_model_MD1,ymodel_MD1,output_name_MD1)
# z_md1 = pd.Series(z_md1, dtype=float)
z_usprr = z_score(df_model_USPrr,ymodel_USPrr, output_name_USPrr)
z_usprr = pd.Series(z_usprr, dtype=float)

z_matrix = pd.DataFrame({'EML21': z[7:], 'MD3': z_md3[4:], 'US-Prr': z_usprr[5:]})
z_matrix.index=output_name[4:]

site_metric_matrix(z_matrix.iloc[4:,:], 'Z-score')


# # soil parameters -> below-ground targets

#Load EML21
per_number=0.5
sa_folder = 'sa_soil_EML21_090_AK-2/'
df_param = pd.read_csv(path_EML+sa_folder+samples)
df_model = pd.read_csv(path_EML+sa_folder+results,header=None)

output_name = ['CarbonShallow','CarbonDeep','CarbonMineralSum','AvailableNitrogenSum']
df_model.columns = output_name

xparams, ymodel =  ut.get_params_r2_rmse(df_param,df_model,r2lim=per_number)


#Load MD3
per_number=0.5
sa_folder = 'sa_soil_MD3_090_AK/'
df_param_MD3 = pd.read_csv(path_MD3+sa_folder+samples)
df_model_MD3 = pd.read_csv(path_MD3+sa_folder+results,header=None)

output_name_MD3 = ['CarbonShallow','CarbonDeep','CarbonMineralSum','AvailableNitrogenSum']
df_model_MD3.columns = output_name_MD3

xparams_MD3, ymodel_MD3 =  ut.get_params_r2_rmse(df_param_MD3,df_model_MD3,r2lim=per_number)


nmae = calculate_nmae(df_model,ymodel)
# nmae_tk = calculate_nmae(df_model_TK,ymodel_TK)
nmae_3 = calculate_nmae(df_model_MD3,ymodel_MD3)
# nmae_md1 = calculate_nmae(df_model_MD1,ymodel_MD1)
# nmae_usprr = calculate_nmae(df_model_USPrr,ymodel_USPrr)

nmae_matrix = pd.DataFrame({'EML21': nmae, 'MD3': nmae_3})

site_metric_matrix(nmae_matrix.iloc[:,:], 'NMAE')

#PLOT ZSCORE:
z = z_score(ymodel,df_model,output_name)
z = pd.Series(z, dtype=float)
# z_tk = z_score(ymodel_TK,df_model_TK,output_name_TK)
# z_tk = pd.Series(z_tk, dtype=float)
z_md3 = z_score(df_model_MD3,ymodel_MD3,output_name_MD3)
z_md3 = pd.Series(z_md3, dtype=float)
# z_md1 = z_score(df_model_MD1,ymodel_MD1,output_name_MD1)
# z_md1 = pd.Series(z_md1, dtype=float)
# z_usprr = z_score(df_model_USPrr,ymodel_USPrr, output_name_USPrr)
# z_usprr = pd.Series(z_usprr, dtype=float)

z_matrix = pd.DataFrame({'EML21': z, 'MD3': z_md3})
z_matrix.index=output_name

site_metric_matrix(z_matrix.iloc[:,:], 'Z-score')

