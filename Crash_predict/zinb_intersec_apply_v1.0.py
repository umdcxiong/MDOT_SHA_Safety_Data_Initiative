# zinb model
# extract volume data to apply on the model and output for different seasons, DOW, TOD
import pandas as pd
import numpy as np
import os
from math import exp
from time import perf_counter

version = '1.0'
Measure = ['Vehicle Volume', 'Pedestrian Bicycle Volume']
TOD = ['AM', 'PM', 'Midday', 'Night']

'''have to copy and paste to
a csv file
and create a column named 'SubModel' manually, as below: '''

intersec_level_data = pd.read_csv(
    '../Crash_model/intersection_data.csv', index_col=0)

for i in list(range(1, 4)):
    intersec_level_data['leg2_%s' % i] = np.where(
        intersec_level_data['leg2'] == i, 1, 0)

coefs = pd.read_csv(
    'input_data/coef_int_v%s.csv' % version, sep=',')
coefs['P>|z|'] = pd.to_numeric(coefs['P>|z|'], errors='coerce')
coefs['Coef.'] = pd.to_numeric(coefs['Coef.'], errors='coerce')


all_coef_list = list(coefs.loc[coefs['SubModel'].notnull()]['Variables'])
all_coef_list = list(filter(lambda a: a != '_cons', all_coef_list))

nb_coef = coefs.loc[coefs['SubModel'] ==
                    'Negative Binomial'][['Variables', 'Coef.']]
infl_coef = coefs.loc[coefs['SubModel'] ==
                      'Logit'][['Variables', 'Coef.']]

nb_coef_list = list(nb_coef['Variables'])
infl_coef_list = list(infl_coef['Variables'])


def filterData(intersec_level_data):
    filtered_data_nb = intersec_level_data[filter(
        lambda a: a != '_cons', nb_coef_list)]
    filtered_data_infl = intersec_level_data[filter(
        lambda a: a != '_cons', infl_coef_list)]
    filtered_data_nb['_cons'] = 1
    filtered_data_infl['_cons'] = 1

    filtered_data = pd.concat([filtered_data_nb, filtered_data_infl], axis=1)
    return filtered_data


def ZeroInflatedNegativeBinomial(nb_coef, infl_coef, dt_row):

    a1 = (nb_coef['Coef.'].values*dt_row[:len(nb_coef)].values).sum()

    a2 = (infl_coef['Coef.'].values*dt_row[len(nb_coef):].values).sum()

    pzero = exp(a2)/(1+exp(a2))
    pcount = exp(a1)*(1-pzero)

    return pcount


if __name__ == "__main__":
    t1_start = perf_counter()
    filtered_data = filterData(intersec_level_data)
    intersec_level_data['Vulnerable road user safety risk'] = filtered_data.apply(
        lambda dt_row: ZeroInflatedNegativeBinomial(nb_coef, infl_coef, dt_row), axis=1)
    intersec_level_data.to_csv(
        'output_data/intersec_pred_results_%s.csv' % version)
    t1_stop = perf_counter()
    print("Elapsed time during the whole program in seconds:",
          t1_stop-t1_start)
