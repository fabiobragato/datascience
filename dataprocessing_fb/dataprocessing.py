import pandas as pd
import numpy as np
import random

def nullTable(dataframe):
    # Find the null numbers of dataframe and count them
    null_numbers = dataframe.isnull().sum().sort_values(ascending=False)

    # Calculate the percentual of nulls
    null_percent = (dataframe.isnull().sum()/dataframe.count()).sort_values(ascending=False)

    # Concat both dataframes
    null_table = pd.concat([null_numbers, null_percent], axis = 1, keys=['countNulls', 'pcNulls'])

    print('Descriptive Analysis: ', '\n')
    print('Number of Observed Variables: ', null_table.count()[0])
    print('Number of Variables with Nulls: ', null_table[null_table['countNulls'] > 0].count()[0])
    print('Percentual of Variables with Nulls: ', (null_table[null_table['pcNulls'] > 0].count()[0]/null_table.count()[0]), '%')
    print('\n')
    print('Resume (top five):')
    print('\n')
    print(null_table.head(5))

    return null_table

def selectNumVar(dataframe):
    # Find only numerical variables
    return list(dataframe.select_dtypes('number').columns)

def selectCatVar(dataframe):
    # Find only categorical variables
    return list(dataframe.select_dtypes('object').columns)

def separate_data_on_time_for_ml(dataframe, dateid, id, frac_dateid, frac_id, random_seed = ""):
    
    # Separates dateid and id inside tuples and armazenate it in a pd.Series
    data = pd.Series([tuple(x) for x in dataframe[np.append(dateid, id)].to_numpy()])
    
    # Separates a list of uniques (dateid)
    list_unique_dateid = sorted(list(set(list(dataframe[dateid]))))
    
    # Separates a list of uniques (id)
    list_unique_id = sorted(list(set(list(dataframe[id]))))
    
    # Separates lists of dates in train and list of dates in test
    list_dateid_train = list_unique_dateid[:round(len(list_unique_dateid) * (1 - frac_dateid))] 
    list_dateid_test = list_unique_dateid[round(len(list_unique_dateid) * (1 - frac_dateid)):]
    
    # Separates ids randomly in a list
    if random_seed != "":
        r_seed = random_seed
    
    list_random_unique_id = random.sample(list_unique_id, len(list_unique_id))
    
    # Separates lists of id's in train and list of id's in test
    list_id_train = list_unique_id[:round(len(list_unique_id) * (1 - frac_id))] 
    list_id_test = list_unique_id[round(len(list_unique_id) * (1 - frac_id)):]
    
    # Separates data in a list of tuples based on it's description
    
    ## Data for train
    tuples_for_train = [(d, i) for d in list_dateid_train for i in list_id_train]

    ## First Data for testing (id's that are in the same date of train that won't be in the training model)
    tuples_for_test_1 = [(d, i) for d in list_dateid_train for i in list_id_test]
    
    ## Second Data for testing (id's that aren't in the same date of train but will be in the training model)
    tuples_for_test_2 = [(d, i) for d in list_dateid_test for i in list_id_train]
    
    ## Third Data for testing (id's that aren't in the same date of train and won't be in the training model)
    tuples_for_test_3 = [(d, i) for d in list_dateid_test for i in list_id_test]
    
    # Separates data for each tuple
    data_for_train = data.isin(tuples_for_train).values
    data_for_test_1 = data.isin(tuples_for_test_1).values
    data_for_test_2 = data.isin(tuples_for_test_2).values
    data_for_test_3 = data.isin(tuples_for_test_3).values
    
    # Creates the final dataframe with mask's information
    final_df = dataframe[[dateid]].copy()
    final_df['data_for_train'] = data_for_train
    final_df['data_for_test_1'] = data_for_test_1
    final_df['data_for_test_2'] = data_for_test_2
    final_df['data_for_test_3'] = data_for_test_3
    
    # Group by dateid (count the number of true's)
    final_df_grouped = final_df.groupby(dateid).sum()
    
    return [final_df_grouped, tuples_for_train, tuples_for_test_1, tuples_for_test_2, tuples_for_test_3]