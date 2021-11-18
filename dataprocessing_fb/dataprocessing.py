import pandas as pd
import numpy as np

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