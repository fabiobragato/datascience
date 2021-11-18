import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import boto3

from io import StringIO
from datetime import datetime
from dateutil.relativedelta import relativedelta

pd.set_option('mode.chained_assignment', None)

def create_variables_in_time(df, dateid, id, operation, time, support_var, filetype, path="", bucket_name="", path_s3="", key="", ACCESS_KEY="", SECRET_KEY=""):
    
    # Auxiliar list containing all dates present on dataset
    dates = list(df[dateid].sort_values(ascending=False).drop_duplicates())
    
    # Auxiliar list containing all id's and dates present on dataset
    df_id_months = df[[id, dateid]].drop_duplicates()
    
    # Empty auxiliar list for the process
    dfList = []
    
    for dt in dates:
        print(f'Date to Process: \n {dt}')
        print(' Processing...')

        # Collect the id's present in the processing date
        ids_from_date = list(df_id_months[df_id_months[dateid] == dt][id].drop_duplicates())

        for t in time:
            # Declare the initial date parameter
            start_date = dt + relativedelta(months= -t + 1)

            # Declare the final date parameter
            final_date = dt
            
            # Create the mask to filter the dataframe based on the range date
            date_mask = (df[dateid] >= start_date) & (df[dateid] <= final_date)

            # Filter the dataframe based on mask created above
            df_fil_dt = df[date_mask]
            
            # Drop the date column
            df_fil_dt.drop(dateid, axis=1, inplace=True)

            # Filter only id's present in the date analysed
            df_fil_id = df_fil_dt[df_fil_dt[id].isin(ids_from_date)]

            # Do the aggregations based on operations declared as parameter
            df_fil_id = df_fil_id.groupby([id]).agg(operation)

            # Create new coluns based on the columnname_time_operation (ex: vl_risco_3m_mean, vl_risco_6m_max)
            df_fil_id.columns = [f'_{t}m_'.join(col) for col in df_fil_id.columns]

            # Reset the column index
            df_fil_id.reset_index(inplace=True)
            
            # Append each new dataframe on list 
            dfList.append(df_fil_id)

        # Set 'tid' as index
        dfs = [dataframe.set_index(id) for dataframe in dfList]
        
        # Concatenate all dataframes on list
        result = pd.concat(dfs, axis=1)
        
        # Adds the original variables
        df_final = pd.merge(df[df[dateid] == dt], result, how="left", on=[id])
        
        if support_var == 1:
            
            # Create a support df
            df_sup = df[[dateid, id]]
            
            for t in time:
                # Create the support variable that indicates if id has all dates for analysis
                df_sup[f'dateid_{t}m'] = df_sup.groupby([id])[dateid].shift(t)

                # Create the flag if id has all dates
                df_sup[f'flag_{t}m'] = df_sup[f'dateid_{t}m'].apply(lambda x: 0 if pd.isnull(x) else 1)

                # Delete the support column
                df_sup.drop(f'dateid_{t}m', axis=1, inplace=True)
                
            df_final = pd.merge(df_final, df_sup, how='left', on=[id, dateid])
        
        if filetype == 'csv':
            # Save in csv mode
            df_final.to_csv(f'{path}{dt.year}_{dt.month}.csv', sep=';', decimal=',', index=False)
        
        elif filetype == 'parquet':
            # Save in parquet mode
            df_final.to_parquet(f'{path}{dt.year}_{dt.month}.parquet')
            
        elif filetype == 's3':
            # Save in S3 AWS           
            csv_buffer = StringIO()
            df_final.to_csv(csv_buffer)
            s3_resource = boto3.resource('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key= SECRET_KEY)
            s3_resource.Object(bucket_name, f'{path_s3}{key}.csv').put(Body=csv_buffer.getvalue())
            
        else:
            print('Filetype input not recognized. File not saved.')

        # Clear dataframe list
        dfList.clear()
    
    print('Process Finished')
