import pandas as pd
import os
from functools import reduce

# Define the directory containing the CSV files
directory = 'data/all_terciaria_df'

# List to hold dataframes
dataframes = []

# Counter to keep track of batches
batch_counter = 0
j = 0
# Iterate over all files in the directory
for filename in os.listdir(directory):
    print(j)
    j += 1
    if filename.endswith('.csv'):
        # Construct full file path
        file_path = os.path.join(directory, filename)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        # Append the DataFrame to the list
        dataframes.append(df)
        
        # If we have 100 dataframes, process the batch
        if len(dataframes) == 100:
            # Merge the batch of 100 dataframes
            merged_df = reduce(lambda x, y: pd.merge(x, y, 
                                                     on=['datetime', 'geo_id', 'geo_name'], 
                                                     how='outer').set_index('datetime'), 
                               dataframes)
            # Save the merged batch to a file
            merged_df.to_csv(f'merged_batch_{batch_counter}.csv')
            # Increment the batch counter
            batch_counter += 1
            # Clear the list for the next batch
            dataframes = []

# Process any remaining dataframes that didn't make a full batch
if dataframes:
    merged_df = reduce(lambda x, y: pd.merge(x, y, 
                                             on=['datetime', 'geo_id', 'geo_name'], 
                                             how='outer').set_index('datetime'), 
                       dataframes)
    merged_df.to_csv(f'merged_batch_{batch_counter}.csv')

# Optionally, you can merge the resulting batch files if needed
# This part is optional and depends on your use case
batch_files = [f'merged_batch_{i}.csv' for i in range(batch_counter + 1)]
final_dfs = [pd.read_csv(file).set_index('datetime') for file in batch_files]
final_df = reduce(lambda x, y: pd.merge(x, y, 
                                        on=['datetime', 'geo_id', 'geo_name'], 
                                        how='outer'), 
                  final_dfs)
final_df.to_csv('data/all_terciaria_df_merged.csv')

import pandas as pd
import numpy as np
# df = pd.read_csv('data/all_terciaria_merged/all_terciaria_df_merged.csv')

# df = df[df.geo_id==8741].drop(columns=['geo_id','geo_name'])
# df = df.dropna(how='all')
# df.to_csv('data/all_terciaria_df/merged_peninsula.csv')
df = pd.read_csv('data/all_terciaria_df/merged_peninsula.csv')

freq = []
h = [5*i for i in range(12)]
for i in range(12):
    freq.append((len(df.iloc[i])-df.iloc[i].isna().sum())/len(df.iloc[i]))
    
import matplotlib.pyplot as plt
plt.close()
plt.plot(h,freq,linewidth=3,marker='o',markersize=8)
plt.xlabel('Minutes', fontsize=24)  # Increase x-axis label size
plt.ylabel('Percentage of available data per row of merged_peninsula.csv', fontsize=24)  # Increase y-axis label size
plt.xticks(fontsize=24)  # Increase font size of x-axis ticks
plt.yticks(np.linspace(0,1,11),fontsize=24)
plt.grid()
plt.savefig('figs/available_data_merged_peninsula.pdf', format='pdf')  # Save as PNG file
plt.show()

import pandas as pd

# Read the CSV file
df = pd.read_csv('data/all_terciaria_df/merged_peninsula.csv')

# Convert 'datetime' column to datetime type
df = df.drop(columns="Unnamed: 0")

df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

# Filter rows to include only those at 15-minute intervals
df = df[df.datetime.dt.minute % 15 == 0]

# Create a new column for the hour
df['hour'] = df['datetime'].dt.floor('H')

# Create a new column for the 15-minute interval within the hour
df['minute'] = df['datetime'].dt.strftime('%M')

# Pivot the DataFrame
pivot_df = df.pivot(index='hour', columns='minute')

# Flatten the MultiIndex columns
pivot_df.columns = [f'{col[0]}.{col[1]}' for col in pivot_df.columns]

# Reset the index to make 'hour' a column again
pivot_df.reset_index(inplace=True)
pivot_df=pivot_df.drop(columns=['datetime.00','datetime.15','datetime.30','datetime.45'])
pivot_df = pivot_df.dropna(axis=1,how='all')
pivot_df = pivot_df.fillna(0)
pivot_df.to_csv("data/all_terciaria_merged/pivoted_quarterly.csv")

