import pandas as pd
import numpy as np
import os

df = pd.read_csv('https://raw.githubusercontent.com/araj2/customer-database/master/Ecommerce%20Customers.csv')

df = df.iloc[:, 3:]

df = df[df['Length of Membership'] > 1]

df.drop(columns=['Avg. Session Length'], inplace = True)


base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)

data_path = os.path.join(base_dir, 'data', 'processed_data.csv')

print("Processed data file will be stored at:", data_path)


df.to_csv(data_path, index=False)

