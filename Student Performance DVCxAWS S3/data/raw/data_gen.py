import pandas as pd
import numpy as np

# number of rows
n =1000

# create student performance data
data = {
  'IQ': np.random.randint(80, 140, n).round(2),
  'CGPA': np.random.uniform(5,10,n).round(2),
  '10th_marks': np.random.randint(60,100,n),
  '12th_marks': np.random.randint(60,100,n),
  'Communication_Skills': np.random.randint(1,10,n),
  'Placed': np.random.randint(0,2,n)
}

df = pd.DataFrame(data)

# save the data
df.to_csv('data/raw/student_performance.csv', index=False)