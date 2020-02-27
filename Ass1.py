import numpy as np
import pandas as pd

data = pd.read_excel (r'C:\Users\Thierry\Documents\Studie\TU Delft Applied Physics\CS4195 Modeling and Data Analysis in Complex Networks\Assignment1\manufacturing_emails_temporal_network.xlsx')

#%% 1)
Nnodes = np.max([data['node1'].max(), data['node2'].max()])
Nlinks = len(data)
