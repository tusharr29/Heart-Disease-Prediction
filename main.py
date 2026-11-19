import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
import scipy.stats as stats

df = pd.read_csv('heart.csv')


#Exploratory Data Analysis (EDA)
print(df.columns)

print(df.duplicated().sum())