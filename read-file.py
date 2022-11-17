import pandas as pd 
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt 


df = pd.read_csv(".\data\coefficient_variation.csv")
print(df,type(df))
df.info()