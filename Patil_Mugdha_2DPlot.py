# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:15:46 2020

@author: mugdha
"""

import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

with open('IrisData.txt', 'r') as in_file:
    lines1 = in_file.read().splitlines()
    stripped = [line.replace(" "," ").split() for line in lines1]
    grouped = zip(*[stripped])
    with open('M1.txt', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'))
        for group in grouped:
            writer.writerows(group)
        
m = pd.read_csv('M1.txt')        


sns.lmplot( x="sepal_length", y="petal_length", data=m, fit_reg=False, hue='species', legend=False , markers=["o", "x", "^"])
plt.legend(loc='lower right')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Three types of Iris flowers')
plt.savefig('Patil_Mugdha_MyPlot')