#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

def loader(stat,fant):
    post = ['QB', 'RB', 'WR','TE', 'K', 'DST']
    global df4
    df4 = []
    
    for i in post:
        
        df1 = pd.read_excel(stat, sheet_name= i, encoding='utf-8')
        df2 = pd.read_excel(fant, sheet_name= i, encoding='utf-8')
        
        if i == 'DST':
            df1.rename(columns={'Team': 'TEAM'}, inplace=True) 
            df2['Team'] = df2['Team'].apply(lambda x: x.split(' ')[-1])
            df2.rename(columns={'Team': 'TEAM'}, inplace=True)
            df3 = pd.merge(df1, df2, on='TEAM')
        else:
            df1['TEAM'] = df1['TEAM'].apply(lambda x: x.replace('JAX', 'JAC'))
            df1['TEAM'] = df1['TEAM'].apply(lambda x: x.replace('OAK', 'LV'))
            
            df2.rename(columns={'Player': 'NAME'}, inplace=True)
            
            df3 = pd.merge(df1,df2, on='NAME')
        
        df4.append(df3)

loader('data_sets/All_Set.xlsx', 'fantasy_points/Fantasy_All_2019.xlsx')

qb = df4[0]
rb = df4[1]
wr = df4[2]
te = df4[3]
k = df4[4]
dst = df4[5]

