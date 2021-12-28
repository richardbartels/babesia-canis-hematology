#!/usr/bin/env python
# coding: utf-8

# # Analayse test predictions
# Analyses of initially false positive validation patients, that were re-examined for the presence of B. Canis.

# In[1]:


import numpy as np
import pandas as pd


# ## Analysis

# In[2]:


df_pred = pd.read_csv("../data/processed/predictions_random_forest_test.csv").astype({"Identifier":'str'})
df_test = pd.read_csv("../data/processed/test.csv", index_col=False).astype({"Identifier":'str'})


# In[3]:


(df_pred["Identifier"] == df_test["Identifier"]).all()
df = df_pred.copy()
df['y_true'] = df_test["Group"].copy()


# In[4]:


mask = df['y_true']==1
print(f"TP: {(df.loc[mask, 'y_pred'] == df.loc[mask, 'y_true']).sum()}")
mask = df['y_true']==0
print(f"FP: {(df.loc[mask, 'y_pred'] != df.loc[mask, 'y_true']).sum()}")


# In[5]:


mask_fp = (df['y_true']==0) & (df.loc[mask, 'y_pred'] != df.loc[mask, 'y_true'])


# In[6]:


df.loc[mask_fp]


# In[7]:


babesia_positive_ids = np.array([117011110202,117060700901,
117092606101,
117112011702,
118052405402,
118051105704]).astype('str')


# In[8]:


assert np.array([x in df.loc[:,"Identifier"].values for x in babesia_positive_ids]).all()
np.array([x in df.loc[mask_fp,"Identifier"].values for x in babesia_positive_ids])  # other 4 already positive


# In[9]:


df.loc[df['Identifier'].isin(babesia_positive_ids)]


# In[10]:



# for x in df.loc[mask_fp].sort_values(by="Identifier")["Identifier"]:
fp_ids = df.loc[mask_fp].sort_values(by="Identifier")["Identifier"].values
print(fp_ids.shape)
np.savetxt("../data/processed/fp_ids_validation.txt", fp_ids, fmt='%s', header='Identifier')


# In[11]:


df.loc[mask_fp].sort_values(by="Identifier")


# ## Complete list

# In[12]:


def load_reanalysis_list():
    df = pd.concat([pd.read_csv("../data/processed/Scan21091408_18_05.csv", 
                           usecols=range(i*3, i*3+3), 
                           index_col=False,
                           header=0,
                           names=['Identifier', 'analysis', 'note'])
               for i in range(4)], axis=0, ignore_index=True)

    df.dropna(subset=['Identifier'], inplace=True)
    assert ~df.Identifier.duplicated().any()
    df = df.astype({'Identifier': 'int'}).astype({'Identifier': 'str'})  # scientific to integer notation

    df['label'] = (df['analysis'].apply(lambda x: 'bab' in x.replace(' ', '').split(','))).astype('int')
    assert df.label.sum() == 6, f"Sum is {df.label.sum()} instead of 6."
    return df


# In[35]:


df_reanalysis = load_reanalysis_list()
# df_reanalysis = df_reanalysis.astype({'patient_id': 'str'})
assert df_reanalysis.loc[df_reanalysis.label==1].Identifier.isin(babesia_positive_ids.astype('str')).all()

print("Patients not in validation dataset:")
df_reanalysis.loc[~df_reanalysis.Identifier.isin(df.Identifier)]


# ### Compare to random forest predictions

# In[14]:


df_rf = df_pred.loc[df_pred.y_pred==1]
print(f"Patients FP in both RF and conventional model: {df_rf.Identifier.isin(df_reanalysis.Identifier).sum()}")
print(f"Deduplicated: {(~df_rf.loc[df_rf.Identifier.isin(df_reanalysis.Identifier)].duplicated(subset=['Identifier'])).sum()} (out of {df_reanalysis.shape[0]})")
print("Patients only in reanalysis file:")
df_reanalysis.loc[~df_reanalysis.Identifier.isin(df_rf.Identifier)]


# In[15]:


temp = df_rf.loc[~df_rf.Identifier.isin(df_reanalysis.Identifier)]
print(f"Patients only in Random Forest ({(~temp.duplicated(subset=['Identifier'])).sum()}):")
temp


# #### Anaplasma

# In[45]:


df_reanalysis['anaplasma/elichlia'] = df_reanalysis['analysis'].apply(lambda x: 'anaplasma' in x or 'ehlichlia' in x)


# In[46]:


df_reanalysis.loc[df_reanalysis['anaplasma/elichlia']]
assert df_reanalysis.loc[df_reanalysis['anaplasma/elichlia']].Identifier.isin(df_rf.Identifier).all()

df_rf.merge(df_reanalysis.loc[df_reanalysis['anaplasma/elichlia']], how='right', on='Identifier')


# In[ ]:




