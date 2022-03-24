#!/usr/bin/env python
# coding: utf-8

# In[6]:


import mlxtend
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder


# In[7]:


dataset = [['Skirt', 'Sneakers', 'Scarf', 'Pants', 'Hat'],

    ['Sunglasses', 'Skirt', 'Sneakers', 'Pants', 'Hat'],

    ['Dress', 'Sandals', 'Scarf', 'Pants', 'Heels'],

    ['Dress', 'Necklace', 'Earrings', 'Scarf', 'Hat', 'Heels', 'Hat'],

   ['Earrings', 'Skirt', 'Skirt', 'Scarf', 'Shirt', 'Pants']]


# In[8]:


te=TransactionEncoder()
te_ary=te.fit(dataset).transform(dataset)    
df=pd.DataFrame(te_ary, columns=te.columns_)  
df


# In[9]:


from mlxtend.frequent_patterns import apriori
apriori(df, min_support=0.6)


# In[10]:


frequent_itemsets=apriori(df, min_support=0.6, use_colnames=True) 
frequent_itemsets


# In[11]:


from mlxtend.frequent_patterns import association_rules 
association_rules(frequent_itemsets,metric="confidence",min_threshold=0.7)


# In[12]:


from mlxtend.frequent_patterns import association_rules 
association_rules(frequent_itemsets,metric="lift",min_threshold=1.25)


# In[ ]:




