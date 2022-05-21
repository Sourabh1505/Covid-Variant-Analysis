#!/usr/bin/env python
# coding: utf-8

# # Importing all the required libraries.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Import Data

# In[2]:


df = pd.read_csv("C:/Users/sloha/Desktop/Sourabh/covid-variants.csv")


# In[3]:


df


# # Data Cleaning

# In[4]:


df.isnull().sum()


# In[5]:


sns.heatmap(df.isnull(),yticklabels=False)


# ## Convert dtype object to datetime

# In[6]:


df["date"] = pd.to_datetime(df["date"])
df["date"]


# ## create seperate columns of year,month,day

# In[7]:


df["year"]=df["date"].dt.year
df["month"]=df["date"].dt.month
df["day"]=df["date"].dt.day


# In[8]:


df


# In[9]:


df


# # Co-Relation

# In[10]:


df.corr()


# In[11]:


sns.heatmap(df.corr(),linewidths=1,annot = True , cmap = "Blues")


# In[12]:


df


# In[13]:


df[df.year == 2020]


# ##  checking  description of the data 

# In[14]:


df.describe().T


# ## Finding and checking for unique value from each column in dataset 

# In[15]:


print(f"Countries number: {df.location.nunique()}")
print(f"Date number: {df.date.nunique()}")
print(f"Variants number: {df.variant.nunique()}")
print(f"Variants names:\n \n {df.variant.unique()}")


# ## List and number of countries whose data is included in the file

# In[16]:


print(f" List of Countries data available in data set:\n \n {df.location.unique()}")


# ## The range from when the data is available and from which location

# In[17]:


first_date = pd.to_datetime(df['date']).min()
last_date = pd.to_datetime(df['date']).max()
print(f"First case registered on {first_date} at {df['location'][pd.to_datetime(df['date']).idxmin()]} ,Variant : {df['variant'][pd.to_datetime(df['date']).idxmin()]}")
print(f"Last case registered on {last_date} at {df['location'][pd.to_datetime(df['date']).idxmax()]},Variant : {df['variant'][pd.to_datetime(df['date']).idxmax()]}")


# In[18]:


place = df.location.unique()
num_sequence = []
num_total = []
for i in place:
    x = df[df.location.values == i]
    num_seq = sum(x.num_sequences)
    num_tot = sum(x.num_sequences_total)
    num_sequence.append(num_seq)
    num_total.append(num_tot)

covid_location = pd.DataFrame({"location" :place,"num_sequences_processed" : num_sequence, "acknowledget_num_sequences_total" :num_total})
covid_location 


# In[19]:


f,ax = plt.subplots(figsize = (15,30))
sns.barplot(y = covid_location.location,x = covid_location["acknowledget_num_sequences_total"],palette = "rocket" )
plt.xticks(rotation = 90)
plt.ylabel("Countries")
plt.xlabel("Number of Sequences" )
plt.title("Total Number of Sequences by Countries",color = "black", fontsize = 18)
plt.grid()
plt.show


# In[20]:


plt.subplots(figsize = (15,30))
sns.barplot(x=covid_location["num_sequences_processed"] ,y= covid_location.location ,palette = "rocket" )
plt.ylabel("Countries")
plt.xlabel("Number of Sequences Processed )" )
plt.title("Total Number of Sequences Processed",color = "black", fontsize = 20)
plt.grid()
plt.show()


# In[21]:


variants = df.variant.unique()
variant_num_seq = []
for i in variants:
    x = df[df.variant.values==i]
    num_seq = sum(x.num_sequences)
    variant_num_seq.append(num_seq)



variant_set = pd.DataFrame({"variant" :variants,"number_of_sequence" : variant_num_seq   })
var_index = variant_set.number_of_sequence.sort_values(ascending=False).index.values
variant_set = variant_set.reindex(var_index)
variant_set


# In[22]:


plt.subplots(figsize=(15,15))
sns.barplot(y=variant_set.variant,x=variant_set.number_of_sequence)
plt.ylabel("Variant")
plt.xlabel("Variant Number of Sequence")
plt.title("Number of Sequence by Variant")


# ## Converting all the data set into Pivot table to get required value against num_sequences

# In[23]:


df_num_seq = pd.pivot_table(df,'num_sequences', index = ['location'], columns = ['variant'], 
                          aggfunc={'num_sequences': np.sum})

df_num_seq.head(5)


# ## Due to that here drop the varient which was not labeled by WHO

# In[24]:


df_num_seq.drop( ['B.1.1.277','B.1.1.302','B.1.1.519','B.1.160','B.1.177','B.1.221','B.1.258','B.1.367','B.1.620','Epsilon', 
           'Eta','Iota','Kappa','Lambda','Mu','S:677H.Robin1','S:677P.Pelican','others','non_who'],axis = 1,inplace= True)


# ## Making new column of total cases country/ location wise in df_num_seq dataframe

# In[25]:


sum_vairant = df_num_seq["Alpha"] + df_num_seq["Beta"] + df_num_seq["Delta"] + df_num_seq["Gamma"] +df_num_seq["Omicron"]
df_num_seq["Total"] = sum_vairant


# ## Sorting all the column datain descending order..

# In[26]:


df_num_seq.sort_values(by = ['Alpha','Beta','Delta','Gamma','Omicron','Total'],ascending = [False,False,False,False,False,False],inplace=True)


# In[27]:


df_num_seq.reset_index(inplace=True)


# In[28]:


df_num_seq.head(5)


# In[29]:


for_map = df_num_seq.iloc[0:10]
for_map


# In[30]:


for_map1 = df_num_seq.iloc[-10:]
for_map1


# In[31]:


f,ax = plt.subplots(figsize = (30,15))
sns.lineplot(x='location',y='Alpha',data=for_map)
sns.lineplot(x='location',y='Beta',data=for_map)
sns.lineplot(x='location',y='Delta',data=for_map)
sns.lineplot(x='location',y='Gamma',data=for_map)
sns.lineplot(x='location',y='Omicron',data=for_map)
plt.xlabel("Countries", fontsize=25)
plt.ylabel("Sample of variant", fontsize=20)
plt.title("Top 10 countries with most Spread of different variants of Covid-19", fontsize = 25)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.legend(labels=["Alpha","Beta","Delta","Gamma","Omicron"], title = "Variant", 
           fontsize = '20', title_fontsize = "20")
plt.grid()
plt.show


# In[32]:


f,ax = plt.subplots(figsize = (30,15))
sns.lineplot(x='location',y='Alpha',data=for_map1)
sns.lineplot(x='location',y='Beta',data=for_map1)
sns.lineplot(x='location',y='Delta',data=for_map1)
sns.lineplot(x='location',y='Gamma',data=for_map1)
sns.lineplot(x='location',y='Omicron',data=for_map1)
plt.xlabel("Countries", fontsize=20)
plt.ylabel("Sample of variant", fontsize=20)
plt.title("Bottom 10 countries with less Spread of different variants of Covid-19", fontsize = 25)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.legend(labels=["Alpha","Beta","Delta","Gamma","Omicron"], title = "Variant", 
           fontsize = '20', title_fontsize = "20")
plt.grid()
plt.show


# In[33]:


plt.figure(figsize=(12, 12))
colors = sns.color_palette('pastel')[0:15]
plt.pie(x=for_map["Total"].iloc[0:15], labels=for_map["location"].iloc[0:15], colors= colors, autopct='%1.2f%%')
plt.show()


# In[34]:


df.head(5)


# # Seperate variants 

# In[35]:


Omicron = df[df['variant'] == 'Omicron']
Delta = df[df['variant'] == 'Delta']
Beta = df[df['variant'] == 'Beta']
Alpha= df[df['variant'] == 'Alpha']
Gamma= df[df['variant'] == 'Gamma']
others = df[(df['variant'] != 'Beta') & (df['variant'] != 'Delta') & (df['variant'] != 'Omicron') & (df['variant'] != 'Alpha') & (df['variant'] != 'Gamma')]


# # Groupin all the above data with respect to date

# In[36]:


omicron_data = Omicron.groupby("date").sum()
omicron_data["date"] = omicron_data.index
delta_data = Delta.groupby("date").sum()
delta_data["date"] = delta_data.index
beta_data = Beta.groupby("date").sum()
beta_data["date"] = beta_data.index
alpha_data = Alpha.groupby("date").sum()
alpha_data["date"] = alpha_data.index
gamma_data = Gamma.groupby("date").sum()
gamma_data["date"] = gamma_data.index
others_data = others.groupby("date").sum()
others_data["date"] = others_data.index


# ## Ploting line plot variants Vs Time or Date for visualizing spread of Covid-19 in world

# In[37]:


plt.figure(figsize = (15,8))
sns.lineplot(x=omicron_data["date"],y=omicron_data["num_sequences"],label="Omicron",linestyle="--")
sns.lineplot(x=delta_data['date'],y=delta_data['num_sequences'],label='Delta',linestyle="--")
sns.lineplot(x=beta_data["date"],y=beta_data['num_sequences'],label='Beta',linestyle="--")
sns.lineplot(x=alpha_data['date'],y=alpha_data['num_sequences'],label='Alpha',linestyle="--")
sns.lineplot(x=gamma_data["date"],y=gamma_data['num_sequences'],label='Gamma',linestyle="--")
sns.lineplot(x=others_data['date'],y=others_data['num_sequences'],label='others',linestyle="--")
plt.xticks(rotation=90)
plt.xlabel("Date", fontsize=16)
plt.ylabel("num_sequences", fontsize=16)
plt.title('Covid-19 cases per day through out the world',fontsize=16)
plt.show()


# In[38]:


Omicron_ind = Omicron[Omicron['location'] == 'India']
Delta_ind = Delta[Delta['location'] == 'India']
Beta_ind = Beta[Beta['location'] == 'India']
Alpha_ind = Alpha[Alpha['location'] == 'India']
Gamma_ind= Gamma[Gamma['location'] == 'India']
others_ind = others[others['location'] == 'India']


# ## Since others contain different type of variants which may be recorded on same date

# In[39]:


others_ind_data = others_ind.groupby("date").sum()


# In[40]:


others_ind_data["date"] = others_ind_data.index
others_ind_data.head(5)


# In[41]:


plt.figure(figsize = (15,8))
sns.lineplot(x=Omicron_ind["date"],y=Omicron_ind["num_sequences"],label="Omicron",linestyle="--")
sns.lineplot(x=Delta_ind['date'],y=Delta_ind['num_sequences'],label='Delta',linestyle="--")
sns.lineplot(x=Beta_ind["date"],y=Beta_ind['num_sequences'],label='Beta',linestyle="--")
sns.lineplot(x=Alpha_ind['date'],y=Alpha_ind['num_sequences'],label='Alpha',linestyle="--")
sns.lineplot(x=Gamma_ind["date"],y=Gamma_ind['num_sequences'],label='Gamma',linestyle="--")
sns.lineplot(x=others_ind['date'],y=others_ind['num_sequences'],label='others',linestyle="--")
plt.xticks(rotation=90)
plt.xlabel("Date", fontsize=16)
plt.ylabel("num_sequences", fontsize=16)
plt.title('Covid-19 Spread in India',fontsize=16)
plt.show()


# # Comparing each varient spread India Vs World

# In[42]:


plt.figure(figsize=(20,15))

plt.subplot(1,2,1).set_title("Spread of Omricron variant India v/s World", fontdict= { 'fontsize': 18, 'fontweight':'bold'})
sns.lineplot(x=Omicron_ind['date'],y=Omicron_ind["num_sequences"],label="India",)
sns.lineplot(x=omicron_data["date"],y=omicron_data["num_sequences"],label="World")
plt.xticks(rotation=90)
plt.grid()
plt.xlabel("Date", fontsize=14)
plt.ylabel("num_sequences", fontsize=14)

plt.subplot(1,2,2).set_title("Spread of Alpha variant India v/s World", fontdict= { 'fontsize': 18, 'fontweight':'bold'})
sns.lineplot(x=Alpha_ind['date'],y=Alpha_ind["num_sequences"],label="India")
sns.lineplot(x=alpha_data["date"],y=alpha_data["num_sequences"],label="World")

plt.xticks(rotation=90)
plt.xlabel("Date", fontsize=14)
plt.ylabel("num_sequences", fontsize=14)
plt.grid()
plt.show()


# # Comparing each varient spread India Vs World

# In[43]:


plt.figure(figsize=(20,15))

plt.subplot(1,2,1).set_title("Spread of Beta variant India v/s World", fontdict= { 'fontsize': 18, 'fontweight':'bold'})
sns.lineplot(x=Beta_ind['date'],y=Beta_ind["num_sequences"],label="India",)
sns.lineplot(x=beta_data["date"],y=beta_data["num_sequences"],label="World")
plt.xticks(rotation=90)
plt.grid()
plt.xlabel("Date", fontsize=14)
plt.ylabel("num_sequences", fontsize=14)

plt.subplot(1,2,2).set_title("Spread of Delta variant India v/s World", fontdict= { 'fontsize': 18, 'fontweight':'bold'})
sns.lineplot(x=Delta_ind['date'],y=Delta_ind["num_sequences"],label="India")
sns.lineplot(x=delta_data["date"],y=delta_data["num_sequences"],label="World")
plt.xticks(rotation=90)
plt.grid()
plt.xlabel("Date", fontsize=14)
plt.ylabel("num_sequences", fontsize=14)
plt.show()


# # Comparing each varient spread India Vs World

# In[44]:


plt.figure(figsize=(20,15))

plt.subplot(1,2,1).set_title("Spread of Gamma variant India v/s World", fontdict= { 'fontsize': 18, 'fontweight':'bold'})
sns.lineplot(x=Gamma_ind['date'],y=Gamma_ind["num_sequences"],label="India",)
sns.lineplot(x=others_data["date"],y=others_data["num_sequences"],label="World")
plt.xticks(rotation=90)
plt.grid()
plt.xlabel("Date", fontsize=14)
plt.ylabel("num_sequences", fontsize=14)

plt.subplot(1,2,2).set_title("Spread of Others variant India v/s World", fontdict= { 'fontsize': 18, 'fontweight':'bold'})
sns.lineplot(x=others_ind['date'],y=others_ind["num_sequences"],label="India")
sns.lineplot(x=others_data["date"],y=others_data["num_sequences"],label="World")
plt.xticks(rotation=90)
plt.grid()
plt.xlabel("Date", fontsize=14)
plt.ylabel("num_sequences", fontsize=14)
plt.show()


# # Comparing each varient spread India Vs World

# In[45]:


plt.figure(figsize=(20,15))

plt.subplot(1,2,1).set_title("Spread of Omricron variant India v/s World using log scale plot", fontdict= { 'fontsize': 16, 'fontweight':'bold'})
ax = sns.lineplot(x=Omicron_ind['date'],y=Omicron_ind["num_sequences"],label="India",)
bx = sns.lineplot(x=omicron_data["date"],y=omicron_data["num_sequences"],label="World")
plt.xticks(rotation=90)
plt.grid()
ax.set(yscale='log')
plt.xlabel("Date", fontsize=14)
plt.ylabel("num_sequences", fontsize=14)

plt.subplot(1,2,2).set_title("Spread of Alpha variant India v/s World using log scale plot", fontdict= { 'fontsize': 16, 'fontweight':'bold'})
cx =sns.lineplot(x=Alpha_ind['date'],y=Alpha_ind["num_sequences"],label="India")
dx = sns.lineplot(x=alpha_data["date"],y=alpha_data["num_sequences"],label="World")

plt.xticks(rotation=90)
cx.set(yscale='log')
plt.xlabel("Date", fontsize=14)
plt.ylabel("num_sequences", fontsize=14)
plt.grid()
plt.show()


# # Comparing each varient spread India Vs World

# In[46]:


plt.figure(figsize=(20,15))

plt.subplot(1,2,1).set_title("Spread of Beta variant India v/s World using log scale plot", fontdict= { 'fontsize': 16, 'fontweight':'bold'})
ex = sns.lineplot(x=Beta_ind['date'],y=Beta_ind["num_sequences"],label="India",)
fx = sns.lineplot(x=beta_data["date"],y=beta_data["num_sequences"],label="World")
plt.xticks(rotation=90)
plt.grid()
plt.xlabel("Date", fontsize=14)
plt.ylabel("num_sequences", fontsize=14)
ex.set(yscale='log')
plt.xlabel("Date", fontsize=14)
plt.ylabel("num_sequences", fontsize=14)

plt.subplot(1,2,2).set_title("Spread of Delta variant India v/s World using log scale plot", fontdict= { 'fontsize': 16, 'fontweight':'bold'})
gx=sns.lineplot(x=Delta_ind['date'],y=Delta_ind["num_sequences"],label="India")
hx=sns.lineplot(x=delta_data["date"],y=delta_data["num_sequences"],label="World")
plt.xticks(rotation=90)
plt.grid()
gx.set(yscale='log')
plt.xlabel("Date", fontsize=14)
plt.ylabel("num_sequences", fontsize=14)
plt.show()


# # Comparing each varient spread India Vs World

# In[47]:


plt.figure(figsize=(20,15))

plt.subplot(1,2,1).set_title("Spread of Gamma variant India v/s World using log scale plot", fontdict= { 'fontsize': 16, 'fontweight':'bold'})
ix = sns.lineplot(x=Gamma_ind['date'],y=Gamma_ind["num_sequences"],label="India",)
jx = sns.lineplot(x=others_data["date"],y=others_data["num_sequences"],label="World")
plt.xticks(rotation=90)
plt.grid()
ix.set(yscale='log')
plt.xlabel("Date", fontsize=14)
plt.ylabel("num_sequences", fontsize=14)

plt.subplot(1,2,2).set_title("Spread of Others variant India v/s World using log scale plot", fontdict= { 'fontsize': 16, 'fontweight':'bold'})
kx = sns.lineplot(x=others_ind['date'],y=others_ind["num_sequences"],label="India")
lx = sns.lineplot(x=others_data["date"],y=others_data["num_sequences"],label="World")
plt.xticks(rotation=90)
plt.grid()
kx.set(yscale='log')
plt.xlabel("Date", fontsize=14)
plt.ylabel("num_sequences", fontsize=14)
plt.show()


# In[ ]:




