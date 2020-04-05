import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
#read in pkl file
df = pd.read_pickle('./lor.pkl')

#even though I feel I cleaned my dataset well, it is still important to check for null values
nulls = list(df.isnull().sum())


#groups cards by region and counts total amount of cards from each region
#this shows that there is an unequal amount cards for each region
#so classification accuracy score metric will be skewed
card_count = df.groupby(by='card_region').count() 


#create separate dataframes to analyze for each region 
ionia = df[df['card_region'] == 'ionia']
demacia = df[df['card_region'] == 'demacia']
freljord = df[df['card_region'] == 'freljord']
piltover_zaun = df[df['card_region'] == 'piltover_and_zaun']
shadow_isles = df[df['card_region'] == 'shadow_isles']
noxus = df[df['card_region'] == 'noxus']

sns.set()


# fig = plt.figure(figsize=(10,10))
# fig.suptitle('Card Type By Region')
# plt.subplot(231)
# sns.countplot(x='card_type',data=ionia,order=['unit','spell','ability','trap'])
# plt.title('ionia')
# plt.subplot(232)
# sns.countplot(x='card_type',data=demacia,order=['unit','spell','ability','trap'])
# plt.ylabel(None)
# plt.title('demacia')
# plt.subplot(233)
# sns.countplot(x='card_type',data=freljord,order=['unit','spell','ability','trap'])
# plt.title('freljord')
# plt.ylabel(None)
# plt.subplot(234)
# sns.countplot(x='card_type',data=piltover_zaun,order=['unit','spell','ability','trap'])
# plt.title('piltover_zaun')
# plt.subplot(235)
# sns.countplot(x='card_type',data=shadow_isles,order=['unit','spell','ability','trap'])
# plt.ylabel(None)
# plt.title('shadow_isles')
# plt.subplot(236)
# sns.countplot(x='card_type',data=noxus,order=['unit','spell','ability','trap'])
# plt.ylabel(None)
# plt.title('noxus')
# fig.savefig('./images/card_type_by_region')


def occurrence_count(df):
    ''' This function takes in the word dataframes from each region, counts the total card count for each word and updates it to a dictionary with word:card count
        key:value pair then creates a new dataframe from this dictionary which displays the top 15 words by card count 
    '''
    card_region = list(df['card_region'].unique())
    df_dict = {}
    for column in df.columns[6::]: #iterates through column names after card_health
        series_dict = dict(df[column].value_counts()) #creates a dict from value count series
        series_dict.pop(0.0,None) #remove counts where value is zero, not interested in it right now
        df_dict.update({column:series_dict})

    new_df = pd.DataFrame(df_dict)
    new_df = new_df.T
    new_df.columns = new_df.columns.astype(str)
    new_df = new_df.fillna(value=0.0)
    new_df = new_df.sort_values(by=list(new_df.columns),ascending=False)
    new_df = new_df.assign(region=card_region*len(new_df))
    #new_df = new_df.head(15)
    new_df = new_df.reset_index()
    new_df = new_df.rename(columns={'index':'word'})
    return new_df

freljord_counts = occurrence_count(freljord)
ionia_counts = occurrence_count(ionia)
demacia_counts = occurrence_count(demacia)
piltover_zaun_counts = occurrence_count(piltover_zaun)
shadow_isles_counts = occurrence_count(shadow_isles)
noxus_counts = occurrence_count(noxus)

region_counts = [ionia_counts,demacia_counts,piltover_zaun_counts,shadow_isles_counts,noxus_counts]

def drop_labels(df):
    labels = ['when','in','to','the','and']
    index = []
    for label in labels:
        item = df.index[df['word']==label].item()
        index.append(item)
    df = df.drop(index)
    df = df.head(15)
    return df

freljord_counts_clean = drop_labels(freljord_counts)
ionia_counts_clean = drop_labels(ionia_counts)
demacia_counts_clean = drop_labels(demacia_counts)
piltover_zaun_counts_clean = drop_labels(piltover_zaun_counts)
shadow_isles_counts_clean = drop_labels(shadow_isles_counts)
noxus_counts_clean = drop_labels(noxus_counts)

# labels = ['when','in','to','the','and']
# index = []
# for label in labels:
#     item = ionia_counts.index[ionia_counts['word']==label].item()
#     index.append(item)
# print(type(index[0]))

# fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(nrows=2,ncols=3,figsize=(9,9),gridspec_kw={'wspace':1.5})
# fig.suptitle('Card Counts By Region')
# sns.barplot(y='word',x='1.0',data=freljord_counts,ax=ax1)
# ax1.set_title('freljord')
# ax1.set_xlabel(None)
# sns.barplot(y='word',x='1.0',data=ionia_counts,ax=ax2)
# ax2.set_title('ionia')
# ax2.set_xlabel(None)
# ax2.set_ylabel(None)
# sns.barplot(y='word',x='1.0',data=demacia_counts,ax=ax3)
# ax3.set_title('demacia')
# ax3.set_xlabel(None)
# ax3.set_ylabel(None)
# sns.barplot(y='word',x='1.0',data=piltover_zaun_counts,ax=ax4)
# ax4.set_title('piltover_and_zaun')
# ax4.set_xlabel(None)
# sns.barplot(y='word',x='1.0',data=noxus_counts,ax=ax5)
# ax5.set_title('noxus')
# ax5.set_xlabel(None)
# ax5.set_ylabel(None)
# sns.barplot(y='word',x='1.0',data=shadow_isles_counts,ax=ax6)
# ax6.set_title('shadow_isles')
# ax6.set_xlabel(None)
# ax6.set_ylabel(None)
# fig.savefig('./images/top_card_count_1')

# fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(nrows=2,ncols=3,figsize=(9,9),gridspec_kw={'wspace':1.5})
# fig.suptitle('Card Counts By Region 2')
# sns.barplot(y='word',x='1.0',data=freljord_counts_clean,ax=ax1)
# ax1.set_title('freljord')
# ax1.set_xlabel(None)
# sns.barplot(y='word',x='1.0',data=ionia_counts_clean,ax=ax2)
# ax2.set_title('ionia')
# ax2.set_xlabel(None)
# ax2.set_ylabel(None)
# sns.barplot(y='word',x='1.0',data=demacia_counts_clean,ax=ax3)
# ax3.set_title('demacia')
# ax3.set_xlabel(None)
# ax3.set_ylabel(None)
# sns.barplot(y='word',x='1.0',data=piltover_zaun_counts_clean,ax=ax4)
# ax4.set_title('piltover_and_zaun')
# ax4.set_xlabel(None)
# sns.barplot(y='word',x='1.0',data=noxus_counts_clean,ax=ax5)
# ax5.set_title('noxus')
# ax5.set_xlabel(None)
# ax5.set_ylabel(None)
# sns.barplot(y='word',x='1.0',data=shadow_isles_counts_clean,ax=ax6)
# ax6.set_title('shadow_isles')
# ax6.set_xlabel(None)
# ax6.set_ylabel(None)
# fig.savefig('./images/top_card_count_2')

# grid = sns.FacetGrid(df,col='card_region')
# grid = grid.map(sns.distplot, 'card_cost',bins=[0,1,2,3,4,5,6,7,8,9,10,11,12])
# grid.fig.suptitle('Card Cost Distribution By Region',y=1)
# grid.fig.subplots_adjust(top=0.8)
# grid.savefig('./images/card_cost_distribution',bbox_inches='tight')


# fig, ((ax1,ax2,ax3,ax4,ax5,ax6),(ax7,ax8,ax9,ax10,ax11,ax12)) = plt.subplots(nrows=2, ncols =6,figsize=(12,7),subplot_kw={'autoscale_on':True})
# fig.suptitle('Attack and Health Distributions By Region')
# sns.distplot(ionia['card_attack'],ax=ax1)
# ax1.set_title('Ionia')
# ax1.set_ylabel('Attack')
# ax1.set_xlabel(None)
# sns.distplot(demacia['card_attack'],ax=ax2)
# ax2.set_title('demacia')
# ax2.set_ylabel(None)
# ax2.set_xlabel(None)
# ax2.set_yticks([])
# sns.distplot(noxus['card_attack'],ax=ax3)
# ax3.set_title('noxus')
# ax3.set_ylabel(None)
# ax3.set_xlabel(None)
# ax3.set_yticks([])
# sns.distplot(piltover_zaun.card_attack[piltover_zaun['card_attack'] < 25],ax=ax4)
# ax4.set_title('piltover__and_zaun')
# ax4.set_ylabel(None)
# ax4.set_xlabel(None)
# ax4.set_yticks([])
# sns.distplot(freljord['card_attack'],ax=ax5)
# ax5.set_title('freljord')
# ax5.set_ylabel(None)
# ax5.set_xlabel(None)
# ax5.set_yticks([])
# sns.distplot(shadow_isles['card_attack'],ax=ax6)
# ax6.set_title('shadow_isles')
# ax6.set_ylabel(None)
# ax6.set_xlabel(None)
# ax6.set_yticks([])
# sns.distplot(ionia['card_health'],ax=ax7)
# ax7.set_title('Ionia')
# ax7.set_ylabel('card_health')
# ax7.set_xlabel(None)
# sns.distplot(demacia['card_health'],ax=ax8)
# ax8.set_title('demacia')
# ax8.set_ylabel(None)
# ax8.set_xlabel(None)
# ax8.set_yticks([])
# sns.distplot(noxus['card_health'],ax=ax9)
# ax9.set_title('noxus')
# ax9.set_ylabel(None)
# ax9.set_xlabel(None)
# ax9.set_yticks([])
# sns.distplot(piltover_zaun.card_attack[piltover_zaun['card_health'] < 25],ax=ax10)
# ax10.set_title('piltover_and_zaun')
# ax10.set_ylabel(None)
# ax10.set_xlabel(None)
# ax10.set_yticks([])
# sns.distplot(freljord['card_health'],ax=ax11)
# ax11.set_title('freljord')
# ax11.set_ylabel(None)
# ax11.set_xlabel(None)
# ax11.set_yticks([])
# sns.distplot(shadow_isles['card_health'],ax=ax12)
# ax12.set_title('shadow_isles')
# ax12.set_ylabel(None)
# ax12.set_xlabel(None)
# ax12.set_yticks([])
# fig.savefig('./images/attack_health_dist_by_region')

first_round = df.drop(columns=['when','in','to','the','and'])
first_round.to_csv('first_round')