import json
import pandas as pd
import numpy as np
import re 



infile = open('./data/set1-en_us.json',encoding='utf8') #reading in JSON file
data = json.load(infile)

#Below is a list of card values I extracted for each card from the respective key in JSON file
attributes = ['name','region','cost','attack','health','descriptionRaw','levelupDescriptionRaw','keywords','type'] 

def data_to_dict(data):
    '''This function takes in the JSON file data, iterates through each individual JSON object extracting specific
       information of each card, converts the extracted information to python dictionary, appends card dict to list,
       and finally returns the list of card dicts '''
    total_cards = []
    for index in data:
        single_card = {}
        for attribute in attributes:
            single_card.update({attribute:index[attribute]})
        total_cards.append(single_card)
    return total_cards

cards = data_to_dict(data)
df = pd.DataFrame(cards)
s = df['descriptionRaw']
s = s.convert_dtypes()
level_up = df['levelupDescriptionRaw']

def clean_series(series):
   '''This function takes in a series containing text data and cleans the data
      with using regex'''
   for i in range(0,len(series)):
       series.iat[i] = series.iat[i].lower()
       series.iat[i] = re.sub('[\.:,()]','',series.iat[i])
       series.iat[i] = re.sub('[\\r]','',series.iat[i])
       series.iat[i] = re.sub('[\\n]',' ',series.iat[i])
   return series 

clean_descript = clean_series(s)  # cleaned series of descriptionRaw
clean_level = clean_series(level_up) # cleaned series of levelupDescriptionRaw


def count_words(series):
    '''This function takes in a series of text data, converts it to a list,
    splits each value and appends it to a new array containing all individual strings
    from the series '''
    descript_words = []
    for value in list(series):
        words = value.split()
        for word in words:
            descript_words.append(word)
    return descript_words


total_words = pd.concat([clean_descript,clean_level],axis=1) #dataframe created from cleaned series

## 
#The following block of code converts description and and levelup word lists to numpy arrays
#combines the arrays and then converts it back to a list with unique word values

descript_words = count_words(clean_descript)
darray = np.array(descript_words)
level_words = count_words(clean_level)
uarray = np.array(level_words)

complete_array = np.append(darray,uarray)
complete_unique = list(np.unique(complete_array))
complete_unique.remove('an')
complete_unique.remove('a')
##

def descriptors(DataFrame,array):
    '''This function takes in a dataframe of all card descriptions/level up descriptions 
       and a list of all the unique words from those descriptions. The function then creates
       a zeros array from the length of dataframe and list, as well as a new zeroes dataframe 
       with list values for columns. Then iterates through dataframe values and adds a count
       to respective column if the value and column matches'''
    zarray = np.zeros(shape=(len(DataFrame),len(array)))
    words_df = pd.DataFrame(data=zarray,columns=array)
    for word in words_df.columns: #iterates over the column labels
        for i in DataFrame.index: #iterates over DataFrame index
            for column in DataFrame.columns: #iterates over DataFrame columns
              for string in DataFrame.at[i,column].split():
                if string == word: #checks if the individual string is equal to the column value
                    words_df.at[i,word] += 1
    return words_df

descript_count = descriptors(total_words,complete_unique)

#The block of code below creates a list of unique keywords attributed to cards
key_list = list(df['keywords'])
karray = np.array(key_list)
keywords = list(np.unique(karray))
nk = []
for keyword in keywords:
    for item in keyword:
        if not item in nk:
            nk.append(item)

def keywords(Series,array):
    '''This function does the same thing as descriptors but for a series'''
    zarray = np.zeros(shape=(len(Series),len(array)))
    words_df = pd.DataFrame(data=zarray,columns=array)
    for word in words_df.columns: #iterates over the column labels
        for i in Series.index: #iterates over DataFrame index
              for value in Series.iloc[i]:
                    if value == word: #checks if the individual string is equal to the column value
                        words_df.at[i,word] += 1
    return words_df

keyword_count = keywords(df['keywords'],nk)
def clean_series2(series):
    '''This function will be similar to clean_series but a little bit simpler, I don't want
       to change clean_series since I used that to create the count dataframe for descriptions'''
    for i in range(0,len(series)):
        series.iat[i] = series.iat[i].lower()
        series.iat[i] = re.sub('[\'!]','',series.iat[i])
        series.iat[i] = re.sub('\s','_',series.iat[i])
        series.iat[i] = re.sub('&','and',series.iat[i])
    return series

clean_name = clean_series2(df['name'])
clean_region = clean_series2(df['region'])
clean_type = clean_series2(df['type'])


#joining all of the cleaned dataframes together and doing an additional cleaning of some column names
clean_df = pd.DataFrame(data=[clean_name,clean_region,clean_type,df['cost'],df['attack'],df['health']])
clean_df = clean_df.T
clean_df = clean_df.rename(mapper={'name':'card_name','type':'card_type','region':'card_region',
                                   'cost':'card_cost','attack':'card_attack','health':'card_health'},axis=1) #renaming column labels to avoid overlap when joining dataframes
clean_df = clean_df.join(descript_count)
clean_df = clean_df.join(keyword_count)
clean_df = clean_df.rename(mapper={'Barrier':'barrier_kw', 'Burst':'burst_kw', "Can't Block":'cannot_block', 'Ephemeral':'ephemeral_kw','Last Breath':'last_breath_kw', 
                                   'Challenger':'challenger_kw','Elusive':'elusive_kw','Regeneration':'regeneration_kw','Double Attack':'double_attack_kw',
                                   'Imbue':'imbue_kw', 'Tough':'tough_kw','Lifesteal':'lifesteal_kw','Fast':'fast_kw','Fearsome':'fearsome_kw','Overwhelm':'overwhelm_kw',
                                   'Quick Attack':'quick_attack_kw','Skill':'skill_kw','Slow':'slow_kw','Fleeting':'fleeting_kw','Trap':'trap_kw'}, axis=1) #had to come back and edit some names to avoid duplicate column names from description/keyword conflict
clean_df.columns = map(str.lower,clean_df.columns) 
clean_df = clean_df.rename(mapper={'+0|+2':'plus_zero_plus_two','+0|+3':'plus_zero_plus_three','+1|+0':'plus_one_plus_zero',
                                   '+1|+1':'plus_one_plus_one','+2|+0':'plus_two_plus_zero','+2|+2':'plus_two_plus_two','+3|+0':'plus_three_plus_zero',
                                   '+3|+3':'plus_three_plus_three','+4|+0':'plus_four_plus_zero','+4|+4':'plus_four_plus_four','+8|+4':'plus_eight_plus_four',
                                   '-1|-0':'minus_one_minus_zero','0':'zero','1':'one_descript','10':'ten','12+':'twelve_plus','15':'fifteen','15+':'fifteen_plus',
                                   '1|1':'one_one','2':'two_descript','20':'twenty','2|5':'two_five','3':'three','3+':'three_plus','4':'four','4+':'four_plus', 
                                   '5':'five','5+':'five_plus','5|2':'five_two','6':'six','6+':'six_plus','7':'seven','7+':'seven_plus','8+':'eight_plus',
                                   "can't block":'cannot_block','last breath':'last_breath','double attack':'double_attack','quick attack':'quick_attack'}, axis=1)

# #finally serializing dataset into a pickle file
clean_df.to_pickle('lor.pkl')
