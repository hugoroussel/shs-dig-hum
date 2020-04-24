import json
import os
import bz2
import io
from bz2 import BZ2File
import pandas as pd
import numpy as np
import re


# a helper function to get the lines from am archive
def read_jsonlines(bz2_file):
    text = bz2_file.read().decode('utf-8')
    for line in text.split('\n'):
        if line != '':
            yield line
            

def extract_particular_files(archives, regex_str, mag=None):
    #Extract firstly the specific dates
    regex = re.compile(regex_str)
    file_names = [file for file in archives if regex.fullmatch(file)]
    file_names = sorted(file_names)
    #Extract for specific magazine
    if mag:
        regex_mag = re.compile(mag + '.*') 
        file_names = [file for file in file_names if regex_mag.fullmatch(file)]
        file_names = sorted(file_names)
        
    return file_names

def extract_content(file_names, files_dir):
    id_, mag_, date_, page_, text_ = [], [], [], [], []
    
    for archive in file_names:

        # take only the transformed archives
        # open the archive
        f = BZ2File(os.path.join(files_dir, archive), 'r')

        # get the list of articles it contains (= a json object on each line)
        articles = list(read_jsonlines(f))

        print(archive, ':', len(articles), 'articles à extraire')

        # load the first 100 articles as json and access their attributes    
        for a in articles:

            # decode the json string into an object (dict)
            json_article = json.loads(a)
            # print(json_article)
            if 'ft' in json_article:
                mag_.append(str(json_article["id"])[:3])
                date_.append(str(json_article["id"])[4:14])
                page_.append(str(json_article["pp"])[1:-1])
                text_.append(str(json_article["ft"]))
                
    return id_, mag_, date_, page_, text_

def measure_articles(df):
    
    lengths = []
    
    for ind, row in df.iterrows():
        lengths.append(len(row['text']))
        
    return lengths

def handle_multiple_pages(df):
    
    page, ppage = [], []
    
    for ind, row in df.iterrows():
        
        found = re.findall('([0-9]+)', row['page'])
        if len(found) > 1:
            page.append(found[0])
            ppage.append(found[1])
        else:
            page.append(row['page'])
            ppage.append(np.nan)
        
    return page, ppage

def preprocessing(filename, df):
    df['length'] = measure_articles(df)
    page, ppage = handle_multiple_pages(df)
    df['page'] = page
    df['ppage'] = ppage
    # Jeter les articles vides ou ne contenant que quelques caractères (p.ex titre des rubriques)
    df = df[df['length'] > 50]
    # Sauvegarder l'index
    #df['id'] = df.index
    # Formater les types
    #df.date = pd.to_datetime(df.date, infer_datetime_format=True)
    df.mag = df.mag.astype('category')
    df.page= df.page.astype('int')
    df.ppage = df.ppage.astype('float')
    df.text = df.text.astype('str')
    df = df[[ 'mag', 'date', 'page', 'ppage', 'text', 'length']]
    display(df.head())
    #Save filter data
    df.to_json('../data/filtered/'+ filename +'.json.bz2', compression = 'bz2')
    df_lengths = df['length'].value_counts()
    print("Length df:", df_lengths)
    return df

def run(gdl_files, jdg_files, file_name, files_dir):
    
    id_gdl, mag_gdl, date_gdl, page_gdl, text_gdl =  extract_content(gdl_files, files_dir)
    df_gdl = pd.DataFrame.from_dict(
        {
            'mag': mag_gdl,
            'date': date_gdl,
            'page': page_gdl,
            'text': text_gdl
        })
    
    id_jdg, mag_jdg, date_jdg, page_jdg, text_jdg =  extract_content(jdg_files, files_dir)
    df_jdg = pd.DataFrame.from_dict(
        {
            'mag': mag_jdg,
            'date': date_jdg,
            'page': page_jdg,
            'text': text_jdg
        })
    
    df_all = pd.concat([df_gdl, df_jdg]).reset_index()
    

    df_all = preprocessing('cleaned_all' + file_name, df_all)
    df_gdl = preprocessing('cleaned_gdl' + file_name, df_gdl)
    df_jdg = preprocessing('cleaned_jdg' + file_name, df_jdg)

    return df_all, df_gdl, df_jdg

def load_corpus(filename):
    print("===LOAD CORPUS===")
    df_cleaned = pd.read_json(filename, compression = 'bz2')
    df_cleaned.mag = df_cleaned.mag.astype('category')
    display("Initital shape: ", df_cleaned.shape)
    return df_cleaned


def keywords_filtering(df, keywords):
    print("===KEYWORDS===")
    counts, k = [], []

    for keyword in keywords:
        k.append(keyword.lower())
    for ind, row in df.iterrows():
        
        counts_ = []
        for k_ in k:
            counts_.append(len(re.findall(k_, row['text'].lower())))
        
        counts.append(counts_)
           
    counts_garbage = np.asarray(counts).T
    df['keywords'] = 0
    for i in range(len(keywords)):
        df['keyword_' + keywords[i]] = counts_garbage[i]
        df['keywords'] += counts_garbage[i]
    
    for i in range(len(keywords)):
        df = df[df['keywords'] > 1]
        
    display("After keyword filtered: ", df.shape)
    return df

def unwanted_filtering(df, keys):
    print('=====UNWANTED====')
    counts = []
    
   
    for ind, row in df.iterrows():
        
        counts_ = []
        for k in keys:
            counts_.append(len(re.findall(k, row['text'].lower())))
        counts.append(counts_)
        
   
    
    counts_garbage = np.asarray(counts).T
    df['garbage'] = 0
    for i in range(len(keys)):
   
        df.garbage += counts_garbage[i]
    df = df[df['garbage'] == 0]
        
    df.drop('garbage', axis=1, inplace=True)
    display("After keyword filtered: ", df.shape)
    return df



def creating_iramutek_file(df, filename):
    txt = open(filename, 'w+', encoding="UTF-8")
    for ind, row in df.iterrows():
        date = str(row['date'])[0:10]
        txt.write('**** *')
        txt.write(str(ind))
        txt.write(' *') 
        txt.write(str(row['page']))
        txt.write(' *')
        txt.write(row['mag'])
        txt.write(' *')
        txt.write(date)
        txt.write('\n')
        txt.write(row['text'])
        txt.write('\n')

    txt.close()



