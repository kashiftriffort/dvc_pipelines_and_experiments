import os
import yaml
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

params = yaml.safe_load(open('params.yaml'))['prepare']
categories = params['categories']

data_path = os.path.join('data', 'prepared')
os.makedirs(data_path, exist_ok=True)

newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

def newsgroups_to_csv(split_name, newsgroups):
    df = pd.DataFrame([newsgroups.data, newsgroups.target.tolist()]).T
    df.columns = ['text', 'target']

    df_target_names = pd.DataFrame(newsgroups.target_names)
    df_target_names.columns = ['target_name']

    out = pd.merge(df, df_target_names, left_on='target', right_index=True)
    out.to_csv(os.path.join(data_path, split_name+".csv"))

newsgroups_to_csv('train', newsgroups_train)
newsgroups_to_csv('test', newsgroups_test)