# This is a sample Python script.

import os
import pandas as pd

from datetime import datetime
import time
import timeit
import copulas

from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.express as px

from sqlalchemy import create_engine
import sqlalchemy

import plotly.express as px

database_connect_url = 'postgresql://hm:hm@localhost:5432/hm'

app = Dash(__name__)

# Web page is published at http://127.0.0.1:8050/
def  create_page(app, fig):
    app.layout = html.Div(children=[
        html.H1(children='XECME'),

        html.Div(children='''
            H&M Kaggle competition in Dash.
        '''),

        dcc.Graph(
            id='example-graph',
            figure=fig
        )
    ])

#start_time = time.time()
#"""Some Code"""
#end_time = time.time()
#print(f"The execution time is: {end_time-start_time}")
#time = timeit.timeit(stmt = stmt_code, setup = setup_code, number = iterations)

def create_db(db_connect: str, articles: pd.DataFrame, customers: pd.DataFrame, training: pd.DataFrame, sample_submission: pd.DataFrame) -> sqlalchemy.engine.Engine:
    '''
    :param db_connect: Access string to database
    :param articles: Dataframe for articles data
    :param customers: Dataframe for customers data
    :param training: Dataframe for training data
    :param sample_submission: Dataframe for sample_submission data
    :return: database engine
    '''

    # Create database
    engine = create_engine(db_connect)

    # Create tables
    articles.to_sql("articles", engine)
    customers.to_sql("customers", engine)
    training.to_sql("training", engine)
    sample_submission.to_sql("sample_submission", engine)
    return engine

if __name__ == '__main__':
    # Data reduction techniques from here: https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/308635
    articles = pd.read_csv('./data/articles.csv')
    articles.columns = [c.lower() for c in articles.columns] #postgres doesn't like capitals or spaces
    articles.set_index('article_id')

    customers = pd.read_csv('./data/customers.csv')
    customers.columns = [c.lower() for c in customers.columns] #postgres doesn't like capitals or spaces
    customers.set_index('customer_id')

    train = pd.read_csv('./data/transactions_train.csv')
    train.columns = [c.lower() for c in train.columns] #postgres doesn't like capitals or spaces
    train['customer_id'].apply(lambda x: int(x[-16:],16) ).astype('int64')
    train['article_id'] = train['article_id'].astype('int32')
    train.t_dat = pd.to_datetime(train.t_dat)

    train['year'] = (train.t_dat.dt.year-2000).astype('int8')
    train['month'] = (train.t_dat.dt.month).astype('int8')
    train['day'] = (train.t_dat.dt.day).astype('int8')
    del train['t_dat']

    train['price'] = train['price'].astype('float16')
    train['sales_channel_id'] = train['sales_channel_id'].astype('int8')
    # trainarticles= train.join(articles.set_index('article_id'), on='article_id')
    # all=trainarticles.join(customers.set_index('customer_id'), on='customer_id')

    # columns
    # ['article_id', 'product_code', 'prod_name', 'product_type_no', 'product_type_name', 'product_group_name', 'graphical_appearance_no', 'graphical_appearance_name', 'colour_group_code', 'colour_group_name', 'perceived_colour_value_id', 'perceived_colour_value_name', 'perceived_colour_master_id', 'perceived_colour_master_name', 'department_no', 'department_name', 'index_code', 'index_name', 'index_group_no', 'index_group_name', 'section_no', 'section_name', 'garment_group_no', 'garment_group_name', 'detail_desc']
    print(list(articles))

    # ['customer_id', 'fn', 'active', 'club_member_status', 'fashion_news_frequency', 'age', 'postal_code']
    print(list(customers))

    #['customer_id', 'article_id', 'price', 'sales_channel_id', 'year', 'month', 'day']
    print(list(train))

    fig = px.bar(customers[:1000], x="age", y="fashion_news_frequency", color="age", barmode="group")
    fig = px.parallel_categories(articles[:1000])


    app.layout = html.Div(children=[
        html.H1(children='Hello Dash'),

        html.Div(children='''
            Dash: A web application framework for your data.
        '''),

        dcc.Graph(
            id='example-graph',
            figure=fig
        )
    ])

    # run dash server
    app.run_server(debug=True)

    #

    sample_submission = pd.read_csv('./data/sample_submission.csv')
    sample_submission.columns = [c.lower() for c in sample_submission.columns] #postgres doesn't like capitals or spaces

    # 1. Find last sale of article / product
    # 2. Find product total sales by number and value
    # 3. Find most popular products

#    from sqlalchemy import create_engine
#    engine = create_engine('postgresql://hm:hm@localhost:5432/hm')

#    articles.to_sql("articles", engine)
#    customers.to_sql("customers", engine)
#   train.to_sql("training", engine)
#    sample_submission.to_sql("sample_submission", engine)

    # sub = cudf.read_csv('sample_submission.csv')[['customer_id']]
    # sub['customer_id_2'] =\
    #     sub['customer_id'].str[-16:].str.hex_to_int().astype('int64')
    # sub = sub.merge(PREDS_DF.rename({'customer_id':'customer_id_2'},axis=1),\
    #     on='customer_id_2', how='left').fillna('')
    # del sub['customer_id_2']
    # sub.to_csv('submission.csv',index=False)

    # Press the green button in the gutter to run the script.



    # See PyCharm help at https://www.jetbrains.com/help/pycharm/