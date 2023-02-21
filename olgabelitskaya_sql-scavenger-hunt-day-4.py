#!/usr/bin/env python
# coding: utf-8



get_ipython().run_cell_magic('html', '', '<style> \nh1, h2, h3 {text-shadow: 4px 4px 4px #aaa;} \nspan {color: black; text-shadow: 4px 4px 4px #aaa;}\ndiv.output_prompt {color: darkblue;} \ndiv.input_prompt {color: steelblue;} \n</style>')




# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")




bitcoin_blockchain.head("transactions")[:1].T




query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, month 
            ORDER BY year, month
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)




# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
ax=transactions_per_month["transactions"].plot(figsize=(14,7),
                                                 title="Monthly Bitcoin Transactions");
ax.set_ylabel('transactions')
ax.set_xlabel('months');




# Your code goes here :)
# Question 1
my_query1 = """ WITH time AS 
                (SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`)
                SELECT COUNT(transaction_id) AS transactions,
                    EXTRACT(DAY FROM trans_time) AS day,
                    EXTRACT(MONTH FROM trans_time) AS month,
                    EXTRACT(YEAR FROM trans_time) AS year
                FROM time
                GROUP BY year, month, day 
                ORDER BY year, month, day
            """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(my_query1, max_gb_scanned=21)
transactions2017_per_day = transactions_per_day[transactions_per_day['year']==2017]
transactions2017_per_day[:10]




ax=transactions2017_per_day["transactions"].plot(figsize=(14,7),
                                                 title="Daily Bitcoin Transactions in 2017")
ax.set_ylabel('transactions')
ax.set_xlabel('days');




import pandas
transactions_per_day2= pandas.DataFrame(index=range(1,366,1), columns=range(2010,2018,1))
transactions_per_day2 = transactions_per_day2.fillna(0)

for i in range(2010,2018,1):
    transactions_per_day2.loc[:,i]=    transactions_per_day[transactions_per_day['year']==i]    ["transactions"][:365].as_matrix()

transactions_per_day2.head()




import seaborn
transactions_per_day2.plot(figsize=(14,7), stacked=False, 
                           title="Daily Bitcoin Transactions by Years",
                           color=seaborn.color_palette("tab10",8))
ax.set_ylabel('transactions')
ax.set_xlabel('days');




# Question 2
my_query2 = """SELECT COUNT(transaction_id) AS transactions, merkle_root
               FROM `bigquery-public-data.bitcoin_blockchain.transactions`
               GROUP BY merkle_root
               ORDER BY transactions DESC
            """
transactions_per_root = bitcoin_blockchain.query_to_pandas_safe(my_query2, max_gb_scanned=37)
transactions_per_root[:10]




ax=transactions_per_root["transactions"][:100].plot(figsize=(14,7),
                                                      title=\
                                                      "Bitcoin Transactions By Merkle Roots")
ax.set_ylabel('transactions')
ax.set_xlabel('roots');

