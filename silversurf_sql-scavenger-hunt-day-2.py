#!/usr/bin/env python
# coding: utf-8



# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")




# query to pass to 
query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """




# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)




popular_stories.head()




hacker_news.head('full')




# query
q_types_of_stories = '''
    SELECT
        type type,
        COUNT(id) count
    FROM `bigquery-public-data.hacker_news.full`
    GROUP BY type
'''
hacker_news.estimate_query_size(q_types_of_stories)




hacker_news.query_to_pandas(q_types_of_stories)




q_deleted_comments = '''
    SELECT count(*) deleted_comments
    FROM `bigquery-public-data.hacker_news.comments`
    WHERE deleted=True
'''
hacker_news.estimate_query_size(q_deleted_comments)




hacker_news.query_to_pandas(q_deleted_comments)




q2_deleted_comments ='''
    SELECT COUNTIF(deleted) deleted_comments
    FROM `bigquery-public-data.hacker_news.comments`
'''
hacker_news.estimate_query_size(q2_deleted_comments)




hacker_news.query_to_pandas(q2_deleted_comments)




# **Optional extra credit**: read about [aggregate functions other than COUNT()](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions) and modify one of the queries you wrote above to use a different aggregate function.

