#!/usr/bin/env python
# coding: utf-8



# Set up feedack system
from learntools.core import binder
binder.bind(globals())
from learntools.sql.ex1 import *
print("Setup Complete")




from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "chicago_crime" dataset
dataset_ref = client.dataset("chicago_crime", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)




# Write the code you need here to figure out the answer
tables = list(client.list_tables(dataset))  #get a list of the tables in the dataset
num_tables = 0  #create a variable to store the number of tables 

for table in tables:   #create a loop that 1) goes through each tables and 
    num_tables += 1    #2) adds 1 to the total number of tables  
print(num_tables)      #get a readout of the total number of tables. 
    




num_tables  

q_1.check()




#q_1.hint()
#q_1.solution()




# Write the code to figure out the answer
for table in tables:
    print(table.table_id) #find out the name of the table, this could have been done earlier 




table_ref = dataset.table("crime") #refer to this table from now on
table = client.get_table(table_ref)
table.schema #the schema gives us information about the rows      




num_timestamp_fields = 2 # I needed to manually count the number of time data entries this time
                            #but if i knew more about the structure of SQL I'd write a loop for this too

q_2.check()




#q_2.hint()
#q_2.solution()




# Write the code here to explore the data so you can find the answer
I can already see the schema above




fields_for_plotting = ["latitude", "longitude"] # Put your answers here

q_3.check()




#q_3.hint()
#q_3.solution()




# Scratch space for your code
client.list_rows(table, max_results=5).to_dataframe()

