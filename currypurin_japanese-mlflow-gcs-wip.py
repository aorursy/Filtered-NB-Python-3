#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip install mlflow')




import mlflow
from pathlib import Path
import os
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
mlflow.__version__




experiment_name = 'first'
experiment_run_id = '001_lgb'

mlflow.set_experiment(experiment_name)




mlflow.set_tag(MLFLOW_RUN_NAME, experiment_run_id)
mlflow.log_metric('cv', 0.9)
mlflow.log_param('lgb_params', {'objective': 'regression',
                                'num_leaves': 128})




for i in Path('./mlruns/1/').glob('*'):
    if str(i).split('/')[-1] != 'meta.yaml':
        run_id = str(i).split('/')[-1]
print(run_id)




mlflow.get_run(run_id)




# Set your own project id here
PROJECT_ID = 'mlflow-sample'
# from google.cloud import bigquery
# bigquery_client = bigquery.Client(project=PROJECT_ID)
from google.cloud import storage
storage_client = storage.Client(project=PROJECT_ID)


for f in Path(f'./mlruns/1/{run_id}').glob('**/*'):
    if f.is_file():
        print(f)




EXPERIMENT_ID = 2
BUCKET_NAME = 'mlflow-sample-curry'
bucket = storage_client.get_bucket(BUCKET_NAME)
for f in Path(f'./mlruns/1/{run_id}').glob('**/*'):
    if f.is_file():
        filename = str(f)[9:]
        print(filename)
        destination_blob_name = f'mlruns/{EXPERIMENT_ID}/{filename}'
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(str(f))

