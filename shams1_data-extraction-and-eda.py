#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd

with open('../input/hashcode-drone-delivery/busy_day.in') as file:
    line_list = file.read().splitlines()




weights = line_list[2].split()
products_df = pd.DataFrame({'weight': weights})

wh_count = int(line_list[3])
wh_endline = (wh_count*2)+4

wh_invs = line_list[5:wh_endline+1:2]
for i, wh_inv in enumerate(wh_invs):
    products_df[f'wh{i}_inv'] = wh_inv.split()

products_df = products_df.astype(int)
products_df




wh_locs = line_list[4:wh_endline:2]
wh_rows = [wl.split()[0] for wl in wh_locs]
wh_cols = [wl.split()[1] for wl in wh_locs]

warehouse_df = pd.DataFrame({'wh_row': wh_rows, 'wh_col': wh_cols}).astype(np.uint16)
warehouse_df




order_locs = line_list[wh_endline+1::3]
o_rows = [ol.split()[0] for ol in order_locs]
o_cols = [ol.split()[1] for ol in order_locs]

orders_df = pd.DataFrame({'row': o_rows, 'col': o_cols})

orders_df[orders_df.duplicated(keep=False)].sort_values('row')

orders_df['product_count'] = line_list[wh_endline+2::3]

order_array = np.zeros((len(orders_df), len(products_df)), dtype=np.uint16)
orders = line_list[wh_endline+3::3]
for i,ord in enumerate(orders):
    products = [int(prod) for prod in ord.split()]
    order_array[i, products] = 1

df = pd.DataFrame(data=order_array, columns=['p_'+ str(i) for i in range(400)], 
                    index=orders_df.index)

orders_df = orders_df.astype(np.uint16).join(df)
orders_df




chart_opts = {'width': 500,
              'xlabel': "Total Demand",
              'ylabel': "Count of Products"}

import holoviews as hv
from holoviews import dim, opts
hv.extension('bokeh')

counts = orders_df.product_count                   .value_counts()                   .sort_index()                   .reset_index()
hv.Bars(counts).opts(**chart_opts)




supply = products_df.drop(columns='weight').sum(axis=1)
supply

demand = orders_df.loc[:, orders_df.columns.str.contains("p_")].sum()
demand

surplus = supply.to_numpy() - demand.to_numpy()
print(np.amin(surplus))


freqs, edges = np.histogram(surplus, 20)
hv.Histogram((edges, freqs)).opts(width=600, xlabel="surplus")




chart_opts = {'width': 500,
              'xlabel': "Warehouse",
              'ylabel': "Total Inventory",
              'yticks': list(range(0,1801,200))}


total_prods = products_df.loc[:, products_df.columns.str.contains("wh")].sum()
hv.Bars(total_prods.value_counts().index).opts(**chart_opts)




hv.Distribution(products_df.weight).opts(width=500)




chart_opts = dict(width=600, height=400, alpha=0.7)

customers = hv.Points(orders_df, kdims = ['col', 'row']).opts(**chart_opts)
warehouses = hv.Points(warehouse_df, kdims = ['wh_col', 'wh_row']).opts(size=8, **chart_opts)
customers * warehouses




inventory_array = np.zeros((400, 600, 400), dtype=np.uint16)

wh = warehouse_df.to_numpy()
inv = products_df.drop(columns='weight').T.to_numpy()
inventory_array[wh[:, 0], wh[:, 1]] = inv

inventory_array.sum()




print(inventory_array[182,193,1], 
    np.array_equal(inventory_array.sum(axis=(0, 1)), inv.sum(axis=0)))

