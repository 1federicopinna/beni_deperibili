# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 16:28:14 2025

@author: zammo
"""

import simpy as sp
import random as rn
from scipy.stats import norm, triang
from Utilities import discr_cont_distrib, Theoretical_SsI_Values, Costs, Policy, CR
from ITEM import Item
from VENDOR import Vendor
from BUYER import Warehouse, Buyer
from AGENT import Warehouse_2, Discount, Shopper, Discount_Shopper

rn.seed(10) # tanto per far sì che la sequenza sia sempre la stessa 
env = sp.Environment()

# DISTRIBUZIONI 
lt = triang(c = 0.8, loc = 0.5, scale = 3) # c posizione della moda 0.5 centrale, loc estremo inferiore, scale ampiezza intervallo
cont_sl = triang(c = 0.8, loc = 5, scale = 7)  # shelf life triangolare continua da 5 a 12, il valore loc non viene considerato
disc_sl = discr_cont_distrib(cont_sl, [i for i in range(1, 100)]) # shelf life discretizzata
dd = norm(100, 10) # mu e sigma della domanda,  attenzione che potrebbe essere negativa

# COSTI
Cst = Costs(pc = 5,  # costo
            oc = 500, # ordinazione
            dc = 0.5, # smaltimento
            u = 0.05,# %per perdita vendita
            h = 0.1 # %costo possesso annuo
           )

Cr = CR(sale_price = 10, cost  = Cst)

# POLITICA
ideal_vals = Theoretical_SsI_Values(demand = dd,
                                    lead_time = lt,
                                    H = Cr.H,
                                    O = Cr.cost.oc,
                                    I = 5,
                                    safety_level = 0.95)

# ideal vals -> {'i': 3.16, 'S': 660, 's': 114}, {'I': 5, 'S': 846, 's': 116}  
Pol = Policy(s = 120, 
            S = 750, 
            I = 5, 
            m_q = 50, # lotto minimo
            m_qw = 5,  # lotto minimo in caso di reso per scadenza
            m_rsl = 3,  # residual shelf life minima
            r_double = False # con False trigger o con s o con I
            )
# ATTORI 

V = Vendor(env, 
           LT_distrib = lt, 
           SL_distrib = disc_sl, 
           product_kind = "I", 
           min_lt = 0.5)


B = Buyer(env, 
          vendor = V, 
          policy = Pol, 
          costs = Cr, 
          wh = None, # si crea automaticametne 
          init_level = 750, 
          val_init = True)

# PROCESSI
update_w = B.update_warehouse(min_q = None, 
                              min_rsl = None, 
                              n_receiv = 3 # si controlla se è arrivato qualcosa 3 volte al giorno
                               )

gen_demand  = B.gen_daily_demand(demand = dd,
                                 pr_long_sl = 0.4, # 40% delle volte si privilegiano prodotti a scadenza massima
                                 split_factor = 5, # la domanda giornaliera è suddivisa in 5 prelievi
                                 print_out = True # mostra a video la domanda generata giorno per giorno
                                 )

make_order = B.periodic_order()

for pr in [make_order, gen_demand, update_w]:
    env.process(pr)
    
env.run(until = 100)
B.wh.show_trend()
results = {'rev': B.tot_revenue, 'so': B.pr_stock_out, 'fr': B.fill_rt}
print(results)
products = B.end_products()
orders = B.end_orders()


