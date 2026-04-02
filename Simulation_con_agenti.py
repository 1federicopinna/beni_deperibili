# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 16:28:14 2025

@author: zammo
"""

import simpy as sp
import random as rn
from scipy.stats import norm, triang
from Utilities import discr_cont_distrib, Theoretical_SsI_Values, Costs, Policy, CR
from VENDOR import Vendor
from AGENT import Warehouse_2, Discount, Shopper, Discount_Shopper, max_sl

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

wh = Warehouse_2(None)          # Warehouse_2 come Warehouse ma aggiunge take_all_items

B = Discount(env,
             vendor=V,
             policy=Pol,
             costs=Cr,
             wh=wh,
             init_level=750,
             val_init=True,
             discount_policy={1: 0.50, 2: 0.30, 3: 0.10} )

# PROCESSI
update_w = B.update_warehouse(min_q = None,
                              min_rsl = None, 
                              n_receiv = 3 # si controlla se è arrivato qualcosa 3 volte al giorno
                               )

### Funzione di domanda triangolare simmetrica
q_dist = triang(c=0.5, loc=1, scale=4) #funzione di domanda triangolare simmetrica intervallo della domanda [1,5]
dt_mean = 5                            # media dei giorni tra due acquisti

### Numero di agenti in modo che la domanda complessiva sia la stessa della simulazione senza agenti
perc_a_dis_shop = 0.20                 # percentuale che ho inserito io di Agenti sensibili agli sconti, 20% del totale
D = dd.mean()                          # domanda media giornaliera senza agenti (100)
q_mean = q_dist.mean()                    # media della funziona della domanda (3)
N = int(round(D * dt_mean / q_mean))   # Numero di agenti per domanda complessiva uguale alla precedente senza agenti

### Suddivisione agenti Shopper e Discount Shopper
N_discount_shopper = int(N * perc_a_dis_shop) # il numero di agenti discount shopper è una percentuale dell'N totale
N_shopper = N - N_discount_shopper

### Creazione agenti
agents = []
for i in range(N_shopper):
    a = Shopper(env=env,
                Idx=f"A_{i}",
                Buyer=B,
                demand=q_dist,
                dt=dt_mean,
                min_sl=Pol.m_rsl,   # stessa soglia della policy
                min_q=0.5,
                behaviour=max_sl
                )   # Ho lasciato il default
    agents.append(a)

for i in range(N_discount_shopper):
    b = Discount_Shopper(
        env,  # env
        f"A_Discount_{i}",
        B,
        q_dist,
        dt_mean,
        Pol.m_rsl,
        0.5,
        max_sl,
        discount_acceptance={1: 0.5, 2: 0.3}
    )
    agents.append(b)

print(f"N agents = {N}, di cui {N_discount_shopper} agenti che sfruttano gli sconti)")

agent_processes = [a.buy_process() for a in agents] # Processi degli agenti
make_order = B.periodic_order()

for pr in [make_order, update_w, *agent_processes]:
    env.process(pr)

if __name__== "main":
    env.run(until = 100)
    B.wh.show_trend()
    
    #####
    # Vari print per debug e statistiche
    #####
    results = {'rev': B.tot_revenue, 'so': B.pr_stock_out, 'fr': B.fill_rt}
    print(results)
    #print("n_stockout =", B.wh.n_stock_out)
    #print("N_cycles  =", len(B.order_types['i']) + len(B.order_types['s']))
    #print("pr_stockout calcolata =", B.pr_stock_out)
    #products = B.end_products()
    #orders = B.end_orders()


