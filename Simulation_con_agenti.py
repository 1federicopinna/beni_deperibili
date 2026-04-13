# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 16:28:14 2025

@author: zammo
"""

import simpy as sp
import random as rn
from scipy.stats import norm, triang
from Utilities import discr_cont_distrib, Theoretical_SsI_Values, Costs, Policy, CR, Prob_mass_func
from VENDOR import Vendor
from AGENT import Warehouse_2, Discount, Shopper, Discount_Shopper, max_sl

from Generatore_agenti import GenP # classe per la generazione clientela secondo il questionario
from parametri_questionario import  discr_min_sl, prob_guarda, prob_ag_sensibile, distri_discount_acceptance, cdf_tabella_sconti, P

rn.seed(10) # tanto per far sì che la sequenza sia sempre la stessa 
env = sp.Environment()

# DISTRIBUZIONI: Lead time (lt), Shelf life (disc_sl), e domanda giornaliera (dd)
lt = triang(c=0.25, loc=1, scale=4) #shelf life triangolare continua da 1 a 5, moda 2
disc_sl: Prob_mass_func = {13:0.1, 14:0.2 , 15:0.5, 16:0.2}  # shelf life discreta
dd = norm(2506, 113) # mu e sigma della domanda

# COSTI
Cst = Costs(pc = 5,  # costo
            oc = 550, # ordinazione
            dc = 0.163, # smaltimento
            u = 0.05875, # 0.47 / 8 = 0.05875 stockout cost in frazione del prezzo
            h = 1.095 # %costo possesso annuo
           )

Cr = CR(sale_price = 8, cost  = Cst)

# POLITICA
ideal_vals = Theoretical_SsI_Values(demand = dd,
                                    lead_time = lt,
                                    H = Cr.H,
                                    O = Cr.cost.oc,
                                    I = 4,
                                    safety_level = 0.95)


# valori ottimi teorici per i=4 -> S: 20230, s: 10206
# valori ottimi per BO 2D per i=4 -> S: 16417 s: 8134 fo media migliore: 501371.58875
# valori ottimi per SA per i=4 -> S: 17566, s: 8549 fo migliore: 504065.4243423543
Pol = Policy(s = 8134,
            S = 16417,
            I = 4, 
            m_q = 50, # lotto minimo
            m_qw = 5,  # lotto minimo in caso di reso per scadenza
            m_rsl = 10,  # residual shelf life minima
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
             discount_policy={1: 0.30, 2: 0.20, 3: 0.10} )

# PROCESSI
update_w = B.update_warehouse(min_q = None,
                              min_rsl = None, 
                              n_receiv = 3 # si controlla se è arrivato qualcosa 3 volte al giorno
                               )

### Funzione di domanda triangolare simmetrica
q_dist: Prob_mass_func = {1: 0.15, 2: 0.45, 3: 0.25, 4: 0.1, 5: 0.05}
dt_mean = 3                            # media dei giorni tra due acquisti


#### GENERAZIONE AGENTI ####
"""
    GENERAZIONE DELLA CLIENTELA
        - si crea un agente "prototipo" che verrà moltiplicato 
    
"""

agente_base = Discount_Shopper(
        env,  # env
        f"A_Discount_{0}",
        B,
        q_dist,
        dt_mean,
        Pol.m_rsl,
        0.5,
        max_sl,
        discount_acceptance={5:1}
    )

generatore = GenP(discr_min_sl, prob_guarda, prob_ag_sensibile, distri_discount_acceptance, cdf_tabella_sconti) 
agents = generatore(P, agente_base)

print(f"N di Agenti nella simulazione = {len(agents)}")

agent_processes = [a.buy_process() for a in agents] # Processi degli agenti
make_order = B.periodic_order()

for pr in [make_order, update_w, *agent_processes]:
    env.process(pr)

env.run(until = 100)
B.wh.show_trend()

#####
# Vari print per debug e statistiche
#####
results = {'rev': B.tot_revenue, 'so': B.pr_stock_out, 'fr': B.fill_rt}
print(results)

rev = float(results['rev'])
fr = float(results['fr'])
daily_penalty = 951.91
target_fr = 0.95

# Calcolo FO “alla SA”
fo = rev - 100 * daily_penalty * max(0, target_fr - fr)

print(f"Ricavo medio (agenti): {rev:.2f}")
print(f"Fill rate (agenti): {fr:.5f}")
print(f"Funzione obiettivo (agenti): {fo:.2f}")

#print("n_stockout =", B.wh.n_stock_out)
#print("N_cycles  =", len(B.order_types['i']) + len(B.order_types['s']))
#print("pr_stockout calcolata =", B.pr_stock_out)
#products = B.end_products()
#orders = B.end_orders()


