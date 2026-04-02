# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 14:34:07 2025

@author: zammo
"""

import bisect
import random as rn
import math as mt
from typing import Protocol, Iterator, NamedTuple, Optional, Callable
from itertools import accumulate
from scipy.stats import norm

# moduli miei
from ITEM import Item

""" DEFINIZIONE DI TIPI E FUNZIONI """


# definizione di distribuzione continua: ogni funzione che ha pfc, cdf e pff, probabilità
class Cont_distrib(Protocol):
    def cdf(self, x: float) -> float: ...  # cumulata

    def pdf(self, x: float) -> float: ...  # ripartizione

    def ppf(self, x: float) -> float: ...  # inversa della cumulata


# definizione di distribuzione discreta: valore (int), probabilità (float)
Prob_mass_func = dict[int, float]


class Policy(NamedTuple):
    """ i parametri della politica S, s secondo cui:
        - sia q la quantità teorica che che serve a riportare l'inventory position al livello S 
        - si ordina, a intervalli di tempo costanti I, la quantità teorica q
        - se si scende sotto il livello minimo s si riordina la quantità  teorica q, 
            * opzione 1. Verrà comunque effettuato un ordine anche a t = I
            * opzione 2. A quel punto riparte il timer e il prossimo ordine sarà dopo altre I unità temporali
        - parametri addizionali:
            * min_rsl, che rappresenta il valore minimo di shelf life accettabile per i prodotti ricevuti dal fornitore
            * m_q, il lotto minimo di riordino
    """
    s: int  # livello di sicurezza - controllo continuo
    S: int  # livello di ricostituzione
    I: int  # intervallo fisso di riordino - controllo periodico
    m_q: int  # lotto di riordino minimo - si deve riordinare per multipli di tale valore
    m_qw: int  # lotto minimo di riordino in caso di reso per shelf life
    m_rsl: int  # valore minimo tollerabile di residual shelf life, non minore di 1.5 altrimenti si rischierebbe di accettare
    # roba che verrebbe poi subito scartata (il limite teorico è 1)
    r_double: bool  # se false ordiniamo o perhcè si scende sotto s o perchè arriviamo a I non in entrambi i casi


class Costs(NamedTuple):
    """ le varie voci di costo """
    pc: float  # costo aquisto - purchase cost
    oc: float  # costo fisso di ordinazione - ordering cost
    dc: float  # costo di smaltimento - disposal cost
    u: float  # percentuale di mancato profitto --> costo mancato profitto U = u*sp
    h: float  # percentuale di costo di possesso annui --> costo mantenimento scorte H = h*sp


class CR:  # Cost and Revenues
    def __init__(self, sale_price: float, cost: Costs):
        self.cost = cost
        self.pr = sale_price
        self.H = cost.h * sale_price  # costo di possesso annuo di 1 item
        self.U = cost.u * sale_price  # costo di mancata vendita di 1 item

    def __call__(self,
                 Ns: int,  # numero vendite (sold)
                 Np: int,  # numero acquisti (purchased)
                 No: int,  # numero di ordini fatti (orders)
                 Nd: int,  # numero smaltiti (disposed off)
                 Nu: int,  # numero non vendoti (unsolded)
                 Tot_t: float,  # tempo totale cumulato a magazzino
                 *,
                 year_conv_factor: float = 365  # fattore conversione in anni, se unità è giorno allora 365
                 ):
        pc, oc, dc, *_ = self.cost
        revenue = Ns * self.pr
        cost = Np * pc + No * oc + Nd * dc + Nu * self.U + self.H * (Tot_t / year_conv_factor)
        return round(revenue - cost, 4)


def gen_random_val(pr: Cont_distrib | Prob_mass_func) -> float:
    """ Genera un numero random a partire da una distribuzione """
    u = rn.random()
    try:
        return pr.ppf(u)  # inversa della cumulata calcolata in un valore casuale tra 0 e 1
    except:
        cdf = list(accumulate(pr.values()))
        idx = bisect.bisect_left(cdf, u) - 1  # bisect_left restituisce il primo valore >= u
        return tuple(pr.keys())[idx]
        


def discr_cont_distrib(Cd: Cont_distrib, values: list[int]) -> Prob_mass_func:
    """ riceve in input una distribuzione continua 
        e la discretizza sui valori passati in input 
        es. al valore i viene associato il valore F(i) - F(i-1) 
        ovviamente approssima un po'verso sinistra... """
    v = sorted(values)
    f_val, l_val = v[0], v[-1]
    # valori cumulati calcolati in corrispondenza dei valori
    pr = {f_val: Cd.cdf(f_val)}
    pr.update({y: Cd.cdf(y) - Cd.cdf(x) for x, y in zip(v[:-2], v[1:-1])})
    pr[l_val] = 1 - sum(pr.values())  # per far somma 1
    return {key: val for key, val in pr.items() if val > 0}


def Theoretical_SsI_Values(demand: Cont_distrib,
                           lead_time: Cont_distrib,
                           H: float,  # annual holding cost per item,
                           O: float,  # ordering cost
                           I: Optional[int] = None,  # Intervallo riordino, se None viene messo a ceil(I)
                           safety_level: float = 0.95,
                           ) -> tuple[dict]:
    def foo(T: float, label: str, foo: Callable):
        """ fa i calcoli, T è il periodo di riordino I, o i """
        Var_dd_lt = var_d * (T) + mt.pow(mu_d, 2) * var_lt  # varianza nel Lead Time
        safety_factor = norm(0, 1).ppf(safety_level)
        s = mu_lt * mu_d + safety_factor * mt.sqrt(Var_dd_lt)
        S = T * mu_d + s
        return {'S': foo(S), 's': foo(s), label: round(T, 2)}

    """ restituisce s, S, I* e Q* 
        demand, lead time e holding cost devono essere espressi nella stessa unità di misura!!!
    """
    mu_lt, var_lt = lead_time.mean(), lead_time.var()
    mu_d, var_d = demand.mean(), demand.var()

    q = mt.sqrt(2 * O * mu_d / H)  # quantità ottimale
    i = q / mu_d  # intervallo teorico di riordino
    if I is None: I = mt.ceil(i)
    return foo(i, 'i', lambda x: x), foo(I, 'I', mt.ceil)


""" 
Due funzioni generatrici (Iteratori) utili per simulare il prelievo
a magazzino. 

Anticipiamo che:
    - il magazzino (classe Warehouse) avrà  un dizionario wh con chiave la shelf life e valore la lista dei prodotti con quella shelf life.
    - se prelevando si svuota una lista, la corrispondente chiave viene eliminata.
    
I due generatori iterano all'infinito sino a quando in wh c'è almeno una chiave. 
"""


def max_sl(wh: dict[int, list[Item]]) -> Iterator:
    """ restituisce sempre la chiave di valore maggiore
                    corrispondente alla shelf life massima """
    while True:
        try:
            yield max(wh.keys())
        except:
            return


def almost_rnd_sl(wh: [dict[int, list[Item]]], *, wh_factor: int = 2, t_cut: int = 2):
    """ restituisce le chiavi in maniera random, con preferenza 
           a quelle più alte determinate dal fattore moltiplicativo d'impotranza wh_factor
              e dal valore di taglio t_cut """
    while True:
        try:
            keys = tuple(sorted(wh.keys(), reverse=True))  # dalla più grande alla più piccola
            treshold = keys[0] // t_cut + 1
            wgh = [1 if key <= treshold else wh_factor for key in keys]
            yield rn.choices(keys, wgh, k=1)[0]
        except:
            return


"""  
# Esempio di utilizzo 
X = {1: [1,1,1], 2:[2,2], 3:[3,3]}
Y = {1: [1,1,1], 2:[2,2], 3:[3,3]}

def modify(x, key):
    y = x[key].pop() 
    if x[key] == []: del x[key]
    return y     

batch, batch2 = [], []
for key in max_sl(X):
    batch.append(modify(X, key))
    if len(batch) == 4:break
else: print('Stock Out')
    
for key in almost_rnd_sl(Y):
    batch2.append(modify(Y, key))
    if len(batch2) == 4:break
else: print('Stock Out')
"""
