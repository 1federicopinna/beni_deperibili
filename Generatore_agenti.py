from __future__ import annotations
import random as rn
from Utilities import Prob_mass_func, gen_random_val, Cont_distrib
import collections as coll
from collections import Counter
from AGENT import Shopper, Discount_Shopper, max_sl
from Simulation_con_agenti import env, lt, cont_sl, disc_sl, dd, Cst, Cr, ideal_vals, Pol, V, wh, B, update_w, q_dist, dt_mean, perc_a_dis_shop, D, q_mean, N, N_discount_shopper, N_shopper
from parametri_questionario import  discr_min_sl, prob_guarda, discr_pop_sensibile_sconti, prob_ag_sensibile, distri_discount_acceptance, cdf_tabella_sconti, P
import bisect

class GenP:
    """ classe che genera una quantità di agenti pari a P
        associando a ciascun agente una shelf life minima accetabile 
        secondo distribuzione di probabilità discreta distrib"""
    
    def __init__(self, 
                 discr_min_sl: Prob_mass_func ,
                 prob_guarda: float, # probabilità sulla popolazion totale che un cliente guardi la shelf life residua 
                 prob_ag_sensibile: float, # probabilità sulla popolazione totale che un cliente sia sensibile agli sconti
                 distri_discount_acceptance: Prob_mass_func,
                 cdf_tabella_sconti: Prob_mass_func
                 ):
        
        ### PUNTI 1 e 2 ###
        self._dst = discr_min_sl
        self._pop = tuple(discr_min_sl.keys()) # la popolazione 
        self._wgh = tuple(discr_min_sl.values()) # i pesi - probabilità

        ### PUNTI 3 4 5 ###
        self._perc_guarda_scad = prob_guarda
        self._perc_non_guarda_scad = 1 - prob_guarda
        self._perc_sensibili_che_guardano_scad = prob_ag_sensibile - self._perc_non_guarda_scad

        ### PUNTO 6 7 ###
        self._distri_discount_acceptance= distri_discount_acceptance
        self.cdf_tabella_sconti=cdf_tabella_sconti
    
    def __call__(self, P: int, agente:Shopper) -> list[Shopper]:

        _gen_agent_msl = self.gen_agent_msl(P, agente)
        _list_agent_sensibili=self.list_agent_sensibili(_gen_agent_msl)
        final_poulation = self.list_agent_discount_acceptance(_list_agent_sensibili[0])
        return final_poulation, _list_agent_sensibili[1] # restituisco la popolazione finale con i sensibili e con gli insensibili alla scontisica

    def gen_agent_msl(self, P: int, agente:Shopper) -> list[Shopper]: 

        """ Primi due punti della procedura: genero P agenti associando a ciascun agente una  
            shelf life minima accetabile secondo distribuzione di probabilità discreta distrib
        """ 
        batch = rn.choices(population = self._pop, weights = self._wgh, k = P) # campionamento con ripetizione
        batch = coll.Counter(batch) # es. {min_sl_1: agente_1, min_sl_2: Agente_2, ...}
        bt = []
        for min_sl in sorted(batch.keys()):
            bt += [agente.copy(_msl = min_sl, _idx=f"{agente.pref_idx}{q}") for q in range(batch[min_sl])]
        return bt
    
    def list_agent_sensibili(self, agenti_con_msl: list[Shopper]):

        """ A partire dalla lista degli clienti con shelf life minima accettabile costruisco la lista degli clienti
            sensibili agli sconti. Si assume che chi non guarda la scadenza sia automaticamente sensibile agli sconti.
        """        
        sensibili: list[Shopper] = [] # lista degli agenti sensibili
        non_sensibili: list[Shopper] = [] # lista degli agenti non sensibili

        for a in agenti_con_msl:

            guarda_scadenza = rn.random() < self._perc_guarda_scad

            if not guarda_scadenza: # chi non guarda la scadenza è sensibile a prescindere
                sensibili.append(a)

            elif rn.random() < self._perc_sensibili_che_guardano_scad/self._perc_guarda_scad: # chi guarda la scadenza ma è sensibile comunque. e.g se il 60% guarda scadenza, e il 40% no.
                # Per arrivare a al 70% di persone sensibili allo sconto (_perc_sensibili_che_guardano), serviranno il 40% che già non guarda scadenza,
                # + un 30% sul totale che arriverà dal 60% delle persone che guarda la scadenza. Cioé il 50% delle persone.
                sensibili.append(a)
            else:
                # 
                non_sensibili.append(a)
    
        return sensibili, non_sensibili

    def list_agent_discount_acceptance(self, agenti_sensibili: list[Discount_Shopper]) -> list[Discount_Shopper]:
           
        for agente in agenti_sensibili:

            if agente.msl == 1: # se la shelf life minima che accetta il cliente è 1 allora, la compra indipendetemente dallo sconto
                agente.da = {1: 0.0}
            else:
                _pop_sl_ammissibili: dict[int, float] = {}

                for k, v in dict(self._distri_discount_acceptance).items():
                    if k < agente.msl: # strettamente minore
                        _pop_sl_ammissibili[k] = v

                _sl_agent: int = rn.choices(population = tuple(_pop_sl_ammissibili.keys()), weights = tuple(_pop_sl_ammissibili.values()), k = 1)[0]
                u = rn.random()
                id_sconto_agent =  bisect.bisect_left(tuple(self.cdf_tabella_sconti.values()), u) -1   # bisect_left restituisce il primo valore >= u
                sconto_scelto = tuple(self.cdf_tabella_sconti.keys())[id_sconto_agent]
                agente.da = {_sl_agent: sconto_scelto}

        return agenti_sensibili
                




        






def genbatch(P:int, gq: GenP) -> list[Shopper]:
    """ Genera un batch di dimensione P """
    agente = Discount_Shopper(
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
    return gq(P, agente)


generatore = GenP(discr_min_sl, prob_guarda, prob_ag_sensibile, distri_discount_acceptance, cdf_tabella_sconti )
agenti = genbatch(P=P, gq=generatore)

print(agenti[0])



"""
Esce qualcosa del genere:
Distribuzione Shelf Life:
Shelf Life 1: 109 agenti
Shelf Life 2: 172 agenti
Shelf Life 3: 468 agenti
Shelf Life 4: 415 agenti
Shelf Life 5: 1936 agenti

conteggio = Counter(a.msl for a in agenti)

print("Distribuzione Shelf Life:")
for sl, qta in sorted(conteggio.items()):
    print(f"Shelf Life {sl}: {qta} agenti")
"""

