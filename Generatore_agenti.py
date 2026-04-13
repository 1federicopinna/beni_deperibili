from __future__ import annotations
import random as rn
from Utilities import Prob_mass_func, gen_random_val, Cont_distrib
import collections as coll
from collections import Counter
from AGENT import Shopper, Discount_Shopper, max_sl
import bisect


class GenP:
    """ classe che genera una quantità di agenti pari a P secondo le specifiche del questionario:
        Natalya Lysova, F. S. (2024). Impact of the discount policies on the purchasing behaviour of perishable Products. Elsevier. 
        
        Procedura:
            1. Genero tutti i clienti senza definire la loro politica d'acquisto. Oppure mettendo di default "controllano la scadenza, 
                e con sconto del 100% accetterebbero shelf life di 5 giorni".
            2. A ciascuno assegno una shelf life minima accettabile, seguendo la distribuzione (discr_min_sl) dei giorni accettabili-
            3. A ciascuno setto il comportamento base: con probabilità del 60% guardano la scadenza (prob_guarda), con probabilità del 40% no (1-prob_guarda).
            4. Tra tutti i clienti si definisce chi e sensibile allo sconto:
                4.1 si assume che chi non guarda la scadenza sia sensibile, quindi 1-prob_guarda sul totale è sensibile allo sconto;
                4.2 agli agenti risultati dal punto 4.2 si aggiungo una quota di quelli che guardano la scadenza in modo da arrivare al 70% (prob_ag_sensibile) 
                    del totale degli agenti. 
                4.3 i perimetri si dividono in non_sensibili e sensibili.
            5. Per chi è stato selezionato nel punto 4.2 (sensibili), si deve definire le shelf life che accetterebbero in caso di sconto.
                5.1 Assumento che ogni agente:
                    - sia sensibile a una sola shelf life associata a una singola scontistica;
                    - sarà sensibile a un prodotto scontato con shelf life strettamente minore della shelf life minima che comprerebbe:
                       - NOTA: per gli agenti con shelf life minima accettabile 1, si assume che siano disposti a comprare il prodotto a shelf life 1
                            anche senza sconto.
                5.2 Si scelgono le shelf life accettare in caso di sconto usando la distribuzione di probabilità distri_discount_acceptance.
                5.3 Si scelgono gli sconti associati in base alla cdf cdf_tabella_sconti.

    """
    
    def __init__(self, 
                 discr_min_sl: Prob_mass_func ,
                 prob_guarda: float, # probabilità sulla popolazione totale che un cliente guardi la shelf life residua
                 prob_ag_sensibile: float, # probabilità sulla popolazione totale che un cliente sia sensibile agli sconti
                 distri_discount_acceptance: Prob_mass_func,
                 cdf_tabella_sconti: dict[float, float]
                 ):
        
        ### DATI PER PUNTI 1 e 2 ###
        self._dst = discr_min_sl
        self._pop = tuple(discr_min_sl.keys()) # la popolazione 
        self._wgh = tuple(discr_min_sl.values()) # i pesi - probabilità

        ### DATI PER PUNTO 3 e 4 ###
        self._perc_guarda_scad = prob_guarda
        self._perc_non_guarda_scad = 1 - prob_guarda
        self._perc_sensibili_che_guardano_scad = prob_ag_sensibile - self._perc_non_guarda_scad

        ### DATI PER PUNTO 5 ###
        self._distri_discount_acceptance= distri_discount_acceptance
        self.cdf_tabella_sconti=cdf_tabella_sconti
    
    def __call__(self, P: int, agente:Shopper) -> list[Shopper]:
        """ 
            Si chiamano in successione i metodi che a poco a poco costruiscono il perimetro che descrive la popolazione del questionario.
            
        """ 

        _gen_agent_msl = self.gen_agent_msl(P, agente) ### PUNTI 1 e 2 ###
        _list_agent_sensibili=self.list_agent_sensibili(_gen_agent_msl) ### PUNTO 3 e 4 ###
        _final_poulation_sensibile = self.list_agent_discount_acceptance(_list_agent_sensibili[0]) ### PUNTO 5 ###
        
        return _final_poulation_sensibile + _list_agent_sensibili[1] # restituisco la popolazione finale con i sensibili e con gli insensibili alla scontisica

    def gen_agent_msl(self, P: int, agente:Shopper) -> list[Shopper]: 

        """ Genero P agenti associando a ciascun agente una  
            shelf life minima accetabile secondo distribuzione di probabilità discreta distrib
        """ 
        batch = rn.choices(population = self._pop, weights = self._wgh, k = P) # campionamento con ripetizione
        batch = coll.Counter(batch) # es. {min_sl_1: agente_1, min_sl_2: Agente_2, ...}
        bt = []
        for min_sl in sorted(batch.keys()):
            bt += [agente.copy(_msl = min_sl, _idx=f"{agente.pref_idx}{q}") for q in range(batch[min_sl])]
        return bt
    
    def list_agent_sensibili(self, agenti_con_msl: list[Shopper])-> tuple[list, list]:

        """ A partire dalla lista degli clienti con shelf life minima accettabile (generato con gen_agent_msl) costruisco la lista degli clienti
            sensibili agli sconti. Si assume che chi non guarda la scadenza sia automaticamente sensibile agli sconti.
        """        
        sensibili: list[Shopper] = [] # lista degli agenti sensibili
        non_sensibili: list[Shopper] = [] # lista degli agenti non sensibili

        for a in agenti_con_msl:

            guarda_scadenza = rn.random() < self._perc_guarda_scad

            if not guarda_scadenza: # chi non guarda la scadenza è sensibile a prescindere e quindi lo aggiungo alal lista dei sensibili a sconti
                sensibili.append(a)

            elif rn.random() < self._perc_sensibili_che_guardano_scad/self._perc_guarda_scad: # chi guarda la scadenza ma è sensibile comunque. e.g se il 60% guarda scadenza, e il 40% no.
                # Per arrivare a al 70% di persone sensibili allo sconto (_perc_sensibili_che_guardano), serviranno il 40% che già non guarda scadenza,
                # + un 30% sul totale che arriverà dal 60% delle persone che guarda la scadenza. Cioé il 50% delle persone.
                sensibili.append(a)
            else:
                # tutti i restanti vanno aggiunti alla lista dei non sensibili agli sconti 
                non_sensibili.append(a)
    
        return sensibili, non_sensibili

    def list_agent_discount_acceptance(self, agenti_sensibili: list[Discount_Shopper]) -> list[Discount_Shopper]:

        """ A partire dalla lista degli agenti sensibili (generabile con list_agent_sensibili[0]) si definisce la Shelf life che 
            accetterebbero in caso di sconto. Si assume che:
                - ogni agente sia sensibile ad una sola shelf life con una singola scontistica.
                - Ogni agente avrà sarà sensibile ad un prodotto scontato con shelf life strettamente minore della shelf life minima che comprerebbe: 
                    - Per gli agenti con shelf life minima accettabile 1, si assume che siano disposti a comprare il prodotto a shelf life 1 anche senza sconto.

        """       
           
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
