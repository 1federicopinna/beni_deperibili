import math as mt
import collections as coll
import random as rn

#moduli miei
from Utilities import Cont_distrib, Prob_mass_func, gen_random_val
from ITEM import Item

class GenQ:
    """ classe che genera una quantità di prodotti pari a Q
        associando a ciascun prodotto una shelf life secondo 
        distribuzione di probabilità discreta distrib"""
    
    def __init__(self, 
                 distrib: Prob_mass_func):
        self._dst = distrib
        self._pop = tuple(distrib.keys()) # la popolazione 
        self._wgh = tuple(distrib.values()) # i pesi - probabilità
    
    def __call__(self, Q: int, it:Item) -> list[Item]:
        """ restituisce la lista dei prodotti generati,
                  ordinati per shelf life crescente """ 
        batch = rn.choices(population = self._pop, weights = self._wgh, k = Q) # campionamento con ripetizione
        batch = coll.Counter(batch) # es. {shelf_life_1: q1, shelf_life_2: q2, ...}
        bt = []
        # crea il batch di prodotti.
        # ordinandoli da shelf life minore a maggiore 
        for shelf_life in sorted(batch.keys()):
            bt += [it.copy(sl = shelf_life) for q in range(batch[shelf_life])]
        return bt

class Vendor:
    """ venditore mono prodotto, se volessimo gestire più prodotti
        basterebbe creare due o più venditori distinti. 
        Oss. La capacità del fornitore è considerata infinita,
                    riesce sempre a vare l'ordine entro il LT """
    
    def __init__(self, env, 
                     LT_distrib:Cont_distrib, 
                     SL_distrib:Prob_mass_func, 
                     product_kind = 'milk', 
                     min_lt:float = 0):
        self.env = env
        self.lt = LT_distrib # lead time di produzione e trasporto
        self.sl = SL_distrib # distribuzione della shelf life dei prodotti realizzati
        self.pr = product_kind # il tipo di prodotto realizzato
        self.gq = GenQ(self.sl) # la classe callable che genera il batch di prodotti
        self.mlt = min_lt # il lead time minimo default = 0
    
    def genbatch(self, Q:int) -> list[Item]:
        """ Genera un batch di dimensione Q """
        item = Item(gen_time = self.env.now, shelf_life = None, kind = self.pr)
        return self.gq(Q, item)
        
    def deliver(self, Q:int, depot:list[Item], round_to:int = 2) -> None:
        """
        Processo simpy che simula la produzione e l'invio di un batch-
            - depot è il magazzino di transito del buyer 
              in cui la merce viene depositata in attesa di 
              essere messa a magazzino """
        
        batch = self.genbatch(Q)
        lead_time = max(self.mlt, round(gen_random_val(self.lt), round_to)) # generazione random del lt
        yield self.env.timeout(lead_time)
        depot.extend(batch) # aggiungiamo il batch al deposito
    
    @property
    def avg_lt(self):
        return self.lt.mean()
    
    @property
    def std_lt(self):
        return self.lt.std()