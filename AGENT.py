# gli agenti che acquistano i prodotti
# %%
from __future__ import annotations
from BUYER import Buyer, Warehouse
from ITEM import Item
from Utilities import Cont_distrib, Prob_mass_func, gen_random_val, max_sl, almost_rnd_sl
from typing import Iterator, Callable, Optional, Iterable
import random as rn
import math
from scipy.stats import norm, triang, expon # possibili distirbuzioni da usare per la domanda dei consumatori
from itertools import chain
from functools import reduce

warehouse = dict[int, list[Item]]

# LEGGERA MODIFICA AI DUE ITERATORI DEFINITI IN UTILITIES 
# PER TENER CONTO DELLA SHELF LIFE MINIMA ACCETTABILE

# Max Shelf Life
def max_sl(wh:warehouse, min_sl:int = 1) -> Iterator:
    """ restituisce sempre la chiave del dizionario wh (warehouse) di valore maggiore
                    corrispondente alla shelf life massima, 
                           però, nel farlo, valori inferiori a min_sl non vengono restituiti """
    while True:
        try: 
            sl = max(wh.keys())
            if sl >= min_sl: yield sl 
            else: return 
        except:return

#Almost Random Shelf Life        
def almost_rnd_sl(wh:warehouse,*, wh_factor:int = 1, t_cut:int = 1, min_sl:int = 1):
    """ restituisce le chiavi del dizionario wh in maniera random, con preferenza 
           a quelle più alte determinate dal fattore moltiplicativo d'importanza wh_factor
              e dal valore di taglio t_cut; non vengono restituite quelle inferiori a min_sl"""
    while True:
        try:
            keys = tuple(i for i in sorted(wh.keys(), reverse = True) if i >= min_sl) # dalla più grande alla più piccola
            treshold = keys[0]//t_cut + 1
            wgh = [1 if key <= treshold else wh_factor for key in keys]
            yield rn.choices(keys, wgh, k = 1)[0]
        except:return

class Purchased:
    """Classe che serve a 
       tracciare gli acquisti fatti """
    def __init__(self):
        self.prc:dict[int, int] = {} # Purchased: Quantità per Shelf life {rsl: purchased_quantity}
        self.dis:dict[float, int] = {} # Discounted: Quantità per livello di sconto - {discount: purchased_quantity}
    
    def add(self, shelf_life:int, discount:float = 0):
        """ Aggiorna l'oggetto aggiungendo un nuovo prodotto
            con una certa shelf_life acquistato con sconto discount"""
        self.prc[shelf_life] = self.prc.get(shelf_life, 0) + 1
        if discount: self.dis[discount] = self.dis.get(discount, 0) + 1

    @property
    def n_purchased(self) -> int:
        """ totali acquistati"""
        return sum(n_item for n_item in self.prc.values())

    @property
    def n_discounted(self) -> int:
        """ totali acquistati con sconto"""
        return sum(n_item for n_item in self.dis.values())

    @property
    def avg_discount(self) -> float:
        """ Sconto medio applicato"""
        if self.dis == {}: return 0
        #return round(sum(n_item*discount for discount, n_item in self.dis.items())/(self.n_purchased + self.n_discounted),4)
        #F. inizialmente come sopra, tuttavia n_discounted è contenuto in n_purchased
        return round(
            sum(n_item * discount for discount, n_item in self.dis.items()) / self.n_purchased, 4)


    def __repr__(self) -> str:
        """ totali, scontati, sconto medio"""
        return f'({self.n_purchased},{self.n_discounted}, {self.avg_discount})'

    def __add__(self, other:Purchased) -> Purchased:
        """ Somma due oggetti di tupo Purchased, 
            utile se l'acquisto è spezzato in tranche, 
            esempio: prima acquisti scontati e poi non scontati
            """
        # somma due oggetti di tipo Purchased, 
        # crea due dizionario prc e dis (partendo da quelli dei due oggetti)
        # facendo l'unione delle coppie chiave:valore; 
        # in caso di chiave presente in entrambi i dizionario le si associa la somma dei valori
        prc = {rsl:self.prc.get(rsl, 0) + other.prc.get(rsl, 0) for rsl in set(self.prc) | set(other.prc)}
        dis = {dis:self.dis.get(dis, 0) + other.dis.get(dis, 0) for dis in set(self.dis) | set(other.dis)}
        pr = Purchased()
        pr.dis = dis
        pr.prc = prc
        return pr

class Warehouse_2(Warehouse):
    """ Mentre nella simulazione senza agenti, generavamo una domanda giornaliera che 
    poteva rimanere parzialmente insoddisfatta (e quindi il metodo take_items prevedeva
    il caso di scorta insufficiente), nella nuova simulazione faremo in modo che se l'agente
    vuole q prodotti quei q prodotti saranno sicuramente disponibili 
    Per gestire questa cosa useremo il nuovo metodo take_all_items definto
    di sotto
    """
    def take_all_items(self, q: int, t:float, iter_rsl:Iterator) -> list[Item]:
        """ Preleva esattamente q prodotti di shelf_life residua (rsl) pari 
            a quella restituite dall'iteratore iter_rsl. 
            Oss. 1 - Bisogna essere sicuri che q prodotti ci siano!!!
            Oss. 2 - Questo metodo sostituisce (senza farne l'override) take_items.
            Serve in particolare a gestire correttamente il "fill rate" e lo "stock out"; 
            dato che q sarà sempre disponibile, non si potrà mai generare uno stock out,
            mentre il fill rate dovrà aumentare. Per cui aumenteremo Fill_Rate[0] e Fill_Rate[1]
            che rappresentano, rispettivamente, il numero di ordini andati a buon fine e il numero
            totale di ordini             
        """
        batch = []
        for rsl in iter_rsl(self.wh): # le chiavi corrispondenti alle rsl da cercare
            item = self._take_single_item(rsl, t) # vecchio metodo che preleva 1 solo item
            if not item:
                raise Exception('Prodotto non trovato!')
            batch.append(item) 
            if len(batch) == q: break
        else: raise Exception('Quantità richiesta non disponibile') #
        #self.fill_rate = [val + 1 for val in self.fill_rate] # + 1 e + 1
        self.fill_rate[0] += 1  # ordini completamente soddisfatti
        self.fill_rate[1] += 1  # ordini totali
        self.update_plots(t) # on hand già modificato dal metodo take_single_item, oo non cambia

        return batch


scontistica = dict[int, float] # {shelf_life: %disount}
""""
La scontistica è definita con un dizionario che associa ad ogni shelf life
lo sconto proposto, dovrebbe essere ordinato dalla shelf life più breve a quella 
più lunga. E ovviamente lo sconto dovrebbe essere decrescente 
Es. {1:0.3, 2.:0.15, 3:0.1} --> shelf life di 1 giorno, sconto del 30%, di due giorni 15% ecc.
Es {1:0.2, 2.:0.2, 3:0.1} --> stesso sconto (20%) per shelf life di 1 o 2 giorni 
"""

def check_discount_policy(ds_p:scontistica) -> scontistica:
    """ verifica la correttezza di una scontistica, 
        se corretta la restituisce ordinata """
    # Ordiniamo dalla shelf life più breve alla più lunga e verifichiamo
    # che gli sconti siano tra 0 e 1
    # e che siano decrescenti all'aumentare della shelf life
    ds_p = {sl:ds for sl,ds in sorted(ds_p.items())} # ordiniamo per shelf life crescente
    discounts = list(ds_p.values())
    shelf_lives = list(ds_p.keys())
    if all(0 <= ds <= 1 for ds in discounts) and all(ds2 <= ds1 for ds1, ds2 in zip(discounts[:-1], discounts[1:])): 
        return ds_p
    else: raise Exception('Scontistica scorretta')
    
class Discount(Buyer):
    """ Il Buyer con le features legate alla scontistica.
        La scontistica è definita tramite il dizionario discount policy (_dp) è
        Si tiene traccia anche deii prodotti venduti in sconto nel dizionario
        discounted sales (_ds), {discount:n_sales} es. {0.3:10, 0.2:20, ...}
    """
    
    def __init__(self, *args, 
                 discount_policy:scontistica = {}, #
                 **kargs):
        """ In init dovremo passare un oggetto di tipo Warehouse_2
            Oss. ho usato *args **kargs per non dover cambiamo la firma della classe padre"""
        super().__init__(*args, **kargs)

        # i seguenti sono attributi protetti, esplicitati solo per chiarezza,
        # non servirebbe esplicitarli, dato che vengono generati dalla property
        
        if not discount_policy:
            self._dp: scontistica = {} # discount_policy 
            self._ds:dict[float,int] = {} # discuounted sales {discount: n_sales}
        else: self.dis_policy = discount_policy #setto la policy
    
    @property
    def dis_policy(self):
        return self._dp.copy()
    
    @property
    def dis_sales(self):
        return self._ds.copy()

    @dis_policy.setter
    def dis_policy(self, discount_policy:Optional[scontistica] = None):
        """ inserisce scontistica e dizionario con prodotti venduti
        per ogni classe di sconto"""
        if not discount_policy: 
            self._dp = {}
            self._ds = {}
        else:
            self._dp = check_discount_policy(discount_policy)
            self._ds = {sl:0 for sl in self._dp.values()} # per ogni sconto zero prodotti venduti {0.2:0, 0.1:0, ...} 

    def available(self, min_sl:int = 1, acc_sl:Optional[Iterable[int]] = None) -> int:
        """ numero di prodotti  disponibili con una shelf life
            superiore alla shelf life minima richiesta dal cliente (msl)
            o con shelf life pari a una di quelle in acc_sl (acceptalbe shelf life)
        """
        if acc_sl is None: acc_sl = tuple()
        return sum([len(item) for key, item in self.wh.wh.items() if key >= min_sl or key in acc_sl])

    def available2(self, acc_sl:Iterable[int]) -> int:
        """ numero di prodotti  disponibili con shelf life pari a una di quelle in acc_sl (acceptalbe shelf life)
        """
        return sum([len(item) for key, item in self.wh.wh.items() if key in acc_sl])


    # METODO CHE SOSTITUISCE gen_daily_demand
    def sale_items(self, 
                   q:int, # la quantità che il cliente acquisterà, se zero, mancata vendita
                   iter_rsl:Iterator # l'iteratore che restituisce le shelf life che vanno bene al cliente
                   ) -> None|Purchased:
        
        """ preleva la quantità richiesta q in base ai valori 
        di shelf life restituiti dall'iteratore iter_rsl e aggiorna il magazzino;
        restituisce un oggetto purchased che sintetizza l'acquisto fatto
        
        Oss (1). Prima di lanciare sale_items bisogna verificare la disponibilitò, questo
        lo fa l'agente shopper

        Oss (2). Qui aggiorniamo anche fill rate e stock_out, gli aggiornamenti fatti
        nel metodo wh.take_items diventano ininfluenti, perchè chiameremo quel metodo
        solo nella sicurezza di trovare la quantità richiesta
        """
        if q == 0: # mancata vendita!
           # self.b.wh.fill_rate[1] += 1 # si registra il nuvo ordine, senza aumentare fill_rate[0], il fill rate diminuisce            return None
           self.wh.fill_rate[1] += 1
           return
        t_now = round(self.env.now, 4)
        batch =  self.wh.take_all_items(q, t_now, iter_rsl) # raccoglie i prodotti e provvede a aggiornare oh stock out, ecc.
        if len(batch) != q: raise Exception('Qualcosa è andato storto') 
         # in questo caso dato che approssimiamo alla quarta decimale potrebbero esserci prelievi nello stesso istante
        self.products['delivered'].setdefault(t_now, []).extend(batch) # li mette tra i prodotti consegnati
        # guarda se e quanti prodotti in sconto sono stati venduti
        sold = Purchased()
        for item in batch:
            rsl = math.floor(item.rsl)
            discount = self._dp.get(rsl, 0) # cerca nella scontistica lo sconto legato alla shelf life rsl
            sold.add(rsl, discount) # aggiorna la vendita al cliente 
            if discount: self._ds[discount] += 1 # aggiorna il dizionario con il numero di prodotti scontati venduti 
        
        """ POSSIBILE ORDINE STRAORDINARIO """
        if self.wh.ip <= self.pol.s: self._extra_order(round(t_now,4)) 
        
        """ Aggiornamento Stock Out """
        if self.wh.oh == 0 and not self.wh._stout: #l'on hand è sceso a zero, e prima non era in stock out (quest'ultima cosa in questa nuova configurazione non dovrebbe succedere)
            self.wh.n_stock_out += 1
            self.wh._stout = True
            print(f"[{int(self.env.now)}] STOCK-OUT: oh=0, n_stockout={self.wh.n_stock_out}") #F. per verificare quando va in stockout
        return sold


    @property
    def tot_revenue(self) -> float:
        """ Il fatturato considera gli sconti
        """
        # fatturato classe padre
        revenue_teorica = super().tot_revenue

        # sconti
        sconto_perso = 0
        for discount, n_venduti in self._ds.items():
            # (Prezzo pieno * Sconto) * Numero di prodotti venduti a quello sconto
            sconto_perso += (self.cst.pr * discount) * n_venduti

        # fatturato reale considerando le vendite con sconti
        return revenue_teorica - sconto_perso


class Shopper:
    """ l'acquirente di base, insensibile allo sconto.
        la sua logica d'acquisto non cambia neppure in presenza di scontistica
    """
    def __init__(self, env,# l'ambiente di simpy
                 Idx:str, # identificativo alfa_numerico dell'agente
                 Buyer: Discount, #il punto di vendita da cui si rifornisce
                 demand: Cont_distrib|Prob_mass_func, # la quantità richiesta
                 dt:float, # intertempo tra due ordini successivi in giorni  
                 min_sl:int = 1, # minima residual shelf life accettabile
                 min_q:float = 0.5, # minima disponibilità percentuale, es. se non c'è almeno il 50% di ciò che volevo non compro
                 behaviour: Callable[[...], Iterator] = max_sl # l'iteratore che definisce le shelf life cercate
                 ):
        self.env = env
        self.idx = Idx
        self.b = Buyer # il negozio dove acqusita la merce
        self.d = demand # probabilità discreta, o continua (es. normale) per la generazione del numero di prodotti da acquistare
        self.dt_mean = float(dt) # F. aggiunta per tenere traccia del dato float dell'intertempo
        self.dt = expon(scale = dt) # intertenmpo esponenziale tra due acquisti, con media dt (lambda = 1/dt) ... per ora questo l'ho fissato
        self.msl = min_sl
        self.mq = min_q
        self.bh = behaviour
        self.prc:dict[float, Purchased] = {} # il dizionario che tiene traccia degli acquisti fatti istante_acquisto -> lista d'acquisto
        self.pref_idx: str= "S_"

    def p_quantity(self, acc_sl:Optional[Iterator[int]] = None) -> int:
        """ restituisce la quantità acquistabile """
        q = gen_random_val(self.d) # la quantità da comprare
        q_min = math.floor(q*self.mq) # la quantità minima disponibile per cui il cliente fa un acquisto
        q_av = self.b.available(min_sl = self.msl, acc_sl = acc_sl) # la quantità disponibile, nel rispetto delle shelf life ammissibili
        if q_av < q_min: return 0
        return min(q, q_av)

    def buy(self, print_log: bool = False) -> None:
        """ Acquisto """
        q = self.p_quantity()
        q = int(round(q)) # F. faccio il round a intero

        purchase = self.b.sale_items(q = q, iter_rsl = self.bh) # qua purchase è il singolo oggetto
        if purchase is not None: # se q == 0 allora purchase è None
            t_now = round(self.env.now, 4)
            self.prc[t_now] = purchase # aggiorniamo gli acquisti fatti
            dettagli = vars(purchase) if hasattr(purchase, '__dict__') else str(purchase)
            #print(f" [{int(self.env.now)}] Domanda agente {self.idx} - richiesti {q} pezzi: acquistato -> {purchase}")

        if print_log:
            self.log_buy_agente( self.env, self.idx, q, purchase)

    def buy_process(self):
        while True:
            dt = self.dt.ppf(rn.random()) # inversa della cumulata calcolata in u -> dà l'intertempo tra deu acquisti
            yield self.env.timeout(dt)
            self.buy()

    def __repr__(self) -> str:
        return f"Agente base id: {self.idx}"

    def log_buy_agente(self, env, idx, q, purchase: Purchased, gestione_None=True) -> None:
        """ Metodo che logga sulla console l'acquisto dell'agente.
            NOTA: potrebbe succedere che l'agente abbia q = 0 questo perché nel metodo p_quantity()
                della classe Discount, probabilmente la domanda q era maggiore di q_av. Conseguentemente
                purchase è None. Per permettere il print anche degli agenti che non riescono ad acquistare

        """

        if purchase is None:
            if gestione_None:
                print(
                    f"  [{int(env.now)}] Domanda agente {idx} - Richiesto e acquistato: {q} pezzi. "
                    f" tentativo di acquisto superiore alla quantità disponibile"
                )
        else:
            print(
                f"  [{int(env.now)}] Domanda agente {idx}: Richiesti {q} pz, acquistati: {purchase.n_purchased},"
                f" {purchase.n_discounted} in sconto, media sconto:{purchase.avg_discount}"
            )


    def copy(self, _msl:Optional[int] = None, _idx: Optional[str] = None) -> Shopper:
            """ metodo che crea una copia di Shopper con una diversa shelf life minima accettata"""
            if _msl is None: _msl = self.msl
            if _idx is None: _idx = self.idx
           
            return Shopper(env=self.env, Idx=_idx, Buyer=self.d, demand=self.d, dt=self.dt, min_sl=_msl, min_q= self.mq, behaviour=self.bh)


# DUE NUOVI ITERATORI CHE POSSONO ESSERE USATI DAL DISCOUNT ATTRACTED SHOPPER
# Random Discounted Shelf Life
def rnd_disc_sl(wh:warehouse, sls:Iterable[int]) -> Iterator:
    """ restistuisce le chiavi del dizionario wh in maniera random pura, 
            ma limitandosi alle solo chiavi sls (shelf life) corrispondenti  
                 ad una scontistica d'interesse per lo shopper """
    while True:
        try:
            keys = tuple(i for i in sls if i in wh.keys()) 
            yield rn.choice(keys)
        except:return

# Max Discounted Shelf Life
def max_disc_sl(wh:warehouse, sls:Iterable[int]) -> Iterator:
    """ restistuisce sempre la chiave più grande del dizionario wh, 
            ma limitandosi alle solo chiavi sls (shelf life) corrispondenti  
                 ad una scontistica d'interesse per lo shopper """
    while True:
        try:
            keys = max(i for i in sls if i in wh.keys()) 
            yield rn.choice(keys)
        except:return

def min_disc_sl(wh, sls):
    while True:
        try:
            yield min(i for i in sls if i in wh.keys())
        except:
            return

class Discount_Shopper(Shopper):
    """ In questo caso l'acquirente ha sempre una minimal_shelf_life accettabile,
        ma in caso di sconto può decidere di acquistare anche prodotti con shelf life
        inferiore. 

        Questo è definito tramite il dizionario da (discount acceptable) che definisce le scontistiche
        accettabili. es. {1:0.3, 2:0.2} con uno sconto del 30% l'acquirente accetta anche shelf life di un giorno.
        con 20% prodotto con ancora 2 giorni prima della scadenza.

        La scelta dei prodotti scontati parte sempre da quelli a sconto maggiore, a parità di sconto
        la scelta tra le shelf life dipende dall'iteratore d_bh (discount behaviour) che definisce
        proprio l'ordine di shelf life. 
        Esempio Il venditore offre sconto del 50% per shelf life = 1 e del 20% per shelf life di 2 e di 3.
        Allora l'acquirente prima cercherà prodotti con shelf life 1, e poi quelli con shelf life di 2 e di 3
        giorni. Con che ordine? Di default mettiamo una scelta randomica, ma si potrebbe pensare
        prima quelli a shelf life maggiore e poi quelli a shelf life minore, come fatto in precedenza
        """
        
    def __init__(self, *args, # stessi argomenti di prima
                 discount_acceptance:dict[int,float] = {1:0.30, 2:0.2}, # dizionario che dice come cambia la shelf life minima accettata in funzione dello sconto
                 discount_behaviour:Iterator = rnd_disc_sl # relativamente ai prodotti scontati quali sceglie? Di default puramente random
                 ):
        super().__init__(*args)
        
        # per fare prima ho solo ordinato e non ho fatto controlli, e non ho usato una property
        self.da = {s:d for s,d in sorted(discount_acceptance.items())} # discount acceptable
        self.dv: dict[float, list[int]] = self._valid_rsl() # gli sconti validi della scontistica del fornitore es. {0.3:[2, 1], 0.2:[3]}
        self.d_bh:Iterator = discount_behaviour # Iteratore che definisce la scelta
        self.pref_idx: str= "DS_"
    
    def _valid_rsl(self)-> dict[float, list[int]]: 
        # restituisce gli sconti vaidi ordinati dal maggiore al minore, a parità ordinati per shelf life
        # es. {0.3:[2, 1], 0.2:[3]} con sconto del 30% shelf life 2 o 1, con 20% shelf life 3
        valid = {}
        for rsl, discount in self.b.dis_policy.items():
            for min_rsl, min_discount in self.da.items():
                if rsl >= min_rsl and discount >= min_discount:
                    valid.setdefault(discount, []).append(rsl)
                    valid[discount].sort(reverse = True) # shelf life ordinate dalla maggiore all minore
                    break
        # sconti dal maggiore al minore
        return {discount: valid[discount] for discount in sorted(valid.keys(), reverse = True)} # qua prima era: {valid[discount] for discount in sorted(valid.keys(), reverse = True)}

    def buy(self, print_log: bool = False) -> None:
        """ Acquisto """
        acc_sl = list(set(chain.from_iterable(self.dv.values()))) # le shelf life accettabili tramite sconto
        q = self.p_quantity(acc_sl = acc_sl) # quantità totale che può essere acquistata
        q = int(round(q))
        q_iniziale = q # ho aggiunto questa variabile per conservare il valore di q iniziale

        purchase_sconti: Purchased | None = None #aggiunto per log

        if not q: self.b.sale_items(q = q, iter_rsl = self.bh) # non fa nulla, serve solo per aggiornare fill rate q== 0
        else:
            t_now = round(self.env.now, 4)
            purchased = []
            # Acquisto prodotti scontati
            # acquistiamo partendo dallo sconto più alto
            for sh in self.dv.values(): # le shelf life ordinate per sconto
                q_ds = self.b.available2(sh) # quantità acquistabile con sconto
                q_take = min(q, q_ds) # passo il minimo tra q che ha richiesto e q disponibile

                def iter_discount(wh): # funzione interna per passare anche sh all' iteratore altirmenti andava in errore
                    if self.d_bh == rnd_disc_sl:
                        return rnd_disc_sl(wh, sh)
                    if self.d_bh == max_disc_sl:
                        return max_disc_sl(wh, sh)
                    return rnd_disc_sl(wh, sh)

                #purchased.append(self.b.sale_items(q = q_ds, iter_rsl = self.d_bh)) F. rimosso perché q_ds da tutte quelle disponibili
                purchased.append(self.b.sale_items(q=q_take, iter_rsl=iter_discount))
                q -= q_ds # quantità residua
                if q <= 0: break
            else:
                purchased.append(self.b.sale_items(q = q, iter_rsl = self.bh)) # Acquisto non scontato del residuo q

            purchased_valid = [p for p in purchased if p is not None]
            if purchased_valid:
                self.prc[t_now] = reduce(lambda a, b: a + b, purchased_valid) # aggiorniamo gli acquisti fatti, sommando gli acquisti parziali
                purchase_sconti = self.prc[t_now]
            else:
                purchase_sconti=None

        if print_log:
            self.log_buy_agente(self.env, self.idx, q_iniziale, purchase_sconti)


    def __repr__(self) -> str:
        return f"Cliente Discount: {self.idx} | sl min: {self.msl} | Accetta sl: {tuple(self.da.keys())[0]} se c'è sconto del: {tuple(self.da.values())[0]*100}%\n"
   


    
    def copy(self, _msl: Optional[int] = None, _idx: Optional[int] = None) -> Discount_Shopper:
        if _msl is None: _msl = self.msl
        if _idx is None: _idx = self.idx

        return Discount_Shopper(
            self.env,      
            _idx,      
            self.b,        
            self.d,        
            self.dt_mean,       
            _msl,          
            self.mq,       
            self.bh,       
            discount_acceptance=self.da, 
            discount_behaviour=self.bh   
        )


# %%
