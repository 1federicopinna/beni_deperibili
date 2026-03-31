from __future__ import annotations
import matplotlib.pyplot as plt
import math as mt
import random as rn
from typing import Optional, Iterator, Iterable, NamedTuple
from itertools import chain
#moduli miei
from Utilities import Cont_distrib, gen_random_val, max_sl, almost_rnd_sl, Policy, CR
from ITEM import Item
from VENDOR import Vendor

class Warehouse:
    """ Classe che simula il magazzino dei prodotti.
        Il magazzino è monoprodotto, in caso di più prodotti se ne 
        possono istanziare due o più """
    
    def __init__(self, wh:Optional[dict[int, list[Item]]] = None):
        """ Se il magazzino iniziale wh non viene passato in input,
            bisognerà farlo in un momento successivo, ad esempio utilizzando il metodo init_inventory 
            Tale magazzino è un dizionario che associa a una chiave nuemerica, che rappresenta la shelf_life residua,
            una lista con tutti i prodotti per i quali, ad inizio giornata, la parte intera della loro shelf life era pari al valore della chiave
            es. 1:[pr1, pr2, pr3] sono tutti prodotti la cui shelf life a inizio giornata era tra 1 e 1.99. 
        """
        self.wh = wh 
        self.oo:int = 0  # on order
        self.oh = 0 if wh is None else self.eval_oh()   
        self._stout = False # se si è in rottura di stock allora True
        
        self.fill_rate:list[int, int] =  [0, 0] # numero di ordini di prelievo completamente eseguiti, numero totale di ordini di prelievo
        self.lost:int = 0 # numero di item persi (non vendute)
        self.n_stock_out: int = 0 # numero di rotture di stock, serve per il calcolo della probabilità di stock out
        # registra i valori per fare il plot del trend delle scorte
        self.inventory_plot:dict[str, dict[float, int]] = {'oh':{0:self.oh}, 'oo':{0:self.oo}, 'ip':{0:self.ip}}
        
    @property
    def ip(self) -> int:
        """inventory position"""
        return self.oh + self.oo
    
    def eval_oh(self, *shelf_life:int) -> int:
        """ Restituisce il totale dei prodotti con x giorni di shelf life """
        if shelf_life == (): shelf_life = tuple(self.wh.keys())
        return sum(len(self.wh[sl]) for sl in self.wh if sl in shelf_life)

    def init_inventory(self, Vnd:Vendor, level:int, min_rsl:int = 1):
         """ Genera il magazzino iniziale 
             - fa un ordine fittizio al fornitore che viene subito messo a magazzino,
             - tale ordine genera un lotto q = level, che generalmente sarà pari a S,
             - si riduce la shelf life originariamente generata del Lead Time medio
             - eventuali prodotti con shelf life residua minore del minimo (min_rsl) non vengono inseriti a magazzino
        """
         batch = Vnd.genbatch(level) 
         self.wh = {}
         lt = Vnd.avg_lt # volendo potremmo arrotondare con ceil(Vnd.avg_lt)
         for item in batch: 
             item.tt = [0, None]
             item.sl = round(item.sl - lt, 3) # riduciamola shelf life del lt 
             if item.sl >= min_rsl: self.wh.setdefault(int(item.sl), []).append(item)
         self.oh = self.eval_oh()
         self.update_oo_and_plots(q = 0, t = 0) # per creare il grafico iniziale
    
    """ METODI DI PRELIEVO E DI STOCCAGGIO """
    
    def take_items(self, q: int, t:float, iter_rsl:Optional[Iterator] = None) -> list[Item]:
        """ Prova a prelevare alcuni prodotti con rsl pari 
            a quelle restituite dall'iteratore iter_rsl. 
            Di default l'iteratore iter_rsl restituisce sempre la rsl massima """
        
        self.fill_rate[1] += 1 # un ordine nuovo registrato
        batch = []
        if iter_rsl is None: iter_rsl = max_sl # iteratore di default   
        for rsl in iter_rsl(self.wh):
            item = self._take_single_item(rsl, t)
            if item: batch.append(item) # non dovrebbe mai essere None, ma così siamo sicuri
            if len(batch) == q: 
                self.fill_rate[0] += 1 # completely fulfilled order!!! 
                break
        else: # aggiorniamo i prodotti persi ed eventualmente lo stock out
            self.lost += max(0, q - len(batch)) # prodotti non venduti
            if self.lost and not self._stout:
                self.n_stock_out += 1
                self._stout = True
        self.update_plots(t) # on hand già modificato dal metodo take_single_item, oo non cambia
        return batch
        
    def _take_single_item(self, rsl:int, t:float) -> Item|None:
        """ Preleva un solo prodotto con residual shelf life pari a rsl;
            se non presente restituisce None 
        """
        try: item = self.wh[rsl].pop()
        except (KeyError, IndexError): return None
        else: # if ok
            item.tt[1] = round(t, 4) # tempo di uscita
            self.oh = max(0, self.oh - 1)  # aggiorna on hand 
            if self.wh[rsl] == []: del self.wh[rsl] # se si svuota cancelliamo
            return item
    
    def stock_items(self, items:Iterable[Item], t_now:float, last_batch:bool = False) -> None:
        """ inserisce a magazzino un batch di prodotti 
            last batch serve a segnalare se stiamo mettendo a magazzino prodotti 
            a fine giornata. Teoricamente non servirebbe esplicitarlo, ma evita
            problemi con le approssimazioni decimali """
        q = len(items)
        for item in items: 
            self._stock_single_item(item, round(t_now,4))
        self.oh += q
        self._stout = False
        self.update_oo_and_plots(-q, round(t_now, 4)) # questo fa anche l'update dei plots
        
    def _stock_single_item(self, item:Item, t_now:float) -> None:
        """ Inserisce un prodotto a magazzino. 
            Si calcola la rsl a inizio giornata e si mette l'ordine 
            nello slot corrispondnte a floor (prev_rsl).
        """
        prev_rsl = item.rem_sl(mt.floor(t_now), in_place = True)  # la rsl a inizio giornata
        item.tt[0] = round(t_now, 4) # tempo di ingresso
        if prev_rsl < 1: raise Exception("Materiale scaduto in ingresso")
        self.wh.setdefault(mt.floor(prev_rsl), []).append(item)
    
    def update_oo_and_plots(self, q:int, t:float):
        self.oo += q
        self.update_plots(t)
    
    def update_plots(self, t:float) -> None:
        self.inventory_plot['oh'][t] = self.oh
        self.inventory_plot['oo'][t] = self.oo
        self.inventory_plot['ip'][t] = self.ip
    
    """ METODI DI UTILITA' """    
    
    def iter_products(self, *shelf_life:int) -> Iterator[Item]:
        """ Restituisce uno a uno i prodotti con una certa shelf life 
                se shelf life non passata, li restituisce tutti dalla sl maggiore alla minore"""
        if shelf_life == (): shelf_life = tuple(self.wh.keys())
        return chain.from_iterable(items for rsl, items in sorted(self.wh.items(), reverse = True) if rsl in shelf_life)
    
    def show_sorted(self) -> dict[int, list[Item]]:
        """ restituisce una copia del dizionario wh 
              con i prodotti ordinato per shelf life """
        rsls = sorted(self.wh.keys())
        return {rsl:self.wh[rsl].copy() for rsl in rsls}
    
    def show_trend(self, t_start:float = 0.0, t_end:Optional[float] = None, all_trends:bool = True, step:float = 0.2):
        """ mostra il grafico inventory nel tempo """
        
        def make_axis(label:str, t_end):
            """ aggiungiamo valori intermedi per avere tratti costanti """
            data = self.inventory_plot[label]
            if not t_end: t_end = mt.ceil(max(data.keys()))
            times = list(sorted(set(i*step for i in range(int(t_end/step) + 1)) | set(data.keys())))
            values = [0]
            for t in times:
                try: val = data[t]
                except: val = values[-1]
                values.append(val)
            return times, values[1:]
    
        t, oh = make_axis('oh', t_end)
        plt.plot(t, oh, color = "blue", label = "On Hand")
        if all_trends:
            t, oo = make_axis('oo', t_end)
            plt.plot(t, oo, color = "green", label = "On Order")
            t, ip = make_axis('ip', t_end)
            plt.plot(t, ip, color = "purple", label = "Inventory Position")
        plt.xlabel("Time")
        plt.ylabel("On Hand")
        plt.grid(True)
        plt.legend()
        plt.show()   
    
 
class Buyer:
    """ La classe che gestisce il punto di vendita, 
        per ora consideriamo un prodotto singolo, per estenderlo a più prodotti basterebbe passare:
            - una lista di venditori
            - una lista di warehouse
            - una lista di depositi 
            - una lista di trigger, ecc.  
    """
    
    def __init__(self, env, 
                     vendor:Vendor, 
                     policy:Policy, 
                     costs:CR, 
                     *, 
                     wh:Optional[dict[int, list[Item]]| Warehouse] = None,
                     init_level:int = 0,
                     val_init:bool = True):
        """ 
            - wh è il warehouse, può essere passato già pieno 
                  oppure può essere riempito automaticamente sino al livello paria a init_level che di default vale S.
            - val_init fa sì che vengano valorizzate e conteggiate nei costi anche le scorte iniziali 
        """
        self.env = env
        self.vd = vendor
        self.pol = policy
        self.cst = costs
        
        self.dp: list[Item] = [] # il deposito dove il venditore scarica la merce
        #self.wh = Warehouse(wh)
        self.wh = wh if isinstance(wh, Warehouse) else Warehouse(wh)
        #if not wh: # creiamo l'inventario iniziale con una quantità pari al massimo ad S (se tutti i prodotti hanno shelf life residua)
        if (self.wh.wh is None) or (len(self.wh.wh) == 0): # siccome wh può essere un dict, None o Warehouse uso questo controllo
            #per creare l'inventario iniziale
            if init_level <= 0: init_level = self.pol.S
            self.wh.init_inventory(Vnd = vendor, level = init_level, min_rsl = 1) # qui non usiamo self.pol.m_rsl per avere potenzialmente sin da subito prodotti con shelf life unitaria
        # statistiche necessarie al calcolo del profitto e di altri indicatori KPIs 
        self.n_Purchase_orders:int = 0 # numero di ordini di acquisto!
        self.products:dict[str, dict[float, list[Item]|Item]] = {
                                                                    'purchased': {}, 
                                                                    'delivered':{}, 
                                                                    'wasted': {}
                                                                    } # tiene traccia dei prodotti aquistati, consegnati e di quelli smaltiti
        
        if val_init: # se valorizziamo anche il magazzino iniziale allora li buttiamo tutti nei prodotti acquistati
            self.products['purchased'] = {0: [item for item in self.wh.iter_products()]}
        
        self.s_trigger = self.env.event() # evento che segnala se abbiamo raggiunto s
        self.order_types:dict[str:list[tuple[float, int]]] = {  # la lista contiene tuple con gli istanti in cui sono stati fatti gli ordini e le quantità teoriche da ordinare (non arrotondate)
                                                                's':[], # s-triggered
                                                                'i':[], # I-triggered
                                                                'w':[] # due to unacceptable shelf life
                                                              } 
    
    
    """ PROCESSI SIMPY """    
        
    def gen_daily_demand(self, demand: Cont_distrib, *, pr_long_sl:float = 0.4, split_factor:int = 4, print_out:bool = False):
        """ Processo simpy che genera la domanda giornaliera, questo processo andrà poi ù
                sostituito dagi agenti clienti. La quantità Q viene spalmata
            - la domanda Q viene suddivisa in ordini q = (1 to Q/split_factor) 
            - le singole quantità q vengono rilasciate in maniera distribuita durante l'arco della giornata
            - si ipotizza che il 40% delle volte venga preso il prodotto a shelf life massima (pr_long_sl)
            - negli altri casi si prende un prodotto du sl qualsiasi, ma comunque con probabilità 
                    comunque maggiore per shelf life alte.
        """
        def split_quantity(Q):
            """ serve per splittare la quantità ordinata"""
            rem_q = Q
            quantities = []
            while rem_q > 0:
                max_q = min(rem_q, Q // split_factor)
                q = rn.randint(1, max_q)
                quantities.append(q)
                rem_q -= q
            return quantities
        
        while True:
            # la domanda
            daily_demand = max(0, int(round(gen_random_val(demand), 0))) # domanda giornaliera intera, generata casualmente
            quantities = split_quantity(daily_demand) # la domanda è suddivisa in ordini q 
            
            """ *********************** """
            if print_out: 
                st =f'domanda a {int(self.env.now)}: {daily_demand} in batch -> {quantities}'
                print(st)
            
            """ *********************** """
            if daily_demand == 0: # caso raro di domanda nulla, in questo caso aspettiamo un giorno e poi ripartiamo
                yield self.env.timeout(1)
                continue
            # gli istanti di tempo
            times = sorted(rn.uniform(0, 1) for _ in quantities) # generiamo n istanti di tempo (i temp di ordinazione)
            delta_t = [times[0]] + [times[i]- times[i-1] for i in range(1, len(times))] # gli intertempi tra un ordine e il successivo
            for q, dT in zip(quantities, delta_t):
                yield self.env.timeout(dT)
                r = rn.random()
                # scelta del tipo di shelf life da utilizzare (si seglie uno dei due Iterator)
                iter_rsl = max_sl if r <= pr_long_sl else  almost_rnd_sl
                t_now = round(self.env.now, 4)
                batch =  self.wh.take_items(q, t_now, iter_rsl) # raccoglie i prodotti e provvede a aggiornare oh stock out, ecc.
                if batch != []: 
                    self.products['delivered'][t_now] = batch # li mette tra i prodotti consegnati
                
                """ POSSIBILE ORDINE STRAORDINARIO """
                if self.wh.ip <= self.pol.s: self._extra_order(round(t_now,4))  
            
            # quanto manca a arrivare a fine giornata
            wait_time = 1 - times[-1] 
            yield self.env.timeout(wait_time) 
    
    def periodic_order(self):
        """ Processo simy che genera gli ordini 
            a periodi costanti I """
        while True:
            I_trigger = self.env.timeout(self.pol.I)
            s_trigger = self.s_trigger
            result = yield I_trigger | s_trigger
            if s_trigger in result: self.s_trigger = self.env.event()
            else: 
                 # se il trigger s è scattato da poco l'ip potrebbe essere ancora sopra a S
                 # in questo caso non facciamo nulla
                q = (self.pol.S - self.wh.ip)
                if q  > 0: 
                    self.order_types['i'].append((round(self.env.now, 2), q))
                    self._make_order(self.env.now)
                #print(self.wh.ip)
                
    def update_warehouse(self,*, min_q:Optional[int] = None, 
                         min_qw:Optional[int] = None, 
                         min_rsl:Optional[int] = None, n_receiv:int = 1):
        """ Processso Simpy che Aggiorna il magazzino:
            - ricalcolando lo shelf life,
            - cancellando gli scaduti,
            - aggiungere quello che è stato consegnato,
            - quanto si mette a magazzino il consegnto, eventuali scaduti o prodotti con shelf_life troppo corta,
              non vengono accettati. Gli scartati vengono immediatamente riordinati, senza costo fisso d'ordine.
            
            Argomenti di input:
            - min_rsl il valore minimo di shelf life residuo ritenuto accettabile, default self.pol.m_rsl
            - min_q è il lotto minimo di riordino che lanciamo se alcuni prodotti non sono stati accettati perchè prossimi a scadenza, di default pari a slef.pol.m_q 
            - n_receiv è il numero di volte in cui si controlla se è arrivato qualcosa, questo non è incluso nella politica!
            
            Oss. 
                - se arrivano prodotti durante la giornata, la shelf life considereremo il giorno precedente come arrivo per calcolare la shelf life, ù
                  altrimenti non ci sarebbe pià coerenza con le chiavi del dizionario sh
                - se q* prodotti sono in scadenza ne riordiniamo proprio max(min_q, q*), non andiamo, come nel caso generale di riordino per multipli interi di min_q
        """

        if min_rsl is None: min_rsl = self.pol.m_rsl
        if min_q is None: min_q = self.pol.m_q
        if min_qw is None: min_qw = self.pol.m_qw
        while True:

            # STAMPA STATO MAGAZZINO A INIZIO GIORNO
            day = int(self.env.now)
            wh_state = {rsl: len(items) for rsl, items in sorted(self.wh.wh.items())} if self.wh.wh else {}
            print(f"---- Giorno [{day}] ---- Stato del magazzino -> {wh_state}")
            
            times = [1/n_receiv for i in range(1, n_receiv)] # gli istanti in cui si guarda se è arrivato qualcosa
            for dt in times:
                yield self.env.timeout(dt) 
                if self.dp != []: 
                    self._add_delivered(min_q, min_qw) # potrebbero partire più micro ordini, uno a ogni dt ... Pazienza è un'eventualità rarissima perchè è praticamente impossibile che arrivino più lotti nella stessa giornata
            yield self.env.timeout(1 - sum(times))
            self._remove_old() # qui aggiorniamo il magazzino togliendo gli scaduti, prima di caricare eventuali nuovi arrivi
            if self.dp != []: 
                self._add_delivered(min_q, min_qw)
            
            """ CONTROLLO CONGRUENZA DELLE SHELF LIFE A FINE GIORNATA """
            self._shelflife_consistency()
    
    """ Metodi di utilità generale """
    
    def _remove_old(self):
        """ toglie eventuali scaduti e aggiorna le shelf lifes,
            alla fine facciamo il controllo di congruenza perchè vale
                 solo a inizio giornata """
        t_now = self.env.now
        for item in self.wh.iter_products(): # cicliamo su tutti gli item
            rsl = item.rem_sl(t_now, in_place = True) # aggiornamento shelf life
            if rsl < 1: item.tt[1] = t_now # questi usciranno dal magazzino
        waste = self.wh.wh.get(1, None)
        if waste: 
            """ CONTROLLO CONGRUENZA """
            if not self.check_shelf_life(waste): raise Exception("Remaining Shelf Life degli scaduti Errata")
    
            self.products['wasted'][round(t_now, 4)] = waste # registriamo gli scaduti
            self.wh.oh = max(0, self.wh.oh - len(waste))
            self.wh.update_plots(round(t_now, 4))
        """ aggiornamento della struttura del magazzino """
        W = self.wh
        W.wh = {rsl - 1: W.wh[rsl] for rsl in W.wh.keys() if rsl != 1}
        
        """ POSSIBILE ORDINE STRAORDINARIO """
        if W.ip <= self.pol.s: self._extra_order(round(t_now,4))        
    
    def _extra_order(self, t_now):
        q = self.pol.S - self.wh.ip
        self.order_types['s'].append((round(t_now, 2), q))
        self._make_order(t_now = round(t_now, 4)) # si lancia il processo di acquisto straordinario!!!
        if self.pol.r_double == False: 
            self.s_trigger.succeed() # attiviamo il trigger
        
    def _add_delivered(self, min_q:int, min_qw:int):
        dp = self.dp
        t_now = self.env.now
        # qui 'è un' approssimazione, 
        # scartiamo quelli prossimi alla scadenza, che però, non appena arrivati 
        # avrebbero potuto essere ancora accettabili ... problema che comunque si riduce all'aumentare della frequenza di controllo!!!
        good_items = [item for item in dp if item.rem_sl(t_now, in_place = False) >= self.pol.m_rsl] # quelle sane, il filtro è basato sulla shelf life effettiva!!! Che non registriamo, lo facciamo dopo
        waste = [item for item in dp if item not in good_items] # quelle in via di scadenza 
        self.dp.clear()
        self.wh.stock_items(good_items, round(t_now, 4)) # qui si aggiorna anche on hand e on_order
        self.products['purchased'][round(t_now, 4)] = good_items # registriamo i prodotti buoni acquistati
        
        """ POSSIBILE ORDINE STRAORDINARIO """
        if waste != []: # partirà un ordine speciale, non si incrementa n_purchase_order perchè questo non viene pagato
            n_waste = len(waste)
            q_order = max(min_qw, n_waste) # qui non andiamo per multipli di min_q
            self.order_types['w'].append((round(t_now, 2), q_order))
            delta_oo = 0 if q_order == n_waste else (q_order - n_waste)
            self.env.process(self.vd.deliver(q_order, self.dp)) # si lancia un processo di rifornimento 
            # abbiamo già incrementato on order di len(good_items) col metodo wh.stock 
            # dobbiamo registrare solo l'eventuale extra ordine
            self.wh.update_oo_and_plots(q = delta_oo, t = round(t_now, 4))   
        
    def _make_order(self,t_now:float):
        """ Ordini straordinari o ordini standard a intervalli di tempo I"""
        Q = mt.ceil(self.pol.S - self.wh.ip) # quantità teorica di ricostituzione
        Q = mt.ceil(Q/self.pol.m_q)*self.pol.m_q # lotto proporzionale al lotto minimo
        self.env.process(self.vd.deliver(Q, self.dp)) # processo di consega!!! 
        self.wh.update_oo_and_plots(q = Q, t = round(t_now, 4))
        self.n_Purchase_orders += 1
        
    def delivered_prod(self)-> Iterable:
        return chain.from_iterable(self.products['delivered'].values())

    def purchased_prod(self) -> Iterable:
        return chain.from_iterable(self.products['purchased'].values())
    
    def wasted_prod(self)-> Iterable:
        return chain.from_iterable(self.products['wasted'].values())
    
    def stocked_prod(self) -> Iterable:
        return self.wh.iter_products()
    
    def end_products(self) -> dict[str, int]:
        """ dizionario con numero di prodotti per tipologia """
        return {'purchased':len(tuple(self.purchased_prod())),
                'delivered': len(tuple(self.delivered_prod())), 
                'disposed': len(tuple(self.wasted_prod())), 
                'in_stock': len(tuple(self.stocked_prod()))
                }
    
    def end_orders(self):
        "numero di ordini fatti per tipologia"
        return {key:len(val) for key, val in self.order_types.items()}
    
    def total_stock_time(self) -> float:
        tot_t = sum(item.time_in() for item in chain(self.delivered_prod(), self.wasted_prod())) # temo a magazzino dei prodotti usciti
        t_now = self.env.now
        tot_t += sum(item.time_in(t_now) for item in self.stocked_prod()) # qui si passa t_now perchè questi non hanno t2 (il tempo di uscita)
        return tot_t
    
    @property
    def tot_revenue(self) -> float:
        """ restituisce il ricavo totale """
        n_sold = sum(1 for _ in self.delivered_prod())
        n_purchased = sum(1 for _ in self.purchased_prod())
        n_disposed = sum(1 for _ in self.wasted_prod())
        n_order = self.n_Purchase_orders
        n_lost = self.wh.lost
        rev = self.cst(Ns = n_sold, 
                 Np = n_purchased, 
                 No = n_order, 
                 Nu = n_lost, 
                 Nd = n_disposed, 
                 Tot_t = self.total_stock_time()
                 )
        return rev
    
    @property
    def pr_stock_out(self) -> float:
        """ probabilità di stock out"""
        o = self.order_types
        N_cycles = len(o['i']) + len(o['s']) # numero di ordini e quindi di cicli d'ordine
        return round(self.wh.n_stock_out/N_cycles, 5)
    
    @property
    def fill_rt(self) -> float:
        """ fill rate, probabilità che un ordine venga completato"""
        fl_or, tot_or = self.wh.fill_rate
        return round((fl_or/tot_or), 5)
    
    @property
    def average_oh(self) -> float:
        """ livello di oh medio """
        # dizionari mantengono ordinamento, non è necessario ordinare i valori
        OH = list(self.wh.inventory_plot['oh'].items()) # (istante, valore oh)
        avg_oh = 0
        for ((t1, oh1), (t2, _)) in zip(OH[:-1], OH[1:]):
            if t2 < t1: raise Exception("istanti di tempo non ordinati")
            avg_oh += oh1*(t2 - t1)
        return round(avg_oh/t2, 2)
    
    @property
    def I_triggered_orders(self) -> float:
        """ percentuale di ordini legati all'intervallo I"""
        end_orders = self.end_orders()
        return round(end_orders['i']/(end_orders['i'] + end_orders['s']), 2)
    
    @property
    def lost_sales(self) -> float:
        """ percentuale di ordini persi sul totale """
        sales = self.end_products()['delivered']
        return round(self.wh.lost/(sales + self.wh.lost), 2)
   
    """ METODI PER CONGRUENZA E VERIFICA SHELF LIFE """
    def _shelflife_consistency(self):
        """ verifica la congruenza tra la shelf life degi item
            e le chiavi del dizionario self.wh.wh
        """
        rsl_check = self.check_shelf_life_warehouse()
        if not all(val for val in rsl_check.values()): raise Exception("Remaining Shelf Life a magazzino non corretta")
        if self.wh.oh != self.wh.eval_oh(): raise Exception("On Hand non correttamente calcolato")
    
    def check_shelf_life(self, items:Iterable[Item],*, srl_range: tuple = (0, 1), t:Optional[float] = None) -> bool:
        """ verifica che la shelf life sia contenuta nel range srl_range """
        def foo(item, t):
            if t is not None: return item.rem_sl(t, in_place = False)
            return item.rsl
        low, high = srl_range
        return all(low <= foo(item, t) <= high for item in items)
    
    def check_shelf_life_warehouse(self, *rsls:int) -> dict[int:bool]:
        """ verifica che la shelf life degli item a magazzino sia corretta """
        if rsls == (): rsls = sorted(self.wh.wh.keys())
        return {rsl: self.check_shelf_life(self.wh.wh[rsl], srl_range = (rsl,rsl + 1)) for rsl in rsls}
    
    