from __future__ import annotations
import simpy as sp
import random as rn
import math as mt
import statistics as stats
import bisect
import time
from scipy.stats import norm, triang
from typing import Optional, Iterable, Callable
from Utilities import discr_cont_distrib, Theoretical_SsI_Values, Costs, Policy, CR
from ITEM import Item
from VENDOR import Vendor
from BUYER import Warehouse, Buyer
from AGENT import Warehouse_2, Discount, Shopper, Discount_Shopper, max_sl
from Regressione import model_predict, aggiornamento_X, aggiornamento_modello
import matplotlib.pyplot as plt  # Per fare il plottare i grafici fo, S, s

Pol = Policy(s=None, S=None, I=None, m_q=50, m_qw=5, m_rsl=3, r_double=False)


def update_policy(old_policy: Policy, **kargs) -> Policy:
    """ Serve a modificare i valori S,s,I della policy. 
        Trattandosi di una NamedTuple i valori non possono essere cambiati direttamente
        per cui usiamo una funzioncina """
    policy_dict = old_policy._asdict()
    kargs = {k: val for k, val in kargs.items() if k in policy_dict}
    policy_dict.update(kargs)
    return Policy(**policy_dict)


"""
TRE FUNZIONI DA PASSARE COME ARGOMENTO DI INPUT ALLA CLASSE SA 

La classe SA richiede in input tre funzioni:
    - una per valutare la funzione obiettivo (fitness function)
    - una per generare dei 'vicini'
    - una per scegliere il livello di 'fidelity'. 
 
Oss_1. 
Una bassa fidelity si basa su poce repliche simulative, una alta su molte repliche simulative. 
Durante il SA la fidelity verrà modificata in base alla temperatura corrente e all'errorevstandard tra le soluzioni confrontate. 
    
Oss._2. 
La funzione obiettivo deve adeguarsi al seguente protocollo.
    - La classe SA, al termine di una simulazione di n_run, restituisce un dizionario contenente delle liste con i risultati di ogni run.     
    - In questo dizionario ogni chiave rappresenta un obiettivo o vincolo e le chiavi sono:
         * revenues, 
         * stock_out_prob, 
         * fill_rate, 
         * average_oh, 
         * I_triggered, 
         * lost
    - Ad ogni chiave è associata una lista contenente i valori calcolati per ogni run per il corrispondnete indicatore. 
    - La funzione obiettivo creata deve ricevere in input questo dizionario!!!

"""


def objective_function(target_fr: float, daily_penalty: float) -> Callable:
    """ Una closure che definisce la funzione che verrà usata per calcolare la fitness di una soluzione Ssi
        FO = REVENUE - penalty*max(0, fill_rate - target_fill_rate).
        In pratica il fill rate è considerato un vincolo incluso ella funzione obiettivo, il gap positivo tra fill rate e 
        fill rate desiderato viene trasformato in un costo, moltiplicandolo per una penalità.
    """

    def inner(results: dict[str, list[float]]) -> list[float]:
        """ restituisce il vettore con la funzione obiettivo per ogni valore testato """
        nd = results['n_days']  # numero di giorni delle simulazioni fatte
        return [round(rev, 2) - round(nd * daily_penalty * max(0, target_fr - fr), 2) for rev, fr in
                zip(results['revenues'], results['fill_rate'])]

    return inner


# Per calcolare la penalty si può usare questa funzione; non va comunque passata in input a SA.
def eval_penalty(S: int, s: int, i: int, *,  # i parametri della politica
                 p_rev: float = 0.05,  # percentuale di variazione del ricavo
                 delta_fr: float = 0.01,  # gap tra fill rate teorico e effettivo
                 sa: SA, seeds: Iterable[int] = (10, 20, 30, 40, 50), n_days: int = 100) -> float:
    """
        In pratica fissa la penalty in modo che un gap di fill rate pari a delta_fr corrisponda a un costo 
        pari a p_rev*ricavo_medio. Di default un punto percentuale di fill rate corrisponde al 5% dei ricavi. 
        Il ricavo_medio è ottenuto simulando usando i seed 10, 20, ... 
        Dato che il numero di giorni potrebbe cambiare, la penalty viene restituita come penalty_giornaliera!!!
    """
    avg_rev = stats.mean(sa.simulate(S=S, s=s, i=i, seeds=seeds, n_days=n_days)['revenues'])
    print(avg_rev)
    return round((p_rev * avg_rev) / (delta_fr * n_days), 2)


def neighbor_function(Sm: int = 1, SM: int = mt.inf, sm: int = 1, sM: int = mt.inf,
                      # valore minimo e massimo di S, valore minimo s e massimo di s
                      dq: tuple = (-20, -10, 10, 20),  # variazioni di q = S - s
                      prob: tuple = (0.4, 0.4, 0.2)  # probabilità di agire solo su S, solo su s, su entrambi
                      ) -> Callable:
    """ closure usata per generare delle soluzioni limitrofe """

    def inner(S: int, s: int, i: int = 1, dT: float = 0):
        """ Si ragiona in termini di q con q = S - s, che viene aggionato pescando casualmente da dq;
            via via che ci si avvicina a Tend (termine del SA) si tende a generare soluzioni più localizzate, dT è proprio il rapporto tra Tend/T con T la temperatura corrente.
            Oss. i non serve a nulla, solo per compatibilità generica.
        """

        def clamp_int(x, lo, hi):
            # appartenenza al range
            return int(max(lo, min(x, hi)))

        # variazione di q
        u = rn.random()
        if u <= 1 - (dT * 0.9):  # caso standard
            q = rn.choice(dq)
        else:  # caso raro, variazioni piccole!!!
            min_q = min(q for q in dq if q > 0)
            q = rn.randint(-min_q, min_q)
            if q == 0: q = 1

        # ripartizione di q tra s e S
        u2 = rn.random()
        if u2 <= prob[0]:
            # solo S
            qS, qs = q, 0
        elif u2 <= prob[0] + prob[1]:
            # solo s
            qS, qs = 0, q
        else:
            # entrambi
            qS = mt.ceil(rn.random() * q)
            qs = q - qS

        S = clamp_int(S + qS, lo=Sm, hi=SM)
        s = clamp_int(s + qs, lo=min(S, sm), hi=min(S, sM))
        return S, s

    return inner


def check_fidelity(T: float, Th: float, deltas: list[float], fidelity: int = 1) -> bool:
    """ Se restituisce True la fidelity usata va bene, altrimenti va aumentata 
            - T: temperatura corrente, 
            - Th temperatura alta (non T_start, ma comunque alta).
            - deltas: le differenze tra le due soluzioni confrontate """
    n_wins = sum(1 for d in deltas if d >= 0)  # numero di volte in cui la soluzione nuova è risultata migliore
    nd = len(deltas)
    n_loss = nd - n_wins
    d_mean = stats.mean(deltas)  # delta medio

    # temperatura alta
    if T >= Th:  # al massimo si va a fidelity 2, se siamo a 1 ci fermiamo se almeno il 75% delle osservazioni è vincente
        if (n_wins / nd >= 0.75 and d_mean >= 0) or (n_loss / nd >= 0.75 and d_mean <= 0) or fidelity > 1:
            return True
        else:
            return False

    # tempereatura fredda
    if T < Th:
        if fidelity == 1 and (
                n_wins == nd or n_loss == nd):  return True  # in questo caso vogliamo che siano tutte vincenti
        if fidelity == 2:  # qui vogliamo almeno 75% vincenti o errore basso
            st_error = stats.stdev(deltas) / mt.sqrt(nd)
            z = abs(d_mean) / st_error
            if (n_wins / nd >= 0.75 and d_mean >= 0) or (n_loss / nd >= 0.75 and d_mean <= 0) or z >= 3:
                return True  # ci accontentiamo del 75%
            else:
                return False
        if fidelity == 3: return True


class SA:
    """ CLASSE che esegue Simulated Annealing per ottimizzare la politica S,s,I con prodotti deperibili 
        Strategia implementativa
         - si usano differenti fidelity level (da 1 a ... X) a ciascuno dei quali sono associate liste di 
           seed via via più numerosi. Ad ogni seed corrisponde un run della simulazione, per cui a basse fidelity corrisponderanno confronti fatti su poche simulazioni
         - usando uno stesso seed, si ottiene sempre la stessa sequenza di valori pseudo casuali
         - le soluzioni vengono confrontate usando gli stessi seed per ridurre la varianza, e scegliendo l'opportuno
           livello di fidelity (all'inizio basso e via via più alto) 
         - si usa una memoization; in pratica c'è una cache in cui vengono salvati i risultati di ogni soluzione Ssi valutate per ogni run simulativa eseguita, 
           di modo che se dovesse ricapitare una stessa soluzione non stiamo a rivalutarla. 
         - i seed sono incrementali es. fidelity 1 -> seed (1,2,3), fidelity 2 -> seed (1,2,3,4,5). In questo modo se una soluzione è già stata valutata a 
           fidelity 1, se dovesse essere valutata a fidelity 2 allora eseguiremmo solo due simulazioni (seed 4 e 5) dato che per le prime tre i 
           risutati sono già salvati nella cache. Inoltre nell cache salviamo solo i risultati relativi alla massima fidelity utilizzata 
           """

    """ VARIABILI DI CLASSE - IN PRATICA QUI SCEGLIAMO I PARAMETRI, OLTRE A QUELLI DEFINITI NELLE CLOSURE """
    # DATA & DISTRIBUTION
    lead_time = triang(c=0.8, loc=0.5,
                       scale=3)  # c posizione della moda 0.5 centrale, loc estremo inferiore, scale ampiezza intervallo
    shelf_life = discr_cont_distrib(triang(c=0.8, loc=5, scale=7),
                                    [i for i in range(1, 100)])  # shelf life discretizzata
    daily_demand = norm(100, 10)  # mu e sigma della domanda

    stock_out_prob: float = 0.05  # probabilità di stock_out per calcolo valori teorici
    vendor_min_lt: float = 0.5  # il lead time minimo del fornitore
    n_shipments: int = 3  # numero di controlli quotidiani per vedere se è arrivato un ordine
    pr_long_shelf_life: float = 0.4  # probabilità che le persone cerchino i prodotti con scadenza più lontana
    n_demand_split: int = 5  # numero di sotto batch in cui la domanda giornaliera viene suddivisa

    # COSTS & REVENUES
    Cr = CR(sale_price=10, cost=Costs(pc=5, oc=500, dc=0.5, u=0,
                                      h=0.1))  # non consideriamo costo mancata vendita, dato che includiamo il fill rate come vincolo
    Pol = Policy(s=None, S=None, I=None, m_q=50, m_qw=5, m_rsl=3, r_double=False)  # S, s, I verranno aggiornati subito

    def __init__(self, f_obj: Callable, f_ngh: Callable, f_fid: Callable,  # le tre funzioni
                 fidelity: dict[int, int]
                 # la fidelity: livello di fidelity e corrispondete numero di seed  - i seed devono essere incrementali, per ridurre varianza
                 ):
        """ funzioni di input 
            - f_obj la funzione che calcola la fitness
            - f_ngh la funzione che crea i vicini
            - f_fid la funzione che sceglie il livello di fidelity
        """

        # aggiorniamo i valori SsI della politica usando i valori 'ottimi teorici'
        SA.Pol = update_policy(SA.Pol,
                               **Theoretical_SsI_Values(SA.daily_demand, SA.lead_time, SA.Cr.H, SA.Cr.cost.oc,
                                                        I=None, safety_level=1 - SA.stock_out_prob)[1])

        self.fid = {fid: n_seeds for fid, n_seeds in
                    sorted(fidelity.items())}  # F. ordina i livelli di fidelity passati in inp
        self.sds = {fid: tuple(range(1, 5 * fidelity[fid], 5)) for fid in
                    fidelity.keys()}  # i seed di ogni livello di fidelity, a step di 5
        self._fid = sorted(self.fid.keys(), reverse=True)  # livelli di fidelity dal più alto al più basso

        self.f_obj = f_obj  # la funzioine obiettivo
        self.f_ngh = f_ngh  # funzione generazione vicini
        self.f_fid = f_fid  # funzione di scelta del livello di fidelity richiesto

        # dizionario delle soluzioni esplorate, ad ogni chiave (S, s, i, f) - con f il livello di fidelity - associa
        # il dizionario dei risultati {kpi_1:[lista valori], kp1_2:[lista valori] ...} le chiavi sono: * revenues,
        # stock_out_prob, fill_rate, average_oh, I_triggered, lost
        self.cache: dict[tuple, tuple[float, dict]] = {}
        # per ogni valore di I testato, registriamo l'evoluzione della fitness durante il miglioramento, in questo
        # dizionario la chiave è una tupla ((S, s, i), nrep) dove nrep è il progressivo delle ripetizioni fatte. Il
        # valore è la funzione obiettivo
        self.improvements: dict[int, dict[tuple, float]] = {}
        # aggiunti per i grafici - f
        self.fo_storico: list[float] = []  # <---  F. riga aggiunta per grafico
        self.storico_iterazioni = []  # <---  F. riga aggiunta per grafico della fo
        self.storico_policy = []  # <---  F. riga aggiunta per grafico della S,s

    def __getitem__(self, Ssi: tuple[int]) -> tuple[dict[str, list[float]], int] | tuple[
        None, None]:  # restituisce i valori della cache
        """ Restituisce l'eventuale soluzione Ssi presente nella cache, 
                    e il livello di fidelity per il quale è calcolato """
        S, s, i = Ssi
        for f in self._fid:  # la fidelity massima a scendere
            try:
                return self.cache[(S, s, i, f)], f  # restituisce i valori letti nella cache
            except:
                continue
        return None, None

    def __setitem__(self, Ssif: tuple[int], result: dict):
        """ inserisce il risultato e cancella dalla cache l'eventuale soluziokne Ssi  
                    predentemente valutata a fidelity minore """
        S, s, i, f = Ssif
        _, f_old = self[S, s, i]
        if f_old is not None: del self.cache[(S, s, i, f_old)]  # cancelliamo quella a bassa fidelity
        self.cache[(S, s, i, f)] = result

    def evaluate_solution(self, S: int, s: int, i: int, f: int, *, n_days: int = 100,
                          n_sol_run: Optional[list[int, int]] = None) -> list[float]:
        """ esegue le simulazioni della soluzione Ssi usando i seed della fidelity f
            se questa soluzione è già stata valutata a fidelity minori, esegue solo le simulazioni addizionali ù
            provvede anche ad aggiornare la cache
            Input opzionali
            n_day: serve a settare il numero di giorni da utilizzare nelle simulazioni.
            n_sol_rep tiene traccia del numero di soluzioni differenti generate e di run simulativi fatti 
            in questo caso serve una lista per via della mutabilità ... Soluzione bruttina, ma vabbè...
            """
        old_results, f_existing = self[S, s, i]
        if f_existing is not None:  # trovato ma a fidelity minore, alcune simulazioni vanno rilanciate
            if f_existing >= f: return self.f_obj(
                old_results)  # trovato ad una fidelity maggiore o uguale (se maggiore, avremo più valori di quanti richiesti, ma non importa)
            ns = len(old_results[
                         'revenues'])  # consideriamo le revenue, ma un output vale l'altro, ci interessa solo capire quanti seed nuovi dovanno essere valutati
            seeds = self.sds[f][ns:]  # seed rimanenti da usare
        else:
            seeds = self.sds[f]
        if n_sol_run:
            n_sol_run[0] += 1  # aggiungiamo la nuova soluzione
            n_sol_run[1] += len(seeds)  # aggiungiamo il nuovo numero di run fatti
        new_results = self.simulate(S, s, i, seeds, n_days=n_days)
        if old_results is not None:
            for key in old_results:
                new_results[key] = old_results[key] + new_results[key]
        self[S, s, i, f] = new_results  # aggioriamo e cancelliamo
        return self.f_obj(new_results)

    def compare_solutions(self, obj_new: Iterable[float], obj_old: Iterable[float]) -> tuple[float, float]:
        """ confronta due soluzioni alternative s1[x11, x12, ...], s2[x12, x22,.. ], 
            e restituisce il vettore della differenze [(x11 - x12), (x21 - x22), ...] """
        return [o_new - o_old for o_new, o_old in zip(obj_new, obj_old)]

    def simulate(self, S: int, s: int, i: int, seeds: Iterable[int] = tuple(i for i in range(1, 6)), n_days: int = 100):
        """ simula per ogni seed e raccoglie:
                ricavo, pro_babilità di stock out, fill rate, 
                    percentuale di ordini triggherati da I, ordini persi """

        results = {'revenues': [], 'stock_out_prob': [], 'fill_rate': [],
                   'average_oh': [], 'I_triggered': [], 'lost': [], 'n_days': n_days}

        for seed in seeds:
            rn.seed(seed)  # cos' abbiamo la stessa generazione di numeri pseudocasuali
            env = sp.Environment()
            # Venditore e Acquirente
            V = Vendor(env, LT_distrib=SA.lead_time, SL_distrib=SA.shelf_life, product_kind="Pr",
                       min_lt=SA.vendor_min_lt)
            B = Buyer(env, vendor=V, policy=update_policy(old_policy=SA.Pol, S=S, s=s, i=i),
                      costs=SA.Cr, wh=None, init_level=S, val_init=True)
            # PROCESSI
            env.process(B.update_warehouse(min_q=None, min_rsl=None, n_receiv=SA.n_shipments))
            env.process(B.gen_daily_demand(demand=SA.daily_demand,
                                           pr_long_sl=SA.pr_long_shelf_life, split_factor=SA.n_demand_split,
                                           print_out=False))
            env.process(B.periodic_order())
            env.run(until=n_days)
            for key, val in zip(results.keys(), (
                    B.tot_revenue, B.pr_stock_out, B.fill_rt, B.average_oh, B.I_triggered_orders, B.lost_sales)):
                results[key].append(val)
        return results
    
    def simulate_agenti(self, S: int, s: int, i: int, seeds: Iterable[int] = tuple(i for i in range(1, 6)), n_days: int = 100):

        results = {
            'revenues': [],
            'stockoutprob': [],
            'fillrate': [],
            'averageoh': [],
            'Itriggered': [],
            'lost': [],
            'ndays': n_days
        }

        for seed in seeds:
            rn.seed(seed)
            env = sp.Environment()

            # Fornitore come nel modello base
            V = Vendor(env,
                       LT_distrib=SA.lead_time,
                       SL_distrib=SA.shelf_life,
                       product_kind=Item,
                       min_lt=SA.vendor_min_lt)

            # Magazzino e punto vendita con scontistica
            wh = Warehouse_2(None)
            B = Discount(env,
                         vendor=V,
                         policy=update_policy(SA.Pol, S=S, s=s, I=i),
                         costs=SA.Cr,
                         wh=wh,
                         init_level=S,
                         val_init=True,
                         discount_policy={1: 0.50, 2: 0.30, 3: 0.10})

            # processi
            env.process(B.update_warehouse(min_q=None,
                                           min_rsl=None,
                                           n_receiv=SA.n_shipments))
            env.process(B.periodic_order())

            # agenti

            #q_dist = SA.daily_demand
            q_dist = triang(c=0.5, loc=1, scale=4)

            dt_mean = 5  # media intertempo tra due acquisti
            perc_disc = 0.20  # percentuale di agenti sensibili agli sconti

            # Numero di agenti calibrato per avere stessa domanda media giornaliera
            D = SA.daily_demand.mean()
            q_mean = q_dist.mean()
            N = int(round(D / (dt_mean * q_mean)))

            N_discount = int(N * perc_disc)
            N_shopper = N - N_discount

            agents: list[Shopper | Discount_Shopper] = []

            # Shopper "base"
            for k in range(N_shopper):
                a = Shopper(env=env,
                            Idx=f"A{k}",
                            Buyer=B,
                            demand=q_dist,
                            dt=dt_mean,
                            min_sl=SA.Pol.m_rsl,
                            min_q=0.5,
                            behaviour=max_sl)
                agents.append(a)

            # Discount_Shopper sensibili agli sconti
            for k in range(N_discount):
                d = Discount_Shopper(env,
                                     f"AD{k}",
                                     B,
                                     q_dist,
                                     dt_mean,
                                     SA.Pol.m_rsl,
                                     0.5,
                                     max_sl,
                                     discount_acceptance={1: 0.5, 2: 0.3})
                agents.append(d)

            # Attivazione dei processi di acquisto degli agenti
            for a in agents:
                env.process(a.buy_process())

            # simulazione fino ad n_days
            env.run(until=n_days)

            # Raccolta
            for key, val in zip(results.keys(),
                                (B.tot_revenue,
                                 B.pr_stock_out,
                                 B.fill_rt,
                                 B.average_oh,
                                 B.I_triggered_orders,
                                 B.lost_sales)):
                if key == 'n_days':
                    continue
                results[key].append(val)

        return results

    """ FUNZIONI SPECIFICHE DEL S.A."""

    def Gen_T(self, S, s, i, init_acc_prob: float = 0.3, end_acc_prob: float = 0.01,
              nsol=200, seed=999, n_days=100):
        """ Serve a definire il valore di T_start e di T_end in modo da avere 
                    una certa probabilità di accettazione iniziale ed una finale. 
            In pratica genera nsol (50) neighbour e per ciascuno di essi stima (con unica run di simulazione) la  funzione obiettivo;
            usa tali valori per settare T_start e T_end di modo che corrispondona alla probabilità di accettazione volutal.
        """
        to_print = f'Valutazone di T_start e di T_end per pr_accett. iniziale {init_acc_prob} e finale di {end_acc_prob}'
        print(to_print)
        objectives = []
        solutions = set()
        for rp in range(1, nsol + 1):
            if rp == 1 or rp % 10 == 0 or rp == nsol: print(f'Sto valutando la soluzione {rp}')
            if not (S, s) in solutions:
                solutions.add((S, s))
                obj = self.f_obj(self.simulate(S, s, i, seeds=(seed,), n_days=n_days))[
                    0]  # [0] serve per prendere la prima e unica soluzione generata
                bisect.insort(objectives, obj)  # teniamo la lista ordinata in senso crescente
            S, s = self.f_ngh(S, s)  # con valore di default dT = 0
        delta_obj = [objectives[-1] - of for of in
                     objectives[:-1]]  # gli errori riseptto alla miglior soluzione trovata
        mean = stats.mean(delta_obj)
        low_percentile = stats.quantiles(delta_obj, n=50, method="inclusive")[0]
        return -(mean / mt.log(init_acc_prob)), -(low_percentile / mt.log(end_acc_prob))

    @staticmethod
    def Choose_M_Moves(T_start, T_end, T_cooling_rate, n_run) -> tuple[int, int]:
        """ calcola il numero di ere, ossia periodi a temperatura costante
            e in base a questo determina il numero di mosse fisse da fare ad ogni era """
        nT = mt.ceil(mt.log(T_end / T_start) / mt.log(T_cooling_rate))  # numero di ere (periodi a temperatura costante)
        return nT, mt.ceil(n_run / nT)

    def optimize_fixed_I(self, i, *,  # di seguito parmetri opzionali che sono anche attributi dell'oggetto SA
                         T_start: Optional[float] = None, T_end: Optional[float] = None,
                         T_cooling_rate: float = 0.95,  # fattore riduzione della temperatura - cooling geometrico
                         n_days: int = 100,  # numero di giorni di ogni simulazione
                         init_acc_prob: float = 0.3,  # probabilità di accettazione iniziale
                         end_acc_prob: float = 0.01,  # probabilità di accettazione finale
                         n_run_era: int = 500,  # numero massimo di run simulativi per ogni era
                         n_no_imp: int = 150,  # numero di generazioni senza miglioramento
                         tot_time: bool = False,  # se true restituiamo anche il tempo totale
                         sa_seed: int = 12345):
        rn.seed(sa_seed)

        def accept_solution(delta_mean, T) -> bool:
            if delta_mean >= 0:
                return True
            u = rn.random()
            return u <= mt.exp(delta_mean / T)

        t_start = time.perf_counter()
        # Parte 1) Inizializzazione
        _, values = Theoretical_SsI_Values(SA.daily_demand,
                                           SA.lead_time, SA.Cr.H, SA.Cr.cost.oc,
                                           I=i,
                                           safety_level=1 - SA.stock_out_prob)  # i valori teorici S, s da cui si parte

        # inizializzazione: best_sol e current_sol alla partenza sono uguali ai valori teorici
        best_sol = (values['S'], values['s'], i)
        current_sol = best_sol

        # Valori inziali relativi al best e current simulati (simulo so se sono vicino al best o alla fine di ogni
        # epoca)
        best_obj_sim: list[float] = []
        mean_best_obj_sim: float = 0.0
        current_obj_sim: list[float] = []
        mean_current_obj_sim: float = 0.0

        # Valori iniziali relativi al modello di regressione
        best_obj_reg = model_predict(*best_sol)
        current_obj_reg = best_obj_reg

        self.improvements[i] = {0: best_obj_reg}

        if T_start is None or T_end is None:
            T_start, T_end = self.Gen_T(*best_sol, nsol=50, seed=999, n_days=n_days)

        T = T_start
        Th = 0.5 * T_start  # questa è la soglia di temperatura alta, usata nella scelta della fidelity. Per ora
        # fissa!!!
        no_improvement = 0
        n_sol_run = [0, 0]
        n_aggiornamenti_X = 0

        breakpoint()
        # Parte 2) Miglioramento
        while T >= T_end:
            to_print = f'T = {T} -> T_end = {T_end}'
            print(to_print)
            n_sol_run[1] = 0  # riportimo a zero solo il secondo valore (run) che serve come criterio di stop,
            # l'altro deve incrementare sempre

            while n_sol_run[1] <= n_run_era:  # numero fisso di repliche ad ogni temperatura
                n_sol_run[0] += 1
                n_sol_run[1] += 1
                no_improvement += 1

                # Parte 2.1 - generiamo un vicino e lo confrontiamo col current
                S, s = self.f_ngh(*current_sol, dT=round(T_end / T, 5))  # vicino

                current_obj_reg = model_predict(*current_sol)
                new_obj_reg = model_predict(S, s, i)

                delta_mean_reg = new_obj_reg - current_obj_reg
                accepted = accept_solution(delta_mean_reg, T)

                # se la soluzione nuova viene accettata new diventa current
                if accepted:
                    current_sol = (S, s, i)
                    current_obj_reg = new_obj_reg

                    # se la new (che è diventata current) è maggiore o limitrofa di best simulo best e new
                    # if current_obj_reg > best_obj_reg or current_obj_reg >= best_obj_reg * (1 - 0.01):
                    if current_obj_reg > best_obj_reg:
                        print(
                            f"Soluzione vicina al best: current sol: {current_sol} obj_new={current_obj_reg:.1f} vs "
                            f"best_obj*(1-0.01)={best_obj_reg * (1 - 0.01):.1f}")

                        best_obj_sim = self.evaluate_solution(*best_sol, f=3, n_days=n_days, n_sol_run=None)
                        current_obj_sim = self.evaluate_solution(*current_sol, f=3, n_days=n_days, n_sol_run=None)

                        # aggiorno il modello con le simulazioni HF appena fatte
                        aggiornamento_X(*best_sol, best_obj_sim)
                        aggiornamento_X(*current_sol, current_obj_sim)
                        n_aggiornamenti_X += 2

                        # calcolo le medie del best e del current
                        mean_best_obj_sim = stats.mean(best_obj_sim)
                        mean_current_obj_sim = stats.mean(current_obj_sim)

                        # se anche la media del current simulato è > della media del best simulato
                        # allora effettivamente il current diventa il best
                        if mean_current_obj_sim > mean_best_obj_sim:
                            print(
                                f"TROVATO NUOVO BEST: la soluzione {current_sol} con obj_new: {current_obj_reg} ha "
                                f"dato esito mean_current_obj_sim: {mean_current_obj_sim} maggiore della best_sol: {best_sol},"
                                f"con delta_best_obj_mean {mean_best_obj_sim}. la best_obj_reg è {best_obj_reg}")
                            best_sol = current_sol
                            best_obj_reg = current_obj_reg  # fo ottima dal regressore
                            print(f"nuovo best_obj_reg: {best_obj_reg}")
                            mean_best_obj_sim = mean_current_obj_sim  # fo ottima dalla simulazione
                            self.fo_storico.append(round(best_obj_reg, 2))

                            no_improvement = 0  # azzero il contantore delle iterazioni senza miglioramenti

                self.improvements[i][n_sol_run[0]] = round(best_obj_reg,
                                                           2)  # registriamo la soluzione nel dizionario improvement, può essere la stessa di prima
                self.storico_iterazioni.append((
                    n_sol_run[0],
                    round(T, 3),  # temperatura
                    round(current_obj_reg, 2),  # obj corrente
                    round(new_obj_reg, 2),  # obj nuovo
                    round(delta_mean_reg, 3),  # delta
                    accepted,  # accettata?
                    round(best_obj_reg, 2)  # best-so-far
                ))
                # per S,s
                self.storico_policy.append((
                    n_sol_run[0],
                    S,
                    s,
                    accepted
                ))

                if no_improvement > n_no_imp:  # condizionie di chiusura
                    print('OPTIMIZATION ENDS')
                    if not tot_time:
                        return best_sol, best_obj_reg
                    else:
                        return best_sol, best_obj_reg, round(time.perf_counter() - t_start, 2)

            to_print = f'Iterazioni SA: {n_sol_run[1]}, soluzioni totali valutate: {n_sol_run[0]}, soluzione migliore: {best_sol} fo: {best_obj_reg}'
            print(to_print)
            print()  # lasciamo uno spazio

            current_obj_sim = self.evaluate_solution(*current_sol, f=3, n_days=n_days, n_sol_run=None)
            aggiornamento_X(*current_sol, current_obj_sim)
            n_aggiornamenti_X += 1

            # Aggiorno il regressore se faccio 5 aggiornamenti della matrice X
            if n_aggiornamenti_X >= 5:
                aggiornamento_modello()
                print("Modello aggiornato")
                n_aggiornamenti_X = 0

            # Aggiorno best_obj_reg e current_obj_reg
            best_obj_reg = model_predict(*best_sol)
            current_obj_reg = model_predict(*current_sol)

            # Raffreddo T
            T = T * T_cooling_rate

        print('OPTIMIZATION ENDS')
        if not tot_time:
            return best_sol, best_obj_reg
        else:
            return best_sol, best_obj_reg, round(time.perf_counter() - t_start, 2)

    def optimize_fixed_I_sim(self, i, *,  # di seguito parmetri opzionali che sono anche attributi dell'oggetto SA
                             T_start: Optional[float] = None, T_end: Optional[float] = None,
                             T_cooling_rate: float = 0.95,  # fattore riduzione della temperatura - cooling geometrico
                             n_days: int = 100,  # numero di giorni di ogni simulazione
                             init_acc_prob: float = 0.3,  # probabilità di accettazione iniziale
                             end_acc_prob: float = 0.01,  # probabilità di accettazione finale
                             n_run_era: int = 500,  # numero massimo di run simulativi per ogni era
                             n_no_imp: int = 150,  # numero di generazioni senza miglioramento
                             tot_time: bool = False  # se true restituiamo anche il tempo totale
                             ):

        def accept_solution(delta_mean, T) -> bool:
            if delta_mean >= 0: return True
            u, acc_pr = rn.random(), mt.exp(delta_mean / T)
            if u <= mt.exp(delta_mean / T): return True
            return False

        """Usa il Simulated Annealing per scegliere il valore ottimale S, s ad un valore fisso dell'intervallo di 
        riordino 'i'"""
        t_start = time.perf_counter()
        # Parte 1) Inizializzazione 
        _, values = Theoretical_SsI_Values(SA.daily_demand,
                                           SA.lead_time, SA.Cr.H, SA.Cr.cost.oc,
                                           I=i,
                                           safety_level=1 - SA.stock_out_prob)  # i valori teorici S, s da cui si parte

        best_sol, best_obj = (values['S'], values['s'],
                              i), 0  # la tupla è la chiave che verrà salvata nel dizionario per valutare l'andamento della fitness
        # objs = self.evaluate_solution(*best_sol, 1)  # la soluzione di partenza a fidelity level basso (1)
        objs = self.evaluate_solution(*best_sol, f=1)

        best_obj = stats.mean(objs)  # valore medio delle funzioni obiettivo delle singole run di simulazione
        self.improvements[i] = {(0, best_sol): best_obj}  # registriamo la soluzione, 0 indica che è quella iniziale

        if T_start is None or T_end is None:
            T_start, T_end = self.Gen_T(*best_sol, nsol=50, seed=999, n_days=n_days)

        T = T_start
        Th = 0.5 * T_start  # questa è la soglia di temperatura alta, usata nella scelta della fidelity. Per ora fissa!!!
        current_sol = best_sol
        no_improvement = 0
        n_sol_run = [0, 0]

        breakpoint()
        # Parte 2) Miglioramento
        while T >= T_end:
            to_print = f'T = {T} -> T_end = {T_end}'
            print(to_print)
            n_sol_run[
                1] = 0  # riportimo a zero solo il secondo valore (run) che serve come criterio di stop, l'altro deve incrementare sempre

            while n_sol_run[1] <= n_run_era:  # numero fisso di repliche ad ogni temperatura

                # Parte 2.1 - generiamo un vicino e lo confrontiamo col current 
                S, s = self.f_ngh(*current_sol, dT=round(T_end / T, 5))  # vicino
                fidelity = 0
                while True:  # partiamo da fidelity 1 e poi via via cresciamo sino a quando serve
                    fidelity += 1
                    obj_old = self.evaluate_solution(*current_sol, f=fidelity, n_days=n_days, n_sol_run=n_sol_run)
                    obj_new = self.evaluate_solution(S, s, i, f=fidelity, n_days=n_days, n_sol_run=n_sol_run)
                    delta_obj = self.compare_solutions(obj_new,
                                                       obj_old)  # confrontiamo e otteniamo il vettore dei delta
                    fidelity_ok = self.f_fid(T, Th=Th, deltas=delta_obj, fidelity=fidelity)
                    if fidelity_ok: break

                    # Parte 2.2 Accettazione delle nuova soluzione
                # Parte 2.2 – decisione di accettazione (una sola volta)
                delta_mean = stats.mean(delta_obj)
                no_improvement += 1

                u = rn.random()
                accepted = (delta_mean >= 0) or (u <= mt.exp(delta_mean / T))

                self.storico_iterazioni.append((
                    n_sol_run[0],
                    round(T, 3),  # temperatura
                    round(stats.mean(obj_old), 2),  # obj corrente
                    round(stats.mean(obj_new), 2),  # obj nuovo
                    round(delta_mean, 3),  # delta
                    accepted,  # accettata?
                    round(best_obj, 2)  # best-so-far
                ))
                # per S,s
                self.storico_policy.append((
                    n_sol_run[0],
                    S,
                    s,
                    accepted
                ))

                if accepted:
                    current_sol = (S, s, i)
                    mean_obj = stats.mean(obj_new)

                    if mean_obj > best_obj:
                        best_sol, best_obj = (S, s, i), mean_obj
                        no_improvement = 0
                        self.fo_storico.append(round(best_obj, 2))

                self.improvements[i][(n_sol_run[1], best_sol)] = round(best_obj,
                                                                       2)  # registriamo la soluzione nel dizionario improvement, può essere la stessa di prima                if no_improvement > n_no_imp:
                if no_improvement > n_no_imp:  # condizionie di chiusura
                    print('OPTIMIZATION ENDS')
                    if not tot_time:
                        return best_sol, best_obj
                    else:
                        return best_sol, best_obj, round(time.perf_counter() - t_start, 2)

            to_print = f'run simulativi fatti: {n_sol_run[1]}, soluzioni totali valutate: {n_sol_run[0]}, soluzione migliore: {best_sol} fo: {best_obj}'
            print(to_print)
            print()  # lasciamo uno spazio
            T = T * T_cooling_rate  # decremento della temperatura

        print('OPTIMIZATION ENDS')
        if not tot_time:
            return best_sol, best_obj
        else:
            return best_sol, best_obj, round(time.perf_counter() - t_start, 2)

    def __call__(self, i_values: Iterable[int], *,
                 # di seguito parmetri opzionali che sono anche attributi dell'oggetto SA
                 T_start: Optional[float] = None, T_end: Optional[float] = None,
                 T_cooling_rate: float = 0.95,  # fattore riduzione della temperatura - cooling geometrico
                 n_days: int = 100,
                 init_acc_prob: float = 0.3,  # probabilità di accettazione iniziale
                 end_acc_prob: float = 0.01,  # probabilità di accettazione finale
                 n_run_era: int = 500,  # numero massimo di repliche per era
                 n_no_imp: int = 150,  # numero di generazioni senza miglioramento
                 deep_check_s_d: tuple[int, int] = (0, 0)) -> list | tuple[list, list]:
        """ Qui si ottimizza su S, s, i. Per farlo:
            - bisogna lanciare l'ottimizzazione di S e s per tutti i valori di i passati in input
            - le soluzioni ottimali vanno salvate in una lista (o dizionario)
            - infine bisognerebbe rivalutare la funzion obiettivo di ciascuna di loro in maniera "profonda"
              quindi ad un'altissima fidelity (molti random seed) per vedere quale sia effettivamente
              la migliore. Per questo usiamo n_deep_check, che genera s seed  e per ciascuno fa d repliche
        
        Oss. In effetti i livelli minimo e massimo di S e di s dipendono anche da i... Per cui andrebbero
             aggiornati prima di lanciare l'ottimizzazione per ciascun i... Per ora non considerato
        """
        # si poteva fare anche con una list comprehension, ma almeno così possiamo plottare qualcosa
        solutions = []
        for i in i_values:
            # azzeriamo anche se non sarebbe necessario, ma evitiamo di portare appresso troppa roba ...
            self.cache: dict[tuple, tuple[float, dict]] = {}
            self.improvements: dict[int, dict[tuple, float]] = {}
            to_print = f'OTTIMIZZAZIONE A LIVELLO I = {i}'
            print(to_print)
            solutions.append(self.optimize_fixed_I(T_start=T_start,
                                                   T_end=T_end,
                                                   T_cooling_rate=T_cooling_rate,
                                                   n_days=n_days,
                                                   init_acc_prob=init_acc_prob,
                                                   end_acc_prob=end_acc_prob,
                                                   n_run_era=n_run_era,
                                                   n_no_imp=n_no_imp,
                                                   tot_time=False))

        if deep_check_s_d == (0, 0): return solutions
        # calcolo preciso della funzione obiettivo delle soluzioni ottimali
        print()
        print('RICALCOLO F.O. DELLE SOLUZIONI OTTIMALI')
        verified_solutions = []
        for solution, *_ in solutions:
            to_print = f'In valutazione {solution}'
            print(to_print)
            result = self.simulate(*solution, seeds=range(999, 999 + deep_check_s_d[0]), n_days=deep_check_s_d[1])
            verified_solutions.append(solution, stats.mean(self.f_obj(result)))

        return solutions, verified_solutions

    def fitness_trend(self, i):
        """ Da rivedere
            una volta fatta l'ottimizzazione ad un livello i
            questa funzione usa i valori memorizzati in 
            self.improvements[i] per mostrare il grafico dell'andamento 
            della fitness (funzione obiettivo) al crescere delle iterazioni
        """

        improv_sortati_it = sorted(self.improvements[i].items())
        x = [it for it, _ in improv_sortati_it]  # iterazioni
        y = [fo for _, fo in improv_sortati_it]  # best_obj

        plt.figure(figsize=(10, 5))
        plt.plot(x, y)

        plt.xlabel("Iterazione")
        plt.ylabel("Funzione obiettivo best so far")
        plt.title(f"Andamento della fitness durante SA (I = {i})")
        plt.grid(True)

        plt.show()

    def grafico_andamento_fo(self):

        it, T, obj_old, obj_new, delta, acc, best = [], [], [], [], [], [], []

        for record in self.storico_iterazioni:
            it.append(record[0])
            T.append(record[1])  # temperatura
            obj_old.append(record[2])
            obj_new.append(record[3])
            delta.append(record[4])
            acc.append(record[5])
            best.append(record[6])

        it_acc = [i for i, a in zip(it, acc) if a]  # lista iterazioni con fo accettata
        of_acc = [n for n, a in zip(obj_new, acc) if a]  # lista della fo accettate

        # lo stesso di sopra ma per le rifiutate
        it_rif = [i for i, a in zip(it, acc) if not a]
        of_rif = [n for n, a in zip(obj_new, acc) if not a]

        plt.figure(figsize=(12, 6))

        plt.scatter(it_rif, of_rif, color="red", s=15, alpha=0.5, label="Rifiutate")
        plt.scatter(it_acc, of_acc, color="blue", s=20, label="Accettate")

        plt.xlabel("Iterazioni")
        plt.ylabel("Funzione obiettivo")
        plt.title("SA – Soluzioni accettate (blu) e rifiutate (rosse)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def grafico_andamento_Politica(self):
        """
        Un grafico che mostri il valore di S e s ad ogni iterazione.
        Se l'iterazione ha proposto una soluzione accettata allora sono colorati di blu, altirmenti di rosso.
        """

        it, S, s, acc = [], [], [], []
        for record in self.storico_policy:
            it.append(record[0])
            S.append(record[1])
            s.append(record[2])
            acc.append(record[3])

        it_acc = [i for i, a in zip(it, acc) if a]  # lista delle iterazioni con accettazione della fo
        S_acc = [S for S, a in zip(S, acc) if a]  # lista delle S con accettazione della fo
        s_acc = [s for s, a in zip(s, acc) if a]  # lista delle iterazioni con accettazione della fo
        it_rej = [i for i, a in zip(it, acc) if not a]  # lista delle iterazioni con rifiuto della fo
        S_rej = [S for S, a in zip(S, acc) if not a]  # lista delle S con rifiuto della fo
        s_rej = [s for s, a in zip(s, acc) if not a]  # lista delle s con rifiuto della fo
        plt.figure(figsize=(10, 5))
        plt.scatter(it_rej, S_rej, color="red", alpha=0.4, s=10, label="S rifiutati")
        plt.scatter(it_acc, S_acc, color="blue", alpha=0.7, s=15, label="S accettati")
        plt.scatter(it_rej, s_rej, color="red", alpha=0.4, s=10, label="s rifiutati")
        plt.scatter(it_acc, s_acc, color="blue", alpha=0.7, s=15, label="s accettati")
        plt.xlabel("Iterazioni Simulated Annealing")
        plt.ylabel("Valore")
        plt.title("Valori di S, s")
        plt.legend()
        plt.grid(True)
        plt.show()


""" ***** ESEMPIO DI CODICE ***** """

if __name__ == "__main__":
    # Parametri target coerenti con la vecchia versione
    target_fr = 0.95

    # 1) Istanzio una SA "dummy" solo per poter usare simulate ed eval_penalty
    f_obj_dummy = objective_function(target_fr=target_fr, daily_penalty=0.0)
    f_ngh = neighbor_function()
    f_fid = check_fidelity

    sa = SA(
        f_obj=f_obj_dummy,
        f_ngh=f_ngh,
        f_fid=f_fid,
        fidelity={1: 4, 2: 8, 3: 16}
    )

    # 2) Policy da testare: qui puoi mettere la tua S,s,I "vecchia"
    #    Se vuoi testare quella teorica, puoi fare:
    _, vals = Theoretical_SsI_Values(
        SA.daily_demand,
        SA.lead_time,
        SA.Cr.H,
        SA.Cr.cost.oc,
        I=1,
        safety_level=1 - SA.stock_out_prob
    )
    S_test = int(vals["S"])
    s_test = int(vals["s"])
    i_test = 1

    pol_test = Policy(
        s=s_test,
        S=S_test,
        I=i_test,
        m_q=50,
        m_qw=5,
        m_rsl=3,
        r_double=False
    )
    print(pol_test)

    # 3) Simulazione con alcuni seed
    seeds = (10, 20, 30)
    n_days = 100

    res = sa.simulate(S=S_test, s=s_test, i=i_test, seeds=seeds, n_days=n_days)

    rev_mean = sum(res['revenues']) / len(res['revenues'])
    fr_mean = sum(res['fill_rate']) / len(res['fill_rate'])

    print("Ricavo medio:", rev_mean)
    print("Fill rate medio:", fr_mean)

    # 4) Calcolo della penalty giornaliera coerente con i parametri di default (vecchia versione)
    pen = eval_penalty(S=S_test, s=s_test, i=i_test, sa=sa, n_days=n_days)
    print("daily_penalty stimata:", pen)

    # 5) Funzione obiettivo con questa penalty
    f_obj = objective_function(target_fr=target_fr, daily_penalty=pen)

    fo_list = f_obj(res)
    fo_mean = sum(fo_list) / len(fo_list)

    # 6) FO "manuale" con rev_mean e fr_mean
    fo_manual = rev_mean - n_days * pen * max(0, target_fr - fr_mean)

    print("FO media (closure):", fo_mean)
    print("FO manuale:", fo_manual)