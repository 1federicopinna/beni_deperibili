from statistics import mean
from skopt import gp_minimize
from SimulatedAnnealing_v2 import SA, objective_function, neighbor_function, check_fidelity

i = 4
contatore = 0


daily_penalty = 951.91
n_days = 80  # 20*I
f_obj = objective_function(target_fr=0.95, daily_penalty=daily_penalty)
f_ngh = neighbor_function(SM=23933, Sm=16417, sM=13165, sm=6567,
                          dq=(-80, -40, -20, -10, 10, 20, 40, 80),
                          #dq=(-20, -10, 10, 20),  # variazioni di q = (S - s)                              
                          # dq=(-80, -40, -20, -10, 10, 20, 40, 80),
                          # dq=(-80, -40, -20, -10, -5, -2, 2, 5, 10, 20, 40, 80),
                          prob=(0.4, 0.4, 0.2)  # probabilità di agire solo su S, solo su s, su entrambi
                          )
f_fid = check_fidelity
sa = SA(f_obj=f_obj, f_ngh=f_ngh, f_fid=f_fid, fidelity={1: 2 ** 2, 2: 2 ** 3, 3: 2 ** 4})  # creiamo l'oggetto sa

seeds = sa.sds[3]


def f(x):
    global contatore
    contatore += 1
    S, s = x
    print(f"Ciclo {contatore}: S={S}, s={s}")
    results = sa.simulate(S, s, i=i, seeds=seeds, n_days=n_days)
    obj_list = f_obj(results)

    fo_mean = mean(obj_list)
    print(f"Media = {fo_mean}")
    return -fo_mean

# processo di regressione gaussiana - distribuzione di probabilità su funzioni
res = gp_minimize(
    func=f, # funzione obbiettivo da minimizzare
    dimensions=[(16417, 23933),       # dominio di S (nuovo range)
                (6567, 13165)],
    acq_func="EI",  # funzione di acquisizione -  Expected Improvement
    n_calls=80, # numero di volte che simulo per il processo di ottimizzazione
    n_random_starts=5, # prima di iniziare campiono 5 punti casusali
    random_state=1234, # seed per avere stessi punti iniziali
)

S_best, s_best = res.x
fo_best = -res.fun
print("BO 2D soluzione migliore:", S_best, s_best, "fo media migliore:", fo_best)
