from Utilities import Prob_mass_func, Cont_distrib

discr_min_sl: Prob_mass_func = { # Probabilità che un prodotto con slr i venga acquistato da un agente. Da leggere come: solo il 4% della popolazione accetta di comprare un prodotto con sh 1 ecc..
    1: 0.04,
    2: 0.06,
    3: 0.15,
    4: 0.12,
    5: 0.63,
}

discr_pop_sensibile_sconti: Prob_mass_func = { # Probabilità che un agente sia sensibile ad uno sconto.
    True: 0.4,
    False:0.6
}

prob_guarda: float = 0.6 # frequenza della popolazione che guarda la shelf life 

prob_ag_sensibile: float = 0.70 # Probabilità che un agente sia sensibile ad uno sconto.

distri_discount_acceptance: Prob_mass_func = { # Probabilità che un prodotto con slr i venga acquistato da un agente se in sconto.
    1: 0.22,
    2: 0.30,
    3: 0.25,
    4: 0.05,
    5: 0.18,
}


cdf_tabella_sconti: Prob_mass_func = { # probabilità cumulata sconto
    0.10: 0.05,
    0.15: 0.07,
    0.20: 0.12,
    0.25: 0.20,
    0.30: 0.30,
    0.35: 0.43,
    0.40: 0.59,
    0.45: 0.78,
    0.50: 1.00
}

P=3100 # numero totale degli agenti