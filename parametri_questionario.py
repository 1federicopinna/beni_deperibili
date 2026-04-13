from Utilities import Prob_mass_func

"""
Parametri aggiornati della simulazione (scenario “questionario”):

    - Lead Time (LT):
        Distribuzione triangolare pura continua su [1, 5] giorni, moda = 2.
            - lt = triang(c=0.25, loc=1, scale=4)

    - Shelf Life (SL):
        Distribuzione discreta della shelf life residua all'arrivo:
        disc_sl = {13: 0.1, 14: 0.2, 15: 0.5, 16: 0.2}

    - Shelf life minima accettabile (min_rsl):
        m_rsl = 10 giorni (né il buyer non accetta prodotti con SL < 10).

    - Processo di domanda giornaliera da questionario:
        dd = norm(2506, 113)
        Domanda media giornaliera ≈ 2506 unità/giorno.

    - Domanda per singolo ordine dell'agente (q_dist):
        Distribuzione discreta della quantità per acquisto:
        q_dist = {1: 0.15, 2: 0.45, 3: 0.25, 4: 0.10, 5: 0.05}
        Media 2.45 pezzi per ordine.
        Varianza 6.46

    - Intertempo tra acquisti degli agenti:
        Intertempo esponenziale con media dt_mean = 3 giorni.
        Ogni agente acquista in media ogni 3 giorni (λ ≈ 0.33 ordini/giorno).

    - Costi:
        costo unitario del prodotto (pc)     = 5
        costo fisso di ordinazione (oc)      = 550 per ordine
        costo di smaltimento (dc)            = 0.163 per unità scartata
        costo di perdita di vendita (u)      = 0.05 (percentuale)
        costo di possesso annuo (h)          = 1.095 (percentuale annua)
        sale_price                           = 8 (prezzo di vendita unitario)

    - Politica teorica adottata:
        {'S': 22741, 's': 10211, 'I': 5}

"""






# Probabilità che un prodotto con slr i venga acquistato da un agente. Da leggere come: solo il 4% della popolazione accetta di comprare un prodotto con sh 1 ecc..
discr_min_sl: Prob_mass_func = { 
    1: 0.04,
    2: 0.06,
    3: 0.15,
    4: 0.12,
    5: 0.63,
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


cdf_tabella_sconti: dict[float, float] = { # probabilità cumulata sconto
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