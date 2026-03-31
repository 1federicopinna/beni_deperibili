import random as rn
import statistics as stats
import pandas
import pandas as pd
import joblib
from SimulatedAnnealing import SA, objective_function, neighbor_function, check_fidelity
import ast
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# def creazione_dati_old(sa: SA, n_sim: int, n_days: int, seed=None) -> list:
#    """
#    genera i dati di S, s, q, i(fisso=4), media della funzione obbiettivo, deviazione standard della funzione
#    obbiettivo. La media e la deviazione standard sono basate su 20 prove simulative con la stessa politica (S, s, i).
#    """
#    i = 4
#    if seed is None:
#        seed = list(range(1, 151, 5))  # default riproducibile
#
#    cont_S = uniform(loc=600, scale=300)  # continuo
#    disc_S = discr_cont_distrib(cont_S, list(range(600, 901, 10)))  # discreto step=5
#
#    cont_s = uniform(loc=200, scale=220)  # continuo
#    disc_s = discr_cont_distrib(cont_s, list(range(200, 421, 10)))  # discreto step=5
#
#    ris = []
#    for j in range(n_sim):
#        S = gen_random_val(disc_S)
#        s = gen_random_val(disc_s)
#        results = sa.simulate(S, s, i, seed, n_days=n_days)
#        # {'revenues': [], 'stock_out_prob': [], 'fill_rate': [],
#        # 'average_oh': [], 'I_triggered': [], 'lost': [], 'n_days': n_days}
#        of_corr = sa.f_obj(results)
#        len_of_corr = len(of_corr)
#        of_curr_avg = stats.mean(of_corr)
#        of_curr_std = stats.stdev(of_corr)
#        ris.append((S, s, i, len_of_corr, of_corr, of_curr_avg, of_curr_std))
#        print(f"Processato il ciclo n: {j}. Risultati fo:\nValori: {of_corr}, \n media: {of_curr_avg}, \n deviazione "
#              f"standard: {of_curr_std}")
#
#    return ris


def creazione_dati(sa: SA, n_sim: int, n_days: int, seed=None) -> list:
    i = 4
    # se non fornisco un seed ne do uno deterministico di 30 valori
    if seed is None:
        seed = list(range(1, 151, 5))

    valori_S = list(range(600, 901, 10))
    valori_s = list(range(200, 421, 10))

    rng = rn.Random(str(seed)) # str(seed) serve ad usare tutti i valori della lista seed per la generazione di n
    # casuali genera l'hash della stringa
    tutte = [(S, s) for S in valori_S for s in valori_s]  # sono tutte le combinazioni delle coppie S,s che
    # definiscono lo spazio dei parametri

    # se il numero delle simulazioni richieste è maggiore della lunghezza dello spazio dei parametri allora ValueError
    if n_sim > len(tutte):
        raise ValueError(f"n_sim={n_sim} troppo alto: max combinazioni = {len(tutte)}")

    # seleziono un numero n_sim di coppie dallo spazio dei parametri tutte
    coppie = rng.sample(tutte, n_sim)  # uniche + riproducibili

    # simulo per ogni coppia dentro la tupla coppie e inserisco tutto dentor ris
    ris = []
    for j in range(n_sim):
        S = coppie[j][0]
        s = coppie[j][1]

        results = sa.simulate(S, s, i, seed, n_days=n_days)
        of_corr = sa.f_obj(results)

        len_of_corr = len(of_corr)
        of_curr_avg = stats.mean(of_corr)
        of_curr_std = stats.stdev(of_corr)

        ris.append((S, s, i, len_of_corr, of_corr, of_curr_avg, of_curr_std))

        print(
            f"Processato il ciclo n: {j}. Risultati fo:\n"
            f"Valori: {of_corr}, \n media: {of_curr_avg}, \n deviazione standard: {of_curr_std}"
        )

    return ris


def list_to_df(ris: list) -> pandas.DataFrame:
    return pd.DataFrame(ris, columns=['S', 's', 'i', 'n_fo', 'fo_list', 'Avg_fo', 'Std_fo'])


def save_df(df: pandas.DataFrame):
    df.to_csv('Mod_reg_poli/MatriceX_Dati.csv', index=False, sep=";")


def calcolo_pesi_OLSW(matrice_X=r"Mod_reg_poli/MatriceX_Dati.csv"):
    df = pd.read_csv(matrice_X, sep=";")

    # Evita divisioni per zero (se Std_fo = 0, metto 1)
    df.loc[df["Std_fo"] == 0, "Std_fo"] = 1

    # Varianza della media
    df["var_mean"] = (df["Std_fo"] ** 2) / df["n_fo"]

    # Pesi = inverso della varianza della media
    weights = 1 / df["var_mean"]

    return df, weights


def preparazione_dati(tipologia: str = "full"):
    def build_dataset(n_sim, n_days, seeds, tipo=""):
        print(f"--------- Creo il DF di {tipo} ---------")
        lista = creazione_dati(sa, n_sim=n_sim, n_days=n_days, seed=seeds)
        df = list_to_df(lista)
        X = df[["S", "s"]]
        y = df["Avg_fo"]
        return X, y, df

    tipologia = tipologia.strip().lower()

    if tipologia == "test":
        X, y, df = build_dataset(200, 80, range(1001, 1101, 5), tipo="test")
        return X, y

    # se la tipologia non è test finisce qua: i seed sono diversi per non avere risultati uguali tra train e test
    X_train, y_train, df_train = build_dataset(200, 80, range(1, 101, 5), tipo="train")
    save_df(df_train)
    X_test, y_test, _ = build_dataset(200, 80, range(1001, 1101, 5), tipo="test")

    return X_train, y_train, X_test, y_test


def regressione():
    X_train, y_train, X_test, y_test = preparazione_dati("full")
    pesi = calcolo_pesi_OLSW()[1]

    model = Pipeline([
        ("polinomiale", PolynomialFeatures(degree=3, include_bias=False)),
        ("lineare", LinearRegression())
    ])

    # model.fit(X_train, y_train)
    model.fit(X_train, y_train, lineare__sample_weight=pesi)

    print("Salvo il modello")
    joblib.dump(model, "Mod_reg_poli/modello_pol_fo.pkl")

    y_pred = model.predict(X_test)

    print("R^2:", r2_score(y_test, y_pred))


def test_modello_R2(nome_modello="modello_pol_fo.pkl"):
    modello_caricato = joblib.load(nome_modello)
    X_test, y_test = preparazione_dati("test")

    y_pred = modello_caricato.predict(X_test)

    print("R^2:", r2_score(y_test, y_pred))


def model_predict(S, s, i=4, nome_modello="modello_pol_fo.pkl"):
    modello_caricato = joblib.load("Mod_reg_poli/modello_pol_fo.pkl")

    X_test = pd.DataFrame([{"S": S, "s": s}]) # devo creare un df con le colonne S s perché il pkl è stato creato così
    y_pred = modello_caricato.predict(X_test)

    return float(y_pred[0])


def aggiornamento_modello(df_path=r"Mod_reg_poli\MatriceX_Dati.csv", model_path=r"Mod_reg_poli\modello_pol_fo.pkl"):
    df = pd.read_csv(df_path, sep=";", converters={'fo_list': ast.literal_eval})
    X = df[["S", "s"]]
    y = df["Avg_fo"]
    model = joblib.load(model_path)
    pesi = calcolo_pesi_OLSW()[1]
    # 7) rifit del modello
    model.fit(X, y, lineare__sample_weight=pesi)
    # 8) risalvo modello
    joblib.dump(model, model_path)


def aggiornamento_X_old(S, s, i, fo, df_path=r"Mod_reg_poli\MatriceX_Dati.csv"):
    # Carico il file csv del modello trainato e converto il valore della colonna fo_list con la lista dei fo in lista
    df = pd.read_csv(df_path, sep=";", converters={'fo_list': ast.literal_eval})
    check_riga = (df['S'] == S) & (df['s'] == s) & (df['i'] == i)
    # S;s;i;n_fo;fo_list;Avg_fo;Std_fo
    if check_riga.any():
        indice_riga = df.index[check_riga][0]
        df.at[indice_riga, 'fo_list'].append(fo)
        lista_fo_aggiornata = df.at[indice_riga, 'fo_list']
        df.at[indice_riga, 'n_fo'] += 1
        df.at[indice_riga, 'Avg_fo'] = stats.mean(lista_fo_aggiornata)
        df.at[indice_riga, 'Std_fo'] = stats.std(lista_fo_aggiornata)
    else:
        if isinstance(fo, list):
            nuova_lista = fo
        else:
            nuova_lista = [fo]
        n_fo = len(nuova_lista)
        media_iniziale = stats.mean(nuova_lista)
        # La deviazione standard si calcola solo se abbiamo più di un valore, altrimenti è 0
        deviazione_iniziale = stats.std(nuova_lista) if n_fo > 1 else 0
        # C. Creiamo un dizionario che rappresenta la riga
        dati_nuova_riga = {
            'S': S,
            's': s,
            'i': i,
            'n_fo': n_fo,
            'fo_list': [nuova_lista],  # Nota: lo mettiamo in una lista per pandas
            'Avg_fo': media_iniziale,
            'Std_fo': deviazione_iniziale
        }
        # D. Trasformiamo il dizionario in un piccolo DataFrame di una riga
        df_nuova_riga = pd.DataFrame(dati_nuova_riga)
        # E. Incolliamo (concateniamo) la nuova riga al DataFrame principale
        # ignore_index=True serve per far sì che la nuova riga prenda il numero successivo (es. 101)
        df = pd.concat([df, df_nuova_riga], ignore_index=True)
    df.to_csv(df_path, sep=";", index=False)


def aggiornamento_X(S, s, i, fo, df_path=r"Mod_reg_poli\MatriceX_Dati.csv"):
    # Abstract Syntax Tree - ast.literal_eval(stringa) prende una stringa la interpreta solo se contiene un valore
    # letterale Python valido. Serve per convertire fo_list da stringa a lista
    df = pd.read_csv(df_path, sep=";", converters={'fo_list': ast.literal_eval})

    # condizione pd.Series booleani che mi dà true se trova quella tupla Ssi nella matrice X
    check_riga = (df['S'] == S) & (df['s'] == s) & (df['i'] == i)

    # controllo se ho almeno un elemento di check riga, cioè se la combinazione Ssi è trovata
    if check_riga.any():
        indice_riga = df.index[check_riga][0] # recupero indice

        lista = df.at[indice_riga, 'fo_list'] # e la lista relativa all'indice

        # qua controllo se fo in input è una lista e se lo è devo usare extend per aggiungerla alla lista della X
        if isinstance(fo, list):
            lista.extend(fo)
        else:
            # altrimenti lo aggiungo seplicemente con append
            lista.append(fo)

        df.at[indice_riga, 'fo_list'] = lista
        df.at[indice_riga, 'n_fo'] = len(lista)
        df.at[indice_riga, 'Avg_fo'] = stats.mean(lista)
        df.at[indice_riga, 'Std_fo'] = stats.stdev(lista) if len(lista) > 1 else 0.0

    else:
        nuova_lista = fo if isinstance(fo, list) else [fo]

        dati_nuova_riga = {
            'S': S,
            's': s,
            'i': i,
            'n_fo': len(nuova_lista),
            'fo_list': nuova_lista,
            'Avg_fo': stats.mean(nuova_lista),
            'Std_fo': stats.stdev(nuova_lista) if len(nuova_lista) > 1 else 0.0
        }

        df_nuova_riga = pd.DataFrame([dati_nuova_riga])
        df = pd.concat([df, df_nuova_riga], ignore_index=True) # aggiungo nuova lista al df princiale

    df.to_csv(df_path, sep=";", index=False) # indice rimosso


####### AVVIO CODICE #######
daily_penalty = 1380  # riscritto per non rieseguire il codice
n_days = 80  # 20*I
f_obj = objective_function(target_fr=0.95, daily_penalty=daily_penalty)

f_ngh = neighbor_function(SM=900, Sm=600, sM=420, sm=200,
                          # dq=(-20, -10, 10, 20),  # variazioni di q = (S - s)
                          dq=(-80, -40, -20, -10, 10, 20, 40, 80),
                          prob=(0.4, 0.4, 0.2)  # probabilità di agire solo su S, solo su s, su entrambi
                          )
f_fid = check_fidelity

sa = SA(f_obj=f_obj, f_ngh=f_ngh, f_fid=f_fid, fidelity={1: 2 ** 2, 2: 2 ** 3, 3: 2 ** 4})  # creiamo l'oggetto sa

# Creazione del modello
# regressione()
# test_modello(nome_modello="modello_pol_fo.pkl")
