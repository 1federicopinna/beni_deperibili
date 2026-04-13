import random as rn
import statistics as stats
import pandas as pd
import joblib
#from SimulatedAnnealing_v2 import SA, objective_function, neighbor_function, check_fidelity # da decommentare se si vouole generare le matrici di test e train
import ast
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

_modello = None

def _get_model(nome_modello="Mod_reg_poli/modello_pol_fo.pkl"):
    global _modello
    if _modello is None:
        _modello = joblib.load(nome_modello)
    return _modello

def model_predict_2(S, s, i=4, nome_modello="modello_pol_fo.pkl"):
    modello = _get_model()
    X_test = pd.DataFrame([{"S": S, "s": s}])
    return float(modello.predict(X_test)[0])

def aggiornamento_modello():
    global _modello
    _modello = joblib.load("Mod_reg_poli/modello_pol_fo.pkl") 


def creazione_dati(sa: SA, n_sim: int, n_days: int, seed=None, i=4) -> list:
    
    # se non fornisco un seed ne do uno deterministico di 30 valori
    if seed is None:
        seed = list(range(1, 151, 5))

    valori_S = list(range(16417, 23933, 260))  # passo 100
    valori_s = list(range(6567, 13165, 260))

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



def list_to_df(ris: list) -> pd.DataFrame:
    return pd.DataFrame(ris, columns=['S', 's', 'i', 'n_fo', 'fo_list', 'Avg_fo', 'Std_fo'])


def save_df(df: pd.DataFrame):
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

    print("Regressione effettuata: R^2:", r2_score(y_test, y_pred))

def regression_from_csv(csv_path:str=r"Mod_reg_poli/MatriceX_Dati.csv"):
    matrice_reg = pd.read_csv(csv_path, sep=';')
    pesi = calcolo_pesi_OLSW(matrice_X=csv_path)[1]

    model = Pipeline([
        ("polinomiale", PolynomialFeatures(degree=3, include_bias=False)),
        ("lineare", LinearRegression())
    ])

    X_train = matrice_reg[["S", "s"]]
    y_train = matrice_reg["Avg_fo"]
    model.fit(X_train, y_train, lineare__sample_weight=pesi)

    print("Salvo il modello")
    joblib.dump(model, "Mod_reg_poli/modello_pol_fo.pkl")



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
if __name__ == '__main__':
#    daily_penalty = 951.91 # riscritto per non rieseguire il codice
#    n_days = 80  # anche se ho messo I =1 
#    f_obj = objective_function(target_fr=0.95, daily_penalty=daily_penalty)
#    
#    f_ngh = neighbor_function(SM=23933, Sm=16417, sM=13165, sm=6567,
#                              # dq=(-20, -10, 10, 20),  # variazioni di q = (S - s)
#                              dq=(-80, -40, -20, -10, 10, 20, 40, 80),
#                              prob=(0.4, 0.4, 0.2)  # probabilità di agire solo su S, solo su s, su entrambi
#                              )
#    f_fid = check_fidelity
#    
#    sa = SA(f_obj=f_obj, f_ngh=f_ngh, f_fid=f_fid, fidelity={1: 2 ** 2, 2: 2 ** 3, 3: 2 ** 4})  # creiamo l'oggetto sa
#    
#    # Creazione del modello
#    
#    
#    # regressione()
#    #regression_from_csv()
    pass