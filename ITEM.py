from __future__ import annotations
from typing import Optional


class Item:
    """ un prodotto generico """
    
    kinds:dict[str, int] = {} # tiene traccia di quanti prodotti per tipo sono stati creati
    
    def __init__(self, 
                     gen_time:float, 
                     shelf_life:Optional[int] = None, 
                     *, 
                     kind:str = 'Item'):
        
        self._t = gen_time # tempo di generazione 
        self.tt = [None, None] # tempo d'inserimento a magazzino, e tempo di prelievo
        self._sl = shelf_life # la shelf life iniziale
        self._rsl = shelf_life # la shelf life rimanente
        self._kd = kind
        """ generiamo un id univoco se e solo se la shelf_life non è None.
                            con shelf life None servono solo per fare poi delle copie  """
        if shelf_life is None:
            self._idx = None
        else:
            Item.kinds[kind] = 1 + Item.kinds.get(kind, 0)
            self._idx = Item.kinds[kind] # progressivo numerico
    
    # remaining shelf life    
    def rem_sl(self, t_now:float, in_place:bool = True) -> float|None:
        """ aggiorna e ritorna la shelf life residua 
            in base all'istante effettivo di simulazione t_now
            Osservazione: Se in_place = True il nuovo valore viene registrato """
        try: r = max(0, round(self._sl - (t_now - self._t), 3)) # shelf life originale - tempo trascorso
        except: r =  None  # eccezzione legata al fatto che sl potrebbe essere None
        if in_place: self._rsl = r
        return r 
    
    @property 
    def sl(self):
        """ shelf life """
        return self._sl 
    
    @sl.setter
    def sl(self, sl:float):
        """ in alcuni casi può servire modificare la shelf life... 
                    da fare con cautela """
        self._sl = sl
        self._rsl = sl # si aggiorna anche la residual
       
    @property
    def rsl(self):
        """ residual shelf life """
        return self._rsl
    
    @rsl.setter
    def rsl(self, rsl:float):
        """ modifica in place la residual shelf life """
        self._rsl = rsl
    
    def time_in(self, t_now:Optional[float] = None) -> float|None:
        """ tempo totale a magazzino; 
            Osserazione: per i prodotti che sono ancora a magazzino 
                         bisogna passare t_now 
        """
        f = lambda x, y: round(y - x, 4)
        t1, t2 = self.tt
        t3 = t_now
        try: return f(t1, t2)
        except TypeError: 
            try: return f(t1, t3)
            except: return None
    
    def copy(self, sl:Optional[int] = None) -> Item:
        """ una copia con differente shelf life """
        if sl is None: sl = self._sl
        return Item(gen_time = self._t, shelf_life = sl, kind = self._kd)
    
    def __repr__(self) -> str:
        return repr((''.join((self._kd, "_", str(self._idx))), self.rsl))
    
    
    
    
    