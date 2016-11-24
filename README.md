# Transfer learning

# getimages.py
Google Bigquery-ből exportált .json fileok alapján tölt le képeket és menti el a szükséges file struktúra szerint.

# train.py
A transfer learninget végző program. A kód nagy része, az órai példa alapján készült. Amit módosítani kellett, az a bináris osztályozás helyett, a kategóriákba sorolás, illetve a kiértékelés létrehozása.
A kiértékelésben minden osztályra számolunk precisiont, recallt, f1-et valamint ezek átlagát, illetve accuracyt.
