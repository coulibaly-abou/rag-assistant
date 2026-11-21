# Programmation Python et pandas pour la science des données (document de démonstration)

Ce document est un cours d'exemple, libre de droits, rédigé pour la démonstration de l'assistant RAG.

## 1. Types de base et structures de données

Python est un langage interprété, à typage dynamique : le type d'une variable est déterminé à l'exécution. Les types de base sont `int` (entiers), `float` (nombres à virgule), `str` (chaînes de caractères) et `bool` (`True` ou `False`).

Une **liste** est une séquence ordonnée et **modifiable** d'éléments, notée entre crochets : `notes = [12, 15, 9]`. On ajoute un élément avec `notes.append(17)`. Les indices commencent à 0 : `notes[0]` vaut 12 et `notes[-1]` renvoie le dernier élément.

Un **tuple** est une séquence ordonnée **non modifiable** (immuable), notée entre parenthèses : `point = (3, 4)`. On l'utilise pour des données qui ne doivent pas changer.

Un **dictionnaire** associe des clés uniques à des valeurs : `etudiant = {"nom": "Durand", "age": 20}`. On accède à une valeur par sa clé : `etudiant["nom"]`. La méthode `get` permet de fournir une valeur par défaut si la clé est absente : `etudiant.get("ville", "inconnue")`.

Un **ensemble** (`set`) contient des éléments uniques et non ordonnés. Il est pratique pour supprimer les doublons d'une liste : `set([1, 1, 2])` donne `{1, 2}`.

## 2. Fonctions et compréhensions

Une fonction se définit avec le mot-clé `def` et renvoie un résultat avec `return` :

```python
def moyenne(valeurs):
    return sum(valeurs) / len(valeurs)
```

Une **compréhension de liste** construit une liste en une seule ligne : `carres = [x ** 2 for x in range(10) if x % 2 == 0]` donne les carrés des nombres pairs de 0 à 8.

Les **exceptions** se gèrent avec `try` / `except`. Par exemple, une division par zéro lève une `ZeroDivisionError`, que l'on peut intercepter pour afficher un message d'erreur clair au lieu d'arrêter le programme.

## 3. Environnements virtuels

Un **environnement virtuel** isole les bibliothèques d'un projet. On le crée avec `python -m venv .venv`, puis on l'active (`.venv\Scripts\activate` sous Windows). Les dépendances sont listées dans un fichier `requirements.txt` et installées avec `pip install -r requirements.txt`.

## 4. La bibliothèque pandas

**pandas** est la bibliothèque de référence pour manipuler des données tabulaires en Python. Elle s'importe par convention sous le nom `pd` : `import pandas as pd`.

Les deux structures principales sont la **Series** (une colonne de données indexée) et le **DataFrame** (un tableau à deux dimensions, avec des lignes et des colonnes nommées).

On lit un fichier CSV avec `pd.read_csv("fichier.csv")`. Pour un fichier dont le séparateur est le point-virgule, très courant en France, on précise `pd.read_csv("fichier.csv", sep=";")`.

Pour explorer un DataFrame `df` :

- `df.head()` affiche les 5 premières lignes ;
- `df.shape` donne le nombre de lignes et de colonnes ;
- `df.info()` affiche les types des colonnes et le nombre de valeurs non manquantes ;
- `df.describe()` calcule les statistiques descriptives (moyenne, écart-type, quartiles…) des colonnes numériques.

## 5. Sélection, filtrage et valeurs manquantes

On sélectionne une colonne avec `df["age"]` et plusieurs colonnes avec `df[["nom", "age"]]`. La méthode `loc` sélectionne par **étiquettes** (noms de lignes et de colonnes), tandis que `iloc` sélectionne par **positions** entières.

On filtre les lignes avec une condition booléenne : `df[df["age"] >= 18]`. Plusieurs conditions se combinent avec `&` (et) et `|` (ou), chaque condition étant placée entre parenthèses.

Les valeurs manquantes sont représentées par `NaN`. `df.isna().sum()` compte les valeurs manquantes par colonne. `df.dropna()` supprime les lignes qui contiennent des valeurs manquantes, et `df.fillna(valeur)` les remplace, par exemple par la médiane de la colonne.

## 6. Agrégation et fusion

La méthode `groupby` regroupe les lignes selon une ou plusieurs colonnes puis applique une fonction d'agrégation : `df.groupby("filiere")["note"].mean()` calcule la note moyenne par filière. La méthode `agg` permet d'appliquer plusieurs fonctions à la fois, par exemple `agg(["mean", "std", "count"])`.

Pour combiner deux DataFrames selon une clé commune, on utilise `pd.merge(df1, df2, on="id", how="left")`. Le paramètre `how` vaut `inner` (intersection des clés, valeur par défaut), `left`, `right` ou `outer` (union des clés). Pour empiler des DataFrames les uns sous les autres, on utilise `pd.concat`.
