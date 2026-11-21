# Cours de statistique descriptive (document de démonstration)

Ce document est un cours d'exemple, libre de droits, rédigé pour la démonstration de l'assistant RAG.

## 1. Vocabulaire

Une **population** est l'ensemble des individus étudiés. Un **échantillon** est une partie de la population sur laquelle on effectue réellement les mesures. Une **variable** est une caractéristique mesurée sur chaque individu.

On distingue deux grands types de variables :

- les **variables qualitatives** (ou catégorielles), dont les modalités ne sont pas des nombres : couleur des yeux, filière d'études, mention obtenue. Elles peuvent être nominales (sans ordre) ou ordinales (avec un ordre naturel, comme les mentions « passable », « assez bien », « bien », « très bien ») ;
- les **variables quantitatives**, qui prennent des valeurs numériques : âge, taille, note. Elles sont dites discrètes lorsqu'elles prennent des valeurs isolées (nombre d'enfants) et continues lorsqu'elles peuvent prendre toutes les valeurs d'un intervalle (taille en centimètres).

## 2. Indicateurs de tendance centrale

La **moyenne arithmétique** d'une série de n valeurs x1, …, xn est la somme des valeurs divisée par leur nombre : x̄ = (x1 + … + xn) / n. La moyenne est sensible aux valeurs extrêmes : une seule valeur aberrante peut la déplacer fortement.

La **médiane** est la valeur qui partage la série ordonnée en deux groupes de même effectif : 50 % des valeurs lui sont inférieures ou égales. Pour la calculer, on trie les valeurs par ordre croissant. Si n est impair, la médiane est la valeur centrale, de rang (n + 1) / 2. Si n est pair, on prend la moyenne des deux valeurs centrales. La médiane est **robuste** : elle est peu sensible aux valeurs extrêmes. Pour des salaires, on préfère donc souvent la médiane à la moyenne.

Le **mode** est la modalité ou la valeur la plus fréquente. C'est le seul indicateur de tendance centrale utilisable pour une variable qualitative nominale.

## 3. Indicateurs de dispersion

L'**étendue** est la différence entre la plus grande et la plus petite valeur. Elle est simple à calculer mais dépend uniquement des deux valeurs extrêmes.

Les **quartiles** Q1, Q2 et Q3 partagent la série ordonnée en quatre parties de même effectif. Q2 est égal à la médiane. L'**écart interquartile** est IQR = Q3 − Q1 ; il mesure la dispersion des 50 % de valeurs centrales.

La **variance** est la moyenne des carrés des écarts à la moyenne : V = (1/n) × Σ (xi − x̄)². Sur un échantillon, on utilise souvent la variance corrigée, où l'on divise par n − 1 au lieu de n, pour obtenir un estimateur sans biais de la variance de la population.

L'**écart-type** est la racine carrée de la variance : σ = √V. Il s'exprime dans la même unité que les données, ce qui le rend plus facile à interpréter que la variance.

Le **coefficient de variation** est le rapport entre l'écart-type et la moyenne : CV = σ / x̄. Sans unité, il permet de comparer la dispersion de séries exprimées dans des unités différentes.

## 4. Valeurs aberrantes et boîte à moustaches

La **boîte à moustaches** (ou boxplot) représente Q1, la médiane et Q3 par une boîte, et les valeurs extrêmes par des moustaches. Selon la règle de Tukey, une valeur est considérée comme **aberrante** lorsqu'elle est inférieure à Q1 − 1,5 × IQR ou supérieure à Q3 + 1,5 × IQR.

## 5. Statistique bivariée

Pour étudier le lien entre deux variables quantitatives X et Y, on trace d'abord un **nuage de points**.

La **covariance** mesure la variation simultanée des deux variables : cov(X, Y) = (1/n) × Σ (xi − x̄)(yi − ȳ).

Le **coefficient de corrélation linéaire de Pearson** vaut r = cov(X, Y) / (σX × σY). Il est toujours compris entre −1 et 1. Une valeur proche de 1 indique une forte liaison linéaire croissante, une valeur proche de −1 une forte liaison linéaire décroissante, et une valeur proche de 0 une absence de liaison linéaire. Attention : **corrélation ne signifie pas causalité**. Deux variables peuvent être corrélées parce qu'elles dépendent toutes deux d'une troisième variable, appelée facteur de confusion.

La **droite de régression** de Y en X, obtenue par la méthode des moindres carrés, a pour équation y = a x + b, avec a = cov(X, Y) / V(X) et b = ȳ − a x̄. Le **coefficient de détermination** R² est égal à r² dans le cas de la régression linéaire simple : il indique la part de la variance de Y expliquée par le modèle.

## 6. Tableau de contingence et test du khi-deux

Pour deux variables qualitatives, on construit un **tableau de contingence** qui croise leurs modalités. Le **test du khi-deux d'indépendance** compare les effectifs observés aux effectifs théoriques attendus sous l'hypothèse d'indépendance. L'effectif théorique d'une case vaut (total de la ligne × total de la colonne) / effectif total. Le test est fiable lorsque tous les effectifs théoriques sont supérieurs ou égaux à 5.
