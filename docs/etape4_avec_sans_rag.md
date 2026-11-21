# Étape 4 — Validation : 5 questions avec et sans RAG

Critère de validation du cahier des charges : les questions donnent des réponses fidèles au cours, avec leurs sources, ce que le LLM seul ne fait pas.

- **Configuration :** `llama3.2:3b`, `nomic-embed-text`, chunks de 300 mots (chevauchement 50), top-k = 5, température 0,1, processeur seul.
- **Commande :** `python -m eval.run_eval --questions eval/questions_etape4.csv --mode no-rag` puis `--mode rag`.

| Question | Sans RAG (LLM seul) | Avec RAG | Verdict |
|---|---|---|---|
| Pourquoi divise-t-on par n − 1 dans la variance corrigée ? | Formule inventée qui mêle corrélation et deux variables | « Pour obtenir un estimateur sans biais de la variance de la population » | ✅ corrigé |
| Quand une valeur est-elle aberrante selon la règle de Tukey ? | « À plus de 3 × IQR de la médiane » (faux) | « Inférieure à Q1 − 1,5 × IQR ou supérieure à Q3 + 1,5 × IQR » | ✅ corrigé |
| Valeur par défaut de `how` dans `pd.merge` ? | `'outer'` (faux) | `inner` | ✅ corrigé |
| À quoi sert la formule de Bayes ? | Réponse générale et vague | « Inverser un conditionnement », avec l'exemple médical du cours | ✅ plus fidèle au cours |
| Que dit la règle des 68-95-99,7 ? | Juste sur les pourcentages, mais la présente à tort comme une règle de détection des valeurs aberrantes | Pourcentages corrects, avec la précision 1,96 tirée du cours | ✅ plus fidèle au cours |

## Conclusion

- **Sans RAG**, le LLM invente : 3 réponses sur 5 sont fausses (formule fantaisiste, mauvais seuil de Tukey, mauvaise valeur par défaut de `how`).
- **Avec RAG**, les 5 réponses sont correctes et reprennent les formulations du cours. Chaque réponse est accompagnée de ses sources (fichier et page).
- **Temps de réponse sur processeur** : médiane de 24 s sans RAG, contre 149 s avec RAG. Le contexte d'environ 1 500 mots à lire par le LLM coûte cher. C'est un axe d'amélioration pour l'étape 7 : réduire top-k ou la taille des chunks, ou utiliser un modèle plus petit.
- **Limite observée** : la recherche ramène parfois des passages hors sujet en plus du bon passage (un extrait du cours Python pour la question sur Bayes). Le LLM les a ignorés ici, mais une recherche hybride ou un re-ranking (étape 7) pourrait les filtrer.
