199 strings in 90 sign-intent groups; v2 is scored by 5-fold cross-validation grouped by intent (a sign and its translations are never split). Cells: accuracy / balanced accuracy (mean of the two class recalls; '-' when the subset has one class).

| subset (n, important) | always 'important' | keyword rules | learned v1 (English templates) | learned v2 (grouped CV) |
|---|---|---|---|---|
| all (199, 160) | 0.80 / 0.50 | 0.32 / 0.57 | 0.63 / 0.71 | 0.86 / 0.84 |
| English taxonomy (83, 76) | 0.92 / 0.50 | 0.37 / 0.66 | 0.78 / 0.82 | 0.87 / 0.80 |
| non-English (pt / es / fr) (108, 84) | 0.78 / 0.50 | 0.22 / 0.50 | 0.49 / 0.60 | 0.85 / 0.81 |
| lang:en (91, 76) | 0.83 / 0.50 | 0.43 / 0.66 | 0.80 / 0.85 | 0.88 / 0.87 |
| lang:pt (36, 28) | 0.78 / 0.50 | 0.22 / 0.50 | 0.44 / 0.51 | 0.86 / 0.78 |
| lang:es (36, 28) | 0.78 / 0.50 | 0.22 / 0.50 | 0.58 / 0.69 | 0.89 / 0.88 |
| lang:fr (36, 28) | 0.78 / 0.50 | 0.22 / 0.50 | 0.44 / 0.60 | 0.81 / 0.79 |

Recall of important / not-important strings in the non-English subset: always 'important' 1.00 / 0.00; keyword rules 0.00 / 1.00; learned v1 (English templates) 0.41 / 0.79; learned v2 (grouped CV) 0.88 / 0.75
