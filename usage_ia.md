# Usage des outils d'IA générative

Le sujet impose de signaler systématiquement l'usage d'outils d'IA générative.
Chaque usage est consigné ici au moment où il a lieu.

| Date | Membre | Outil | Usage | Partie concernée |
|---|---|---|---|---|
| 05/11/2026 | Axel | Claude (Anthropic) | Rédaction du plan de travail (PDF dans `docs/`) à partir du sujet | Organisation du projet |
| 05/11/2026 | Axel | Claude (Anthropic) | Génération de la structure du dépôt : `config.py`, `main.py`, signatures et docstrings des modules, fonctions utilitaires (`bbox_from_radius`, `haversine_m`, `covered_fraction`, `parse_yes_no`), tests, README | Tous les modules |
| 05/11/2026 | Axel | Claude (Anthropic) | Implémentation de la collecte Mapillary (`mapillary_client.py`) et de ses tests (`tests/test_mapillary_client.py`) | Étape 1 : collecte |
| 05/11/2026 | Axel | Claude (Anthropic) | Implémentation de la détection YOLO (`detection.py`), du rapport (`report.py`), des versions provisoires `--filter none` et `--parked none`, et de leurs tests | Étapes 3 et 4 |
| 25/11/2026 | Axel | Claude (Anthropic) | Implémentation de l'analyse statistique et des graphiques (`analysis.py`) et de ses tests (`tests/test_analysis.py`) | Étape 5 : analyse |
| 25/11/2026 | Axel | Claude (Anthropic) | Préparation des diapositives de la revue de projet du 26/11 | Revue de projet |
| 26/11/2026 | Axel | Claude (Anthropic) | Script de préparation et de regroupement des annotations (`prepare_annotation.py`), consigne d'annotation et tests | Évaluation : jeu annoté |
