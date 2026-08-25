# BioCLIP2 Zero-Shot Classification - 138 Classes

Ce depot contient les fichiers conserves pour l'utilisation et l'evaluation du modele BioCLIP2 zero-shot.

Le modele utilise est charge avec `open_clip` depuis Hugging Face :

```text
hf-hub:imageomics/bioclip-2
```

Il n'y a pas de poids entraines localement dans ce depot.

## Configuration

La configuration utilisee est dans :

```text
meilleur_modele_bioclip2_autonome/model_config.json
```

Prompts utilises :

```text
a photo of {binomial}
a photo of the plant {binomial}
a field photo of the weed species {binomial}
a botanical photo of {binomial}
a close-up photo of the leaves or flowers of {binomial}
```

## Metriques

Les metriques de reference sont dans :

```text
meilleur_modele_bioclip2_autonome/metrics_reference.json
evaluation/final_bioclip2_binomial/metrics.json
```

Valeurs enregistrees :

```text
Classes: 138
Images evaluees: 642
Accuracy top-1: 0.7258566978193146
Accuracy top-5: 0.926791277258567
Balanced accuracy: 0.7151915113871636
F1 macro: 0.6889676400360879
```

## Fichiers

```text
meilleur_modele_bioclip2_autonome/
  predict.py
  classes.txt
  model_config.json
  metrics_reference.json
  requirements.txt
  example_images/

scripts/
  eval_bioclip_zeroshot.py
  eval_bioclip_prompt_experiments.py

evaluation/
  prompt_experiments/
  final_bioclip2_binomial/
```

## Prediction

```powershell
cd meilleur_modele_bioclip2_autonome
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python predict.py "chemin\vers\image.jpg" --top-k 5
```

## Evaluation

Commande correspondant au fichier `evaluation/final_bioclip2_binomial/metrics.json` :

```powershell
python scripts/eval_bioclip_zeroshot.py --data "Bases de donnees photos Dr Taleb" --arch "hf-hub:imageomics/bioclip-2" --output-dir "results_bioclip2_zeroshot_binomial_prompts_all_dataset" --sample all --prompt-mode binomial_plant --batch-size 8 --workers 0 --seed 42
```

Le dataset complet n'est pas inclus dans ce depot.
