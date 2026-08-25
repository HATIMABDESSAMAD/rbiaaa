# Meilleur modele actuel: BioCLIP 2 zero-shot

Ce dossier est autonome par rapport au projet principal. Il contient les classes, la configuration gagnante, les metriques de reference et un script d'utilisation sur une image.

Le modele utilise est telecharge automatiquement au premier lancement depuis Hugging Face via `open_clip`:

```text
hf-hub:imageomics/bioclip-2
```

## Installation

Depuis ce dossier:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Exemple d'utilisation

Tester avec l'image d'exemple incluse:

```powershell
python predict.py
```

Tester avec une autre image:

```powershell
python predict.py "C:\chemin\vers\photo.jpg" --top-k 5
```

## Fichiers importants

- `predict.py`: script de prediction.
- `classes.txt`: les 138 classes utilisees par le meilleur resultat.
- `model_config.json`: modele BioCLIP 2 et prompts binomiaux.
- `metrics_reference.json`: score de reference obtenu dans le projet principal.
- `example_images/abutilon_theophrasti_example.JPG`: image de test.

## Performance de reference

Sur le dataset complet:

```text
accuracy_top1: 72.59%
accuracy_top5: 92.68%
balanced_accuracy: 71.52%
f1_macro: 68.90%
classes: 138
images: 642
```
