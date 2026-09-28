# AAU P0 - Kingdomino

Repository til vores P0-projekt om klassifikation af terræntyper i Kingdomino ved hjælp af HSV og kNN.

## Filer

- `dataset/`
  - Indeholder billederne af Kingdomino-spillepladerne.

- `ground_truth.json`
  - Indeholder den korrekte terræntype for hvert felt på hver spilleplade.

- `knn_implementation.py`
  - Den endelige Python-implementering af vores kNN-model.
  - Indeholder blandt andet HSV-konvertering, medianberegning, feature scaling, valg af k-værdi og evaluering.

- `knn_implementation.ipynb`
  - Jupyter Notebook-version af implementeringen.
  - Bruges til at vise og dokumentere processen mere trin for trin.

- `random_dataset.py`
  - Script til at opdele datasættet i trænings-, validerings- og testdata med et fast seed.

## Kørsel

Installer de nødvendige pakker:

```bash
pip install opencv-python numpy scikit-learn matplotlib
```
Kør implementeringen:
```bash
python knn_implementation.py
```
