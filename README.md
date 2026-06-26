# Detecția automată a bolilor pulmonare din radiografii toracice

Sistem de detecție automată a 13 patologii pulmonare din radiografii toracice, cu clasificarea severității și hărți de activare Grad-CAM, folosind transfer learning pe arhitectura DenseNet-121.

Proiect de licență — Universitatea Babeș-Bolyai, Cluj-Napoca.

## Descriere

Sistemul primește o radiografie toracică și oferă trei tipuri de informații:

1. **Detecția patologiilor** — probabilitatea de prezență pentru fiecare dintre cele 13 patologii
2. **Clasificarea severității** — gradul de severitate (ușoară / moderată / severă) pentru patologiile detectate
3. **Explicabilitate vizuală** — hărți de activare Grad-CAM care evidențiază regiunile pe care modelul s-a concentrat

Sistemul este accesibil printr-o aplicație web construită cu FastAPI.

## Patologiile detectate

Sistemul detectează 13 patologii: lărgire cardiomediastinală, cardiomegalie, atelectazie, consolidare, edem pulmonar, fractură, leziune pulmonară, efuziune pleurală, pneumonie, pneumotorace, dispozitiv de suport, opacitate pulmonară și pleurezie.

> Notă: clasa „dispozitiv de suport" este detectată de model, dar exclusă din rezultatele afișate în aplicație, deoarece setul de date conține prea puține exemple negative pentru ca modelul să distingă fiabil prezența de absența acesteia (vezi secțiunea Limitări).

## Arhitectura

Sistemul utilizează două modele complementare, ambele bazate pe DenseNet-121:

- **Modelul principal de detecție** — backbone DenseNet-121 + Global Average Pooling + cap de probabilitate cu activare Sigmoid (13 ieșiri)
- **Modelul de severitate** — model dedicat care primește caracteristicile imaginii (1024) concatenate cu identitatea patologiei sub forma unui vector one-hot (13), producând clasificarea în 4 clase de severitate

Cele două modele funcționează în cascadă: modelul principal detectează patologiile prezente, iar pentru fiecare patologie detectată, modelul de severitate estimează gravitatea.

## Setul de date

Proiectul folosește setul **NIH ChestX-ray14** (112.120 de radiografii) cu etichetele îmbunătățite **MAPLEZ** (Lanfredi et al., 2025), care oferă adnotări de calitate superioară privind prezența, probabilitatea, severitatea și localizarea patologiilor.

Împărțirea predefinită a datelor:
- Antrenament: 78.506 imagini
- Validare: 8.018 imagini
- Test: 25.596 imagini

## Rezultate

Pe setul de test (25.596 imagini):

| Metrică | Valoare |
|---|---|
| AUC-ROC mediu | 0,8947 |
| Precizie medie | 0,9243 |
| F1 mediu | 0,6914 |

Comparația cu setul de antrenament (AUC 0,8950) confirmă că modelul nu suferă de supraadaptare.

Modelul de severitate atinge o acuratețe de 66,07% pe cazurile cu severitate specificată.


## Instalare

Se recomandă utilizarea unui mediu virtual.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

Instalarea bibliotecilor necesare:

```bash
pip install torch torchvision
pip install fastapi uvicorn python-multipart
pip install opencv-python albumentations
pip install scikit-learn pandas numpy matplotlib
pip install mlflow
```

Pentru rularea pe CPU (fără placă grafică), PyTorch se instalează astfel:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```


## Fișiere neincluse în repository

Din cauza dimensiunilor mari, următoarele nu sunt incluse în repository:

- **Modelele antrenate (`*.pth`)** — fișierele de ponderi ale modelului principal și ale modelului de severitate. Acestea pot fi regenerate prin rularea scripturilor `train.py` și `severity_train.py`.
- **Imaginile setului de date (`data/images/`)** — cele 112.120 de radiografii din setul NIH ChestX-ray14, disponibile public pe site-ul oficial NIH.
- **Fișierele de etichete și baza de date MLflow** — pot fi regenerate din scripturile de procesare.

Pentru a rula proiectul complet, este necesară descărcarea setului NIH ChestX-ray14 și plasarea imaginilor în folderul `data/images/`, urmată de antrenarea modelelor.



## Utilizare

### Pregătirea datelor

```bash
python explore.py     # explorarea datelor brute
python clean.py       # curățarea etichetelor → fișiere *_clean.csv
```

### Antrenament

```bash
python train.py            # antrenarea modelului principal (30 epoci)
python severity_train.py   # antrenarea modelului de severitate (20 epoci)
```

### Evaluare

```bash
python evaluate.py         # metrici pe setul de test
python evaluate_train.py   # metrici pe setul de antrenament
```

### Aplicația web

```bash
uvicorn app.main:app
```

Apoi se accesează `http://127.0.0.1:8000` în browser.

### Monitorizarea experimentelor (MLflow)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Apoi se accesează `http://127.0.0.1:5000`.

## Limitări

- **Validarea intrării** — sistemul nu verifică dacă imaginea încărcată este o radiografie toracică
- **Pragul fix** — detecția folosește un prag unic de 0,5 pentru toate patologiile
- **Deplasarea distribuției** — modelul a fost antrenat doar pe NIH ChestX-ray14; performanța poate scădea pe imagini din alte surse
- **Dispozitiv de suport** — exclus din afișare din cauza numărului insuficient de exemple negative
- **Severitate severă** — dificil de prezis din cauza numărului redus de exemple în setul de date

## Tehnologii

Python, PyTorch, torchvision, Albumentations, FastAPI, MLflow, OpenCV, scikit-learn, pandas, NumPy, matplotlib.

## Note

Acest sistem este un instrument de asistență și nu înlocuiește diagnosticul unui medic radiolog specialist.



- 
