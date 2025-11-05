# Sobel Edge – Randdetectie op bankbiljetten

## Doel
Deze module voert **Sobel edge-detectie** uit op de reeds **voorbewerkte** bankbiljetbeelden uit het project *Banknote Verifier*.  
Het doel is om de **randen en microstructuren** van echte en valse biljetten te accentueren.  
Door de randinformatie te analyseren kunnen we later onderscheid maken tussen de scherpe, complexe texturen van echte biljetten en de gladdere structuren van vervalsingen.

---

## Bestandsoverzicht

| Bestand | Omschrijving |
|----------|---------------|
| `Testing.py` | Voert de volledige pijplijn uit: preprocessing + Sobel edge-detectie. |
| `Preprocessing.py` | Zorgt voor kleurconversie, contrastverbetering en ruisonderdrukking. |
| `Output/` | Bevat de resultaten van de preprocessing. |
| `Output/Sobeledge/` | Bevat de beelden met Sobel edge-detectie (onderverdeeld in `real/` en `fake/`). |

---

## Werking van de Sobel Edge Detection

### 1. Theoretische achtergrond
De **Sobel-operator** is een eenvoudige maar krachtige methode om **randen** in een beeld te detecteren.  
Hij berekent de intensiteitsverandering (gradiënt) in twee richtingen:
- **X-richting:** verticale randen (linker- en rechterovergangen)
- **Y-richting:** horizontale randen (boven- en onderovergangen)

Door deze twee richtingen te combineren met de Euclidische norm:

\[
M(x, y) = \sqrt{(G_x^2 + G_y^2)}
\]

ontstaat een randbeeld waarin hoge waarden overeenkomen met sterke contrastovergangen (randen).

---

### 2. Gebruikte OpenCV-functies
```python
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)
```

#### Parameterverklaring:
| Parameter | Waarde | Betekenis / Reden |
|------------|---------|-------------------|
| `gray` | Grijswaardenbeeld van preprocessed afbeelding | De Sobel-filter werkt op luminantie, niet op kleur. |
| `cv2.CV_64F` | 64-bit floating point precisie | Voorkomt dat negatieve gradiëntwaarden worden afgesneden. |
| `dx=1, dy=0` | (voor `sobel_x`) | Detecteert randen in X-richting (verticaal). |
| `dx=0, dy=1` | (voor `sobel_y`) | Detecteert randen in Y-richting (horizontaal). |
| `ksize=3` | Kernelgrootte van 3×3 | Evenwicht tussen detailgevoeligheid en ruisonderdrukking. |
| `cv2.magnitude` | Combineert X- en Y-gradiënten | Berekening van de totale randsterkte. |
| `cv2.normalize` | Schaal naar 0–255 | Maakt het beeld visueel interpreteerbaar en geschikt voor opslag. |

---

### 3. Reden voor parameterkeuze

- **`ksize=3`**  
  Grotere kernels (zoals 5 of 7) maken randen breder maar minder scherp.  
  Een kernel van 3 is ideaal voor bankbiljetten, waar fijne microprintlijnen belangrijk zijn.

- **`sigma` van Gaussian blur in preprocessing**  
  De lichte vervaging in de preprocessing (σ = 0.5) vermindert willekeurige ruis,  
  zodat Sobel zich richt op echte textuurpatronen in plaats van toevallige pixelvariaties.

- **Gebruik van `cv2.CV_64F`**  
  Voorkomt dat negatieve randen worden weggesneden bij conversie naar 8-bit.  
  Pas na normalisatie wordt het beeld omgezet naar `uint8` voor opslag.

---

## Bestandslocaties

| Soort | Locatie | Beschrijving |
|--------|----------|--------------|
| **Preprocessed beelden** | `Output/real/` en `Output/fake/` | Resultaat van CLAHE + Gaussian blur. |
| **Sobel edge-beelden** | `Output/Sobeledge/real/` en `Output/Sobeledge/fake/` | Randdetectieresultaten per klasse. |

---

## Uitvoeren van de pipeline

### 1️⃣ Preprocessing + Sobel edge detectie uitvoeren
Open een terminal in de map  
`3EAI-IP-2526-BankNote\Software\Testing\SobelEdge\`  
en voer uit:

```bash
python Testing.py
```

### 2️⃣ Resultaten bekijken
Na voltooiing verschijnen de resultaten in:
```
Output/
├── real/
├── fake/
└── Sobeledge/
    ├── real/
    └── fake/
```

Je kunt de Sobel-beelden openen met een standaard image viewer of in Python weergeven via:
```python
import cv2
cv2.imshow("Sobel Edge", cv2.imread("Output/Sobeledge/real/5euro_processed_sobel.png"))
cv2.waitKey(0)
```

---

## Interpretatie van de resultaten

- **Echte biljetten:** tonen een dichte structuur van scherpe, fijne lijnen.  
- **Valse biljetten:** vertonen zachtere randen, minder microstructuur en lagere edge-dichtheid.  
- Door deze verschillen kwantitatief te analyseren (bijv. via edge-density),  
  kunnen biljetten worden geclassificeerd als “waarschijnlijk echt” of “waarschijnlijk vals”.

---

## Toekomstige uitbreidingen

- Automatische **edge-density analyse** (percentage edge-pixels per oppervlak).  
- **Histogram van gradiëntrichtingen (HOG)** toevoegen voor meer detail.  
- **Thresholding** op Sobel-resultaten om binaire randen te verkrijgen.  
- Integratie met de frequentiedomeinanalyse (FFT) voor gecombineerde classificatie.

---

**Auteur(s):**  
- Emiel Mangelschots  
- Bjarni Heselmans  

**Project:** *Banknote Verifier – Echt vs Vals Geld Herkenning*  
**Versie:** v1.0  
**Laatste update:** 2025-11-05
