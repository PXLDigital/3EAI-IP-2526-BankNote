# Banknote Verifier – Echt vs Vals Geld Herkenning

## Groepsleden
- Emiel Mangelschots
- Bjarni Heselmans

---

## Projectdoel
Dit project heeft als doel een systeem te ontwikkelen dat met een **webcam** of **camera via microcontroller** kan bepalen of een bankbiljet **echt of vals** is.
In tegenstelling tot klassieke beeldherkenning focust dit project op **beeldverwerkingstechnieken** zoals edge detection, filtering en textuuranalyse.

---

## 1. Opzet
Het systeem bestaat uit drie hoofdonderdelen:
1. **Beeldverwerking** (Preprocessing, Edge Detection, Filtering)
2. **Analyse** (Edge Density, Feature Extractie)
3. **Beslissing** (Echt / Vals)

De software werd ontwikkeld in **Python** met behulp van de bibliotheken `OpenCV`, `NumPy` en `Matplotlib`.

---

## 2. Preprocessing
Voor elke opname van een bankbiljet worden de volgende stappen uitgevoerd:
1. Omzetten naar **grijswaarden**
2. **Normalisatie** en **ruisreductie**
3. Eventueel **Gaussian blur** of **bilateral filter** om artefacten te verminderen

Doel: een consistente invoer creëren voor de edge-detectie.

---

## 3. Edge Detection & Feature Analyse

### 3.1 Canny Edge Detection
De **Canny**-methode wordt gebruikt om duidelijke contouren te vinden. Deze methode reageert sterk op goed afgelijnde lijnen, zoals die op echte biljetten.

### 3.2 Laplacian Filter
De **Laplacian**-filter accentueert lokale intensiteitsveranderingen en is gevoelig voor kleine texturen.
Deze filter helpt om subtiele drukverschillen in echte biljetten zichtbaar te maken.

### 3.3 Edge Density
De **edge density** berekent het percentage randpixels ten opzichte van het totale oppervlak van het biljet.
- **Echte biljetten** vertonen een gemiddelde en consistente dichtheid.  
- **Valse biljetten** vertonen vaak een hogere dichtheid (door onnatuurlijke texturen of extra ruis).

### 3.4 Gabor Filters
Met **Gabor-filters** worden textuurkenmerken geëxtraheerd in verschillende richtingen en frequenties.
Dit is nuttig om oriëntatiepatronen te onderscheiden die typisch zijn voor echte biljetten.

---

### 3.5 Gecombineerde Edge Detection (Laplacian + Canny)

#### Doel
Om zowel **fijne texturen** als **duidelijke contouren** in biljetten te isoleren, werd een gecombineerde aanpak getest:  
1. **Laplacian filter** – versterkt kleine intensiteitsvariaties en microtexturen.  
2. **Canny edge detection** – detecteert de sterke, consistente randen in dat versterkte beeld.

#### Resultaten
Vergelijking van echte en valse biljetten (voorbeeld: €5):

| Type | Observatie | Interpretatie |
|------|-------------|---------------|
| **5euroReal_edges_combined.png** | Minder dicht netwerk van randen, goed afgelijnde structuren, dunne contouren. | Echt biljet: verfijnde druk en gecontroleerde microstructuren. |
| **5euroFake_edges_combined.png** | Zeer dicht patroon, kruisingen en textuurvulling, ruisachtig effect. | Vals biljet: onnatuurlijke, onregelmatige patronen door inferieure print. |

#### Conclusie
De combinatie van **Laplacian + Canny** vergroot het contrast tussen echte en valse biljetten:
- Laplacian benadrukt de textuurverschillen.  
- Canny filtert de relevante contouren uit deze textuurinformatie.  

Visueel én kwantitatief (hogere edge density bij valse biljetten) bevestigt deze methode dat de **microstructuren van echte biljetten consistenter en minder chaotisch** zijn.

---

## 4. Toekomstige uitbreidingen
- Integratie met **machine learning**-model op basis van textuurfeatures.  
- Real-time analyse via **microcontroller of embedded camera**.  
- Dataset-uitbreiding met verschillende denominaties en belichtingscondities.

---

## 5. Bronnen
- OpenCV Documentation – Image Filtering & Edge Detection  
- Gonzalez & Woods – Digital Image Processing  
- Eigen observaties en testresultaten met Europese bankbiljetten
