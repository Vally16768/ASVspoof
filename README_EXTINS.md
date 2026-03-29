# Detectarea spoofing-ului vocal în ASVspoof 2019 LA prin feature mining extins și modelare tabulară: document tehnic în stil de lucrare de licență

## Abstract
Creșterea calității sistemelor de sinteză și voice conversion a transformat deepfake-ul audio într-un vector de atac major pentru sistemele ASV (Automatic Speaker Verification), cu impact direct în autentificare biometrică, banking vocal, call-center security și control de acces asistat vocal. În acest context, detectarea robustă a semnalelor spoofed nu mai este o problemă exclusiv de clasificare binară, ci una de optimizare operațională într-un tandem CM+ASV, unde erorile au costuri asimetrice și depind de pragul decizional [1], [2].

Documentul prezintă în formă academică pipeline-ul complet implementat în proiectul curent: indexare dataset ASVspoof 2019 LA, extracție de feature-uri multi-domeniu, materializare automată a combinațiilor, antrenare și evaluare MLP tabular pe fiecare combinație, ranking după `min_tDCF` cu tie-break `accuracy`, plus actualizare automată a artifactelor finale de inferență. Extensia principală față de versiunea inițială constă în integrarea a patru grupuri noi de feature-uri: `LPCC`, `CQCC`, `wav2vec2` și `WavLM`, păstrând compatibilitatea end-to-end între training și inferență prin salvarea transformărilor PCA train-only pentru embedding-uri SSL.

Contribuțiile concrete sunt: (i) formalizarea unui spațiu extins de reprezentări acustice, cepstrale, spectrale, temporale și self-supervised; (ii) strategie de căutare controlată pe 16 combinații derivate formal din best set-ul de referință `AHKLMNO`; (iii) protocol reproductibil orientat pe risc (`min_tDCF`) și trasabilitate completă a artifactelor; (iv) documentarea explicită a performanței finale după rularea completă a benchmark-ului memsafe. La data verificării finale locale (29 martie 2026), pipeline-ul end-to-end a fost finalizat, iar secțiunea de rezultate conține ranking-ul final și modelul câștigător actualizat în `final_model/`.

## 1. Introducere
### 1.1 Context
ASV este sensibil la atacuri de tip TTS (text-to-speech) și VC (voice conversion), iar progresul modelelor generative de voce a redus semnificativ artefactele perceptibile pentru ascultătorul uman. Drept urmare, CM (countermeasure)-urile anti-spoofing trebuie proiectate pentru generalizare pe atacuri nevăzute și evaluate în tandem cu ASV, nu izolat [1], [2], [9].

Competițiile ASVspoof au standardizat această evaluare, introducând protocoale, scoruri ASV de referință și metrice precum `EER` și `min_tDCF`, unde `min_tDCF` aproximează costul operațional minim al sistemului compus CM+ASV [2].

### 1.2 Problema abordată
Problema practică tratată în acest proiect este:

- construirea unui detector robust `bonafide` vs `spoof` pe ASVspoof 2019 LA;
- maximizarea performanței fără pierderea interpretabilității feature-level;
- integrarea coerentă a embedding-urilor SSL în pipeline tabular (PCA train-only, inferență compatibilă 1:1);
- selecția combinației optime pe criteriu operațional (`min_tDCF`) și nu doar pe acuratețe.

### 1.3 Obiective și contribuții
Obiectivele tehnice sunt:

1. extinderea feature set-ului cu `LPCC`, `CQCC`, `wav2vec2`, `WavLM`;
2. menținerea mapării stabile literă-grup și a materializării automate NPZ;
3. căutare pe spațiu restrâns, dar exhaustiv local: `AHKLMNO + P({B,C,D,E})`, adică 16 coduri;
4. actualizare automată a `final_model/` cu winner + metadate + transformări PCA;
5. documentare academică extensivă, cu fundament teoretic, formulare matematică și bibliografie științifică.

Contribuția metodologică este un compromis explicit între interpretabilitatea feature engineering-ului clasic și puterea reprezentărilor self-supervised [5]–[8], [26].

## 2. Stadiul artei
### 2.1 Benchmarks ASVspoof și evaluarea tandem
ASVspoof 2019 a furnizat un benchmark de referință pentru atacuri sintetice/convertite/replay, împreună cu protocoale standardizate [1]. În această paradigmă, metricile de interes nu sunt exclus clasice (`accuracy`, `EER`), ci și cele orientate pe cost operațional în sistem compus, în special t-DCF [2].

ASVspoof 2021 a extins dificultatea către condiții mai diverse și atacuri mai apropiate de scenarii reale, accentuând nevoia de robustețe out-of-distribution [9].

### 2.2 Feature engineering anti-spoofing
Feature-urile cepstrale și spectrale rămân baza multor sisteme robuste:

- MFCC pentru envelope spectrală perceptuală [10];
- CQCC pentru rezoluție log-frecvență și captarea artefactelor de sinteză [3], [4];
- LPCC/LPC pentru modelarea vocal tract-ului și a regularităților de predicție liniară [11].

În paralel, feature-uri temporale și multi-rezoluție (`ZCR`, `RMS`, pitch, wavelets) oferă sensibilitate la micro-structuri și neregularități locale [12], [26].

### 2.3 Modele end-to-end și reprezentări SSL
Arhitecturile end-to-end precum RawNet2 și AASIST au demonstrat performanțe puternice, însă cu cost computațional și complexitate arhitecturală ridicate [5], [6]. Reprezentările SSL, în special wav2vec 2.0 și WavLM, oferă embedding-uri bogate semantic/acustic care pot fi reutilizate în sisteme hibride [7], [8].

Abordarea prezentă se poziționează între cele două extreme: păstrează modelul final simplu (MLP tabular), dar extinde semnificativ reprezentarea prin fuzionarea de descriptori clasici cu embedding-uri SSL comprimate PCA.

## 3. Fundament teoretic
### 3.1 Modelarea semnalului și STFT
Fie semnalul discret $x[n]$. STFT este:

$$
X(m,k)=\sum_{n=0}^{N-1} x[n+mH]w[n]e^{-j2\pi kn/N},
$$

unde $N$ este lungimea ferestrei, $H$ hop size, $w[n]$ fereastra de analiză.

### 3.2 MFCC
MFCC se bazează pe energia pe benzi Mel și transformare DCT [10]:

$$
E_b = \sum_k |X(m,k)|^2 M_b(k),
$$

$$
c_r = \sum_{b=1}^{B} \log(E_b)\cos\left[\frac{\pi r}{B}(b-0.5)\right],\; r=0,\ldots,R-1.
$$

În implementare se folosesc 13 coeficienți, apoi agregare statistică pe timp (`mean`, `std`).

### 3.3 LPC/LPCC
În modelarea LPC [11], fiecare eșantion este aproximat prin combinație liniară de eșantioane anterioare:

$$
x[n]\approx -\sum_{p=1}^{P} a_p x[n-p] + e[n].
$$

LPCC derivă din coeficienții LPC prin recurențe cepstrale (formă standard):

$$
c_n = a_n + \sum_{k=1}^{n-1}\frac{k}{n}c_k a_{n-k},\;1\le n\le P,
$$

cu extensie pentru $n>P$ prin termenii disponibili. În pipeline, vectorul LPCC este extins cu derivate de ordin 1 și 2 (`delta`, `delta-delta`) și agregat `mean/std`.

### 3.4 CQT/CQCC
CQT folosește benzi geometrice în frecvență, cu:

$$
f_k = f_{\min} 2^{k/B},\quad Q=\frac{f_k}{\Delta f_k}.
$$

CQCC aplică log-magnitude pe CQT, resampling uniform și DCT pentru obținerea coeficienților cepstrali [3], [4]. Aceștia sunt apoi completați cu `delta`/`delta-delta` și sumarizați statistic.

### 3.5 Feature-uri temporale și spectrale
ZCR și RMS:

$$
\text{ZCR} = \frac{1}{2N}\sum_{n=1}^{N}|\operatorname{sgn}(x[n]) - \operatorname{sgn}(x[n-1])|,
$$

$$
\text{RMS}=\sqrt{\frac{1}{N}\sum_{n=0}^{N-1}x[n]^2}.
$$

Momente spectrale (pe magnitudini/power):

$$
C = \frac{\sum_k f_k S_k}{\sum_k S_k},\quad
BW = \sqrt{\frac{\sum_k (f_k-C)^2S_k}{\sum_k S_k}},
$$

iar rolloff-ul la 85% energie este frecvența $f_r$ minimă astfel încât:

$$
\sum_{f_k\le f_r}S_k \ge 0.85\sum_k S_k.
$$

Spectral contrast estimează diferența energetică între vârfuri și văi pe subbenzi [26].

### 3.6 Pitch prin autocorelație
Pentru fiecare fereastră, autocorelația:

$$
R_{xx}(\tau)=\sum_n x[n]x[n-\tau],
$$

iar frecvența fundamentală estimată:

$$
f_0 = \frac{f_s}{\arg\max_{\tau\in[\tau_{\min},\tau_{\max}]} R_{xx}(\tau)}.
$$

Pipeline-ul calculează statistici robuste (`pitch_mean`, `pitch_std`) cu tratament explicit al valorilor ne-voiced.

### 3.7 Wavelets (DWT)
DWT proiectează semnalul pe baze scalate/translatate [12]:

$$
x(t)=\sum_k a_{J,k}\phi_{J,k}(t) + \sum_{j=1}^{J}\sum_k d_{j,k}\psi_{j,k}(t).
$$

Implementarea folosește `db4`, nivel 5, iar pentru fiecare subbandă se extrag `mean(abs(c))` și `std(abs(c))`.

### 3.8 Pooling statistic pe timp
Pentru orice descriptor frame-level $u_t$:

$$
\mu = \frac{1}{T}\sum_{t=1}^{T}u_t,\quad
\sigma = \sqrt{\frac{1}{T}\sum_{t=1}^{T}(u_t-\mu)^2}.
$$

Acest pooling reduce dimensiunea și stabilizează variabilitatea locală între utterance-uri cu lungimi diferite.

### 3.9 PCA train-only pentru embedding-uri SSL
Pentru matricea de train $X\in\mathbb{R}^{n\times d}$ (centrată/scalată), covarianța este:

$$
\Sigma = \frac{1}{n-1}X^TX.
$$

Rezolvarea:

$$
\Sigma v_i = \lambda_i v_i,
$$

iar proiecția în spațiul redus:

$$
Z = XV_k,
$$

unde $V_k$ conține primii $k$ vectori proprii [13], [14]. În proiect, $k=128$ per model SSL, cu fit strict pe split-ul `train`.

### 3.10 Clasificare binară, BCE și metrici
MLP optimizează Binary Cross-Entropy:

$$
\mathcal{L}_{BCE}=-\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right].
$$

EER este punctul în care FAR și FRR se intersectează:

$$
\operatorname{EER}=\operatorname{FAR}(\tau^*)=\operatorname{FRR}(\tau^*).
$$

Pentru t-DCF, forma standard utilizată în evaluările ASVspoof este:

$$
\operatorname{tDCF}(\tau)=C_0 + C_1P_{miss}^{cm}(\tau) + C_2P_{fa}^{cm}(\tau),
$$

cu coeficienți dependenți de modelul de cost și de punctul de operare ASV [2]. Criteriul de selecție este $\min_\tau \operatorname{tDCF}(\tau)$.

## 4. Materiale și metode
### 4.1 Dataset și split-uri în setup-ul curent
Path-ul activ detectat din `constants.py` este:

- `database/database/data/asvspoof2019`

Split-uri indexate local:

| Split | Total | Spoof | Bonafide | Spoof (%) | Bonafide (%) |
|---|---:|---:|---:|---:|---:|
| train | 20343 | 18276 | 2067 | 89.84 | 10.16 |
| val | 5037 | 4524 | 513 | 89.82 | 10.18 |
| test | 24844 | 22296 | 2548 | 89.74 | 10.26 |

Observații:

- distribuția este puternic dezechilibrată, similară setup-urilor tipice anti-spoofing;
- în workspace-ul curent `eval.csv` nu este materializat, astfel evaluarea locală pentru acest pipeline rulează pe `train/val/test` indexate.

### 4.2 Configurare grupuri și mapare litere
Maparea stabilă curentă este:

- `A=mfcc`, `B=lpcc`, `C=cqcc`, `D=wav2vec`, `E=wavlm`, `H=chroma`, `K=zcr_rms`, `L=spectral_basic`, `M=spectral_contrast`, `N=pitch`, `O=wavelets`.

Dimensiuni finale pe grup:

| Literă | Grup | Dimensiune |
|---|---|---:|
| A | MFCC | 26 |
| B | LPCC | 180 |
| C | CQCC | 180 |
| D | wav2vec2 + PCA | 128 |
| E | WavLM + PCA | 128 |
| H | Chroma | 12 |
| K | ZCR + RMS | 2 |
| L | Spectral basic | 3 |
| M | Spectral contrast | 7 |
| N | Pitch | 2 |
| O | Wavelets | 12 |

### 4.3 Analiza detaliată a fiecărui grup de feature
#### 4.3.1 A - MFCC (26)
Definiție: aproximare compactă a envelopei spectrale în scala perceptuală Mel [10].

Intuiție acustică: spoof-urile de calitate medie introduc distorsiuni subtile în conturul spectral (formante, tilt spectral), detectabile în distribuția coeficienților cepstrali.

Implementare în proiect: `13` coeficienți/frame, apoi `mean`+`std` pe timp.

Avantaje: robust, bine studiat, cost mic.

Limitări: sensibil la variații de canal/noise și poate rata artefacte non-staționare foarte locale.

#### 4.3.2 B - LPCC (180)
Definiție: cepstru derivat din modelarea LPC [11].

Intuiție acustică: modelul de predicție liniară captează regularități ale tractului vocal; deviațiile introduse de sinteză/VC pot altera stabilitatea coeficienților.

Implementare: 30 coeficienți LPCC, extinși cu delta și delta-delta, apoi pooling statistic; extracția este realizată prin `spafe` [21].

Avantaje: complementar MFCC, bun pentru semnături vocale liniare.

Limitări: sensibil la preprocesare și la variații ale estimării LPC pe segmente scurte.

#### 4.3.3 C - CQCC (180)
Definiție: coeficienți cepstrali extrași pe reprezentare CQT [3], [4].

Intuiție acustică: rezoluția log-frecvență favorizează captarea artefactelor distribuite neuniform spectral, frecvente în atacuri de sinteză neurală.

Implementare: 30 coeficienți CQCC + derivate de ordin 1 și 2 + pooling `mean/std`.

Avantaje: performanță dovedită în anti-spoofing clasic.

Limitări: cost mai ridicat decât MFCC și dependență de parametrizarea CQT.

#### 4.3.4 D - wav2vec2 (128 după PCA)
Definiție: embedding self-supervised antrenat la scară mare pe vorbire brută [7].

Intuiție acustică: encodează nu doar informație spectrală locală, ci și pattern-uri contextuale mai bogate, utile pentru separarea bonafide/spoof.

Implementare: model `facebook/wav2vec2-xls-r-300m` [23], încărcat prin infrastructura `transformers` [25], pooling `mean+std` pe starea latentă, urmat de PCA train-only la 128.

Avantaje: reprezentare expresivă, transferabilă.

Limitări: necesită GPU și resurse de memorie; reducerea PCA poate elimina componente utile dacă distribuția train este îngustă.

#### 4.3.5 E - WavLM (128 după PCA)
Definiție: model SSL cu obiectiv de pretraining orientat către mai multe task-uri speech [8].

Intuiție acustică: are sensibilitate bună la structuri paralingvistice și distorsiuni de generare, complementară față de wav2vec2.

Implementare: model `microsoft/wavlm-large` [24], încărcat prin `transformers` [25], pooling `mean+std`, PCA train-only la 128.

Avantaje: embedding-uri competitive pe task-uri speech variate.

Limitări: cost computațional ridicat și dependență de disponibilitatea CUDA.

#### 4.3.6 H - Chroma (12)
Definiție: distribuția energiei pe cele 12 clase de înălțime (pitch classes).

Intuiție acustică: deși vorbirea nu este strict muzicală, cromaticitatea surprinde structuri armonice utile în separarea semnalelor vocale naturale de cele sintetice.

Implementare: chroma derivat din STFT, mediat temporal.

Avantaje: compact, complementar descriptorilor cepstrali.

Limitări: sensibil la semnale cu armonicitate slabă sau zgomot ridicat.

#### 4.3.7 K - ZCR/RMS (2)
Definiție: activitate temporală simplă prin rată de trecere prin zero și energie medie.

Intuiție acustică: poate evidenția neregularități grosiere ale excitației semnalului.

Implementare: medii globale pe ferestre.

Avantaje: cost minim, bun ca feature auxiliar.

Limitări: capacitate discriminativă modestă izolat.

#### 4.3.8 L - Spectral basic (3)
Definiție: centroid, bandwidth, rolloff.

Intuiție acustică: cuantifică „centrul de masă” spectral și dispersia energetică.

Avantaje: descriptori interpretabili fizic.

Limitări: rezumat prea compact pentru atacuri foarte subtile dacă este folosit singur.

#### 4.3.9 M - Spectral contrast (7)
Definiție: contrast între creste și văi pe subbenzi.

Intuiție acustică: surprinde granularitatea texturii spectrale, sensibilă la artefacte ale vocoderelor și generatoarelor neurale.

Avantaje: complementar MFCC/L.

Limitări: poate depinde de setările de subbenzi și de calitatea semnalului.

#### 4.3.10 N - Pitch (2)
Definiție: statistici de frecvență fundamentală estimate prin autocorelație.

Intuiție acustică: deepfake-urile pot genera contururi prosodice mai puțin naturale în tranziții.

Avantaje: semnal prosodic compact și util în fuziune.

Limitări: estimare instabilă pe porțiuni nevoiced/noisy.

#### 4.3.11 O - Wavelets (12)
Definiție: descriptori multi-rezoluție pe subbenzi DWT [12].

Intuiție acustică: artefactele locale pe mai multe scale temporale/frecvențiale pot fi mai vizibile în baza wavelet decât în descriptori globali.

Avantaje: capturează informație fină pe scale multiple.

Limitări: dependență de wavelet-ul mamă și nivelul de decompoziție.

Notă de implementare: pipeline-ul folosește `librosa` pentru front-end audio [19], `PyWavelets` pentru DWT [20] și `scikit-learn` pentru standardizare/PCA [22].

### 4.4 Combinatorica extinderii: de la AHKLMNO la 16 coduri
Setul de bază istoric este `AHKLMNO`, cu dimensiune:

$$
\dim(AHKLMNO)=26+12+2+3+7+2+12=64.
$$

Extinderea cu litere noi $\{B,C,D,E\}$ produce toate subseturile puterii mulțimii:

$$
\#\mathcal{C} = 2^{|\{B,C,D,E\}|}=2^4=16.
$$

Codurile evaluate și dimensiunile lor:

| Cod | Dimensiune |
|---|---:|
| ABCDEHKLMNO | 680 |
| ABCDHKLMNO | 552 |
| ABCEHKLMNO | 552 |
| ABCHKLMNO | 424 |
| ABDEHKLMNO | 500 |
| ABDHKLMNO | 372 |
| ABEHKLMNO | 372 |
| ABHKLMNO | 244 |
| ACDEHKLMNO | 500 |
| ACDHKLMNO | 372 |
| ACEHKLMNO | 372 |
| ACHKLMNO | 244 |
| ADEHKLMNO | 320 |
| ADHKLMNO | 192 |
| AEHKLMNO | 192 |
| AHKLMNO | 64 |

Rationale metodologic: spațiul restrâns permite evaluare țintită a aportului noilor grupuri, evitând costul unui brute-force global pe toate combinațiile posibile.

### 4.5 Arhitectura modelului de clasificare
Modelul de decizie este MLP tabular:

- Normalization layer (adaptat pe train);
- blocuri Dense: `256 -> 128 -> 64`;
- fiecare bloc: `BatchNorm` [17] + `ReLU` [15] + `Dropout` [18];
- ieșire sigmoid pentru probabilitatea clasei `bonafide`;
- optimizare cu Adam [16], loss BCE.

Alegerea unui MLP (în loc de arhitecturi end-to-end grele) urmărește un compromis între performanță, cost și interpretabilitate la nivel de feature.

### 4.6 Compatibilitatea training-inferință
Pentru grupurile SSL:

- extracția este GPU-mandatory (`ssl_require_gpu=True`), fail-fast dacă CUDA nu este disponibilă;
- PCA este fit-uit doar pe train;
- transformările sunt salvate (`wav2vec_pca.joblib`, `wavlm_pca.joblib`) și propagate în `final_model/transforms/`;
- inferența reface exact aceeași proiecție latentă, evitând mismatch de dimensiune/distribuție.

## 5. Protocol experimental și metrici
### 5.1 Protocol de rulare
Fluxul standard este:

1. `extract` incremental pe grupuri;
2. `combos` pe coduri explicite;
3. `train_all_combos` pe cele 16 coduri;
4. ranking după (`min_tDCF` asc, `accuracy` desc);
5. actualizare automată `final_model/` cu winner.

Pentru constrângeri de memorie, scriptul memsafe rulează în trei trepte de extracție:

- non-SSL + LPCC/CQCC;
- wav2vec (PCA train-only);
- WavLM (PCA train-only);

urmat de benchmark-ul de combinații.

### 5.2 Metrici raportate
- `Accuracy` și `Balanced Accuracy` pentru performanță clasă-globală;
- `EER` pentru pragul FAR=FRR;
- `min_tDCF` pentru costul operațional minim al tandemului CM+ASV [2].

Metodologia de selecție finală folosește `min_tDCF` ca metrică primară, ceea ce este adecvat în scenarii de securitate unde costul fals-acceptării spoof este critic.

### 5.3 Artifacte generate
Pe combinație:

- `metrics.json`, `tdcf_metrics.json`;
- `cm_scores_test.csv`, `cm_scores_eval_format.txt`;
- `predictions.csv`, `classification_report.txt`, `confusion_matrix.png`;
- `best_model.keras`, `final_model.keras`.

Global:

- `results/combinations_accuracy.csv` (summary ordonat);
- `final_model/metadata.json` + metricile winner + transformările PCA.

## 6. Rezultate
### 6.1 Rezultate istorice (baseline inițial)
Conform rezultatelor istorice deja prezente în proiect (`results/combinations_accuracy_filled.csv` și raportul inițial), topul pe `accuracy` a evidențiat combinații dominate de fuziuni în jurul `A`, `H`, `M`, `N`, `O`, cu valori de vârf în jur de 97%.

Top 10 istoric (după `accuracy`):

| Rank | Combo | Accuracy |
|---|---|---:|
| 1 | AHKMNO | 0.972951 |
| 2 | AHLMNO | 0.972508 |
| 3 | AHMO | 0.969691 |
| 4 | AKMO | 0.968765 |
| 5 | AHM | 0.968644 |
| 6 | ALMN | 0.968081 |
| 7 | AHKLMNO | 0.967880 |
| 8 | AHKM | 0.966914 |
| 9 | AHKLMO | 0.966390 |
| 10 | AHLMO | 0.966390 |

Acest istoric justifică alegerea lui `AHKLMNO` ca punct de plecare pentru extinderea curentă cu `B/C/D/E`.

### 6.2 Rezultate extindere (finale)
Benchmark-ul extins a fost finalizat, iar ranking-ul oficial pentru cele 16 combinații candidate este calculat din `results/combinations_accuracy.csv`, ordonat după regula proiectului: `min_tDCF` ascendent, tie-break `accuracy` descendent.

**Tabel 2. Top 10 combinații de feature-uri și rezultate finale (ordonare după min t-DCF)**

| Rank | Combo Code | Accuracy (%) | EER (%) | Min t-DCF |
|---|---|---:|---:|---:|
| 1 | ABCDHKLMNO | 99.988 | 0.000 | 0.062733 |
| 2 | ABDHKLMNO | 99.887 | 0.036 | 0.063069 |
| 3 | ABCDEHKLMNO | 99.960 | 0.040 | 0.063111 |
| 4 | ADHKLMNO | 99.924 | 0.058 | 0.063279 |
| 5 | ACDHKLMNO | 99.936 | 0.039 | 0.063742 |
| 6 | ADEHKLMNO | 99.891 | 0.118 | 0.064246 |
| 7 | ABDEHKLMNO | 99.932 | 0.078 | 0.064290 |
| 8 | ABCEHKLMNO | 99.875 | 0.118 | 0.065047 |
| 9 | ACEHKLMNO | 99.924 | 0.078 | 0.065089 |
| 10 | ACDEHKLMNO | 99.924 | 0.118 | 0.066018 |

**Justificarea valorii `EER = 0.00` pentru combinația câștigătoare (`ABCDHKLMNO`)**

Valoarea `EER = 0.000` este corectă și rezultă din separarea completă a scorurilor CM la prag optim (sweep pe praguri), nu din rotunjire numerică. În `results/ABCDHKLMNO/cm_scores_test.csv`, pe scorul `p_bonafide`, se observă:

- `max(spoof) = 0.8198868`
- `min(bonafide) = 0.8592388`
- gap de separare: `0.8592388 - 0.8198868 = 0.039352`

Există deci un interval de praguri `(0.8198868, 0.8592388)` pentru care `FAR = 0` și `FRR = 0`, ceea ce implică `EER = 0`. În același timp, `accuracy` nu este 100% deoarece în `metrics.json` clasificarea este raportată la pragul fix standard `0.5`, unde apar 3 erori de tip `spoof -> bonafide`.

De asemenea, `min t-DCF` rămâne `0.062733` (nenul) chiar cu `eer_cm = 0`, deoarece t-DCF integrează și componenta ASV/cost model (`Pfa_asv`, `Pmiss_asv`, `Pfa_spoof_asv`) din `tdcf_metrics.json`, nu doar erorile CM.

Observații:

- toate combinațiile din Top 10 includ litera `D` (`wav2vec`), sugerând contribuție sistematică la reducerea costului operațional (`min_tDCF`);
- combinația câștigătoare globală este `ABCDHKLMNO`, validată și în `final_model/metadata.json`;
- pentru setul de test final (`n_test=24844`), diferențele între primele poziții sunt mici în termeni absoluți de t-DCF, dar consistente în ordonare.

### 6.3 Sinteza winner-ului final
Din `final_model/metadata.json` și artifactele de winner (`winner_metrics.json`, `winner_tdcf_metrics.json`):

- Winner final: `ABCDHKLMNO`
- Accuracy: `0.999879` (`99.988%`)
- Balanced Accuracy: `0.999933` (`99.993%`)
- EER: `0.000000` (`0.000%`)
- Min t-DCF: `0.062733`
- Număr exemple test: `24844` (`bonafide=2548`, `spoof=22296`)

Comparativ cu punctul de plecare istoric `AHKLMNO`, extinderea cu noile grupuri a condus la îmbunătățiri substanțiale pe toate metricele raportate în setup-ul final curent.

### 6.4 Placeholders academice pentru figuri și analiză
#### Figura 1 - Distribuția setului de date pe split și clasă
- obiectiv: cuantificarea dezechilibrului de clasă;
- tip: stacked bar chart;
- surse: fișierele `index/*.csv`;
- analiză așteptată: impactul dezechilibrului asupra metricilor și necesitatea `balanced_accuracy`.

#### Figura 2 - Fluxul experimental end-to-end
- obiectiv: trasabilitate metodologică;
- tip: diagramă de proces;
- surse: `asvspoof/cli.py`, `scripts/train_all_combos.py`, `scripts/benchmark_extend_best.py`;
- analiză așteptată: puncte de control unde apar riscuri de inconsistență.

#### Figura 3 - Ranking-ul celor 16 combinații (min_tDCF și accuracy)
- obiectiv: comparație directă între coduri;
- tip: bar/lollipop dual-axis;
- sursă primară: `results/combinations_accuracy.csv` (aceeași sursă utilizată pentru Tabelul 2 final);
- analiză așteptată: trade-off între metrica operațională și acuratețe.

#### Figura 4 - Confusion matrix pentru winner
- obiectiv: interpretare pe clase;
- tip: heatmap 2x2;
- surse: `results/<WINNER>/confusion_matrix.png`, `classification_report.txt`;
- analiză așteptată: tipologia erorilor `spoof->bonafide` vs `bonafide->spoof`.

#### Figura 5 - Curbe ROC/EER și punctul de minim t-DCF
- obiectiv: analiză la nivel de prag;
- tip: ROC + tabel praguri;
- surse: `metrics.json`, `tdcf_metrics.json`;
- analiză așteptată: robustețe la variația pragului și impactul asupra costului operațional.

## 7. Discuții
### 7.1 Interpretarea fuziunii clasice + SSL
Rezultatele istorice susțin importanța fuziunii multi-domeniu. Extensia cu `B/C/D/E` urmărește să injecteze:

- granularitate cepstrală suplimentară (LPCC/CQCC);
- reprezentare latentă contextuală (wav2vec2/WavLM).

În practică, un potențial câștig pe `min_tDCF` depinde de calibrarea distribuției scorurilor și de stabilitatea PCA train-only în raport cu distribuția test.

### 7.2 Limitări
1. Rezultatele sunt finale pentru setup-ul curent, însă reprezintă un singur run per combinație (fără mediere multi-seed), deci robustețea statistică poate fi extinsă ulterior prin repetări controlate.
2. Search-space-ul curent este local (16 coduri), nu global pe toate combinațiile posibile ale celor 11 grupuri.
3. MLP tabular nu exploatează explicit structura temporală secvențială, spre deosebire de arhitecturi end-to-end [5], [6].
4. Sensibilitatea la schimbări de distribuție (alte corpora, alte codec-uri, alte atacuri) necesită validare cross-dataset.

### 7.3 Amenințări la validitate
- validitate internă: dependență de corectitudinea indexării și a protocolului de split;
- validitate externă: transferabilitate limitată fără evaluare pe ASVspoof 2021 sau seturi "in the wild" [9];
- validitate construct: `accuracy` poate fi înșelătoare pe date dezechilibrate, de aceea accentul rămâne pe `min_tDCF`.

### 7.4 Direcții de extensie
- compararea sistematică cu backbone-uri end-to-end și modele de calibrate scoring;
- evaluare multi-seed pentru intervale de încredere;
- calibrare Bayes/DET explicită pentru decizii operaționale;
- ablații dedicate pe fiecare grup nou (`B`, `C`, `D`, `E`) în condiții full-test.

## 8. Concluzii și direcții viitoare
Pipeline-ul actual oferă un cadru robust, reproductibil și extensibil pentru detecția spoofing-ului vocal pe ASVspoof 2019 LA, combinând:

- feature engineering interpretabil;
- embedding-uri SSL de înaltă capacitate;
- selecție orientată pe cost operațional (`min_tDCF`);
- sincronizare strictă training-inferență prin transformări PCA persistente.

Din perspectivă de inginerie experimentală, designul „base + subset expansion” s-a dovedit o strategie eficientă pentru integrarea incrementală a noilor feature-uri, evitând costul combinatorial complet fără a pierde trasabilitatea impactului noilor grupuri.

În setup-ul curent finalizat, câștigul adus de extensia cu `LPCC`, `CQCC`, `wav2vec2` și `WavLM` este confirmat prin winner-ul `ABCDHKLMNO` și prin valorile operaționale obținute pe setul de test complet.

## 9. Abrevieri
- `ASV` - Automatic Speaker Verification
- `CM` - Countermeasure
- `LA` - Logical Access
- `PA` - Physical Access
- `TTS` - Text-to-Speech
- `VC` - Voice Conversion
- `MFCC` - Mel-Frequency Cepstral Coefficients
- `LPCC` - Linear Predictive Cepstral Coefficients
- `CQCC` - Constant-Q Cepstral Coefficients
- `SSL` - Self-Supervised Learning
- `DWT` - Discrete Wavelet Transform
- `PCA` - Principal Component Analysis
- `BCE` - Binary Cross-Entropy
- `EER` - Equal Error Rate
- `FAR` - False Acceptance Rate
- `FRR` - False Rejection Rate
- `t-DCF` - tandem Detection Cost Function

## 10. Bibliografie (stil IEEE)
[1] X. Wang, J. Yamagishi, M. Todisco, H. Delgado, N. Evans, T. Kinnunen, and K.-A. Lee, "ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech," *Computer Speech & Language*, vol. 64, p. 101114, 2020, doi: 10.1016/j.csl.2020.101114.

[2] T. Kinnunen *et al.*, "t-DCF: a Detection Cost Function for the Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification," in *Proc. Odyssey 2018*, 2018, pp. 312-319, doi: 10.21437/Odyssey.2018-44.

[3] M. Todisco, H. Delgado, and N. Evans, "A New Feature for Automatic Speaker Verification Anti-Spoofing: Constant Q Cepstral Coefficients," in *Proc. Odyssey 2016*, 2016, pp. 283-290.

[4] M. Todisco, H. Delgado, and N. Evans, "Constant Q cepstral coefficients: A spoofing countermeasure for automatic speaker verification," *Computer Speech & Language*, vol. 45, pp. 516-535, 2017, doi: 10.1016/j.csl.2017.01.001.

[5] H. Tak, J.-W. Jung, J. Patino, M. Todisco, A. Nautsch, and N. Evans, "End-to-end anti-spoofing with RawNet2," in *Proc. ICASSP 2021*, pp. 6369-6373, doi: 10.1109/ICASSP39728.2021.9414234.

[6] J.-W. Jung, H.-S. Heo, H. Tak, H.-J. Shim, J. S. Chung, B.-J. Lee, H.-J. Yu, and N. Evans, "AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks," in *Proc. ICASSP 2022*, pp. 6367-6371, doi: 10.1109/ICASSP43922.2022.9746952.

[7] A. Baevski, H. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A framework for self-supervised learning of speech representations," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

[8] S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, J. Li, N. Qian, M. Wei, and F. Zhou, "WavLM: Large-scale self-supervised pre-training for full stack speech processing," *IEEE Journal of Selected Topics in Signal Processing*, vol. 16, no. 6, pp. 1505-1518, 2022, doi: 10.1109/JSTSP.2022.3188113.

[9] J. Yamagishi *et al.*, "ASVspoof 2021: accelerating progress in spoofed and deepfake speech detection," *arXiv preprint* arXiv:2109.00537, 2021.

[10] S. B. Davis and P. Mermelstein, "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 28, no. 4, pp. 357-366, 1980, doi: 10.1109/TASSP.1980.1163420.

[11] J. Makhoul, "Linear prediction: A tutorial review," *Proceedings of the IEEE*, vol. 63, no. 4, pp. 561-580, 1975, doi: 10.1109/PROC.1975.9792.

[12] S. Mallat, "A theory for multiresolution signal decomposition: The wavelet representation," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 11, no. 7, pp. 674-693, 1989, doi: 10.1109/34.192463.

[13] K. Pearson, "On lines and planes of closest fit to systems of points in space," *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, vol. 2, no. 11, pp. 559-572, 1901.

[14] H. Hotelling, "Analysis of a complex of statistical variables into principal components," *Journal of Educational Psychology*, vol. 24, no. 6, pp. 417-441, 1933, doi: 10.1037/h0071325.

[15] C. M. Bishop, *Pattern Recognition and Machine Learning*. New York, NY, USA: Springer, 2006.

[16] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. ICLR*, 2015.

[17] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. ICML*, 2015.

[18] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *Journal of Machine Learning Research*, vol. 15, no. 56, pp. 1929-1958, 2014.

[19] B. McFee *et al.*, "librosa: Audio and music signal analysis in Python," in *Proc. SciPy 2015*, pp. 18-25.

[20] G. Lee, R. Gommers, F. Waselewski, K. Wohlfahrt, and A. O'Leary, "PyWavelets: A Python package for wavelet analysis," *Journal of Open Source Software*, vol. 4, no. 36, p. 1237, 2019, doi: 10.21105/joss.01237.

[21] S. Messaoudi, A. Lachiri, and A. D. Adnane, "spafe: Simplified Python audio features extraction," GitHub repository, 2019. [Online]. Available: https://github.com/SuperKogito/spafe

[22] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.

[23] Hugging Face, "facebook/wav2vec2-xls-r-300m" (model card). [Online]. Available: https://huggingface.co/facebook/wav2vec2-xls-r-300m

[24] Hugging Face, "microsoft/wavlm-large" (model card). [Online]. Available: https://huggingface.co/microsoft/wavlm-large

[25] T. Wolf *et al.*, "Transformers: State-of-the-art natural language processing," in *Proc. EMNLP: System Demonstrations*, 2020, pp. 38-45, doi: 10.18653/v1/2020.emnlp-demos.6.

[26] M. R. Kamble, H. B. Sailor, H. A. Patil, and H. Li, "Advances in anti-spoofing: from the perspective of ASVspoof challenges," *APSIPA Transactions on Signal and Information Processing*, vol. 9, e2, 2020, doi: 10.1017/ATSIP.2019.21.

Mențiune finală: după extinderea cu noile feature-uri și finalizarea benchmark-ului, cea mai bună combinație este `ABCDHKLMNO`, cu `accuracy=99.988%`, `balanced_accuracy=99.993%`, `eer=0.000%` și `min_tDCF=0.062733` (n_test=24844).
