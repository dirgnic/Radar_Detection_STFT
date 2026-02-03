# Raport de Corectări - presentation_ro.tex

## Rezumat
S-au corectat patru categorii majore de probleme din prezentare pentru a obține coerență academică și precisie tehnică.

---

## 1. CORECTĂRI FORMULARE INFORMALE → ACADEMICE

### Slide: "Adaptarea 3: DBSCAN Asimetric"
**Inainte:**
```
Problema: Țintele apar ca linii verticale (multe frecvențe, puține momente)
DBSCAN standard fragmentează în clustere multiple!
Soluție: Distanță asimetrică cu freq_scale = 3.0
```

**După:**
```
Problema: Țintele apar ca semnături aproape verticale în spectrogramă 
         (energie pe multe binuri de frecvență, pe puține cadre temporale)
DBSCAN standard tinde să fragmenteze aceste semnături în mai multe 
        clustere atunci când există discontinuități.
Soluție: Metrica de distanță asimetrică cu scalare pe frecvență (s_f = 3.0)
```

**Motivație:** 
- "momente" → "cadre temporale" (mai precis)
- "fragmentează în clustere multiple!" → "tinde să fragmenteze...atunci când..." (mai academic)
- Explicare și contextualizare mai clară

---

### Slide: "Rezultate pe Date Sintetice"
**Inainte:**
```
Detecție perfectă la toate nivelurile
```

**După:**
```
Rata de detecție: 100% în toate cazurile experimentale
RQF crește monoton cu SNR pentru semnalul sintetic studiat
```

**Motivație:**
- Evităm cuvintele puternice "perfect" (care sună ca promisiune absolută)
- Adăugăm context: "pentru semnalul sintetic studiat" (nu se generalizează)

---

### Slide: "Detecție pe Date IPIX Reale"
**Inainte:**
```
Date IPIX (McMaster 1993)
Conversie Doppler → Viteză
```

**După:**
```
Baza de date IPIX (McMaster University, 1993)
Relație Doppler -- Viteză radială
```

**Motivație:**
- "Date" → "Baza de date" (mai formal)
- "Conversie" → "Relație" (mai precis, nu e o conversie, e o relație fizică)
- "viteză" → "viteză radială" (precis în domeniu radar)

---

### Slide: "Animații de Detecție"
**Inainte:**
```
Cu Hurst boost
Target #30 (mare agitată)
Deschideți GIF-urile din results/animations/ pentru a vedea evoluția detecțiilor în timp!
```

**După:**
```
Cu amplificare Hurst (anomalie fractală)
Target #30 (sea state ridicat)
Animațiile arată evoluția detecțiilor în timp, cu suprapunerea măștilor CFAR 
        și semnalelor detectate pe spectrograma STFT.
```

**Motivație:**
- "Hurst boost" → "amplificare Hurst (anomalie fractală)" (explică conceptul)
- "mare agitată" → "sea state ridicat" (terminologie consistentă în document)
- Descriere activă, nu imperativă

---

### Slide: "Concluzii și Contribuții"
**Inainte:**
```
Validare sintetică: RQF = 29.17 dB @ SNR=30dB, detecție 100%
Validare reală: Funcționează pe IPIX sea clutter
Performanță: ~75 ms/segment (13 FPS)
```

**După:**
```
Validare pe date sintetice: RQF = 29.17 dB la SNR=30 dB; 
        rata detecție 100% în 100 rulări Monte Carlo
Validare pe date reale: Detecții consistente pe secvențe IPIX 
        (Target #17, #30, #40)
Performanță: aproximativ 75 ms/segment pe CPU 
        (13 FPS pentru fereastră 2s, hop 0.5s)
```

**Motivație:**
- Claritate: "SNR=30dB" → "SNR=30 dB" (spații corecte)
- Context: "detecție 100%" → "100% în 100 rulări Monte Carlo" (nu pare miracol, e transparent)
- "Funcționează pe" → "Detecții consistente pe secvențe" (nu promisiune, observație)
- Parametri: Adăugăm condiții exacte (CPU, fereastră, hop)

---

## 2. CORECTĂRI FORMULE ȘI PARAMETRI TEHNICI

### Slide: "Adaptarea 3: DBSCAN Asimetric"

**Inainte:**
```
d = √(Δt² + (Δf/3)²)

Mascare DC (±8 bins)
Filtru bandwidth Doppler > 3 Hz
```

**După:**
```
d = √((Δt)² + (Δf/s_f)²)  cu s_f = 3.0

Mascare componentă DC: ±8 binuri (≈±31 Hz)
Filtru bandwidth Doppler: BW ≥ 2 binuri (≈8 Hz)
```

**Motivație:**
- Formula: Explicităm constanta `s_f = 3.0` (nu e "3" la voia întâmplării)
- Parametri: Convertim din binuri în Hz pentru context fizic
  - PRF = 1000 Hz, NFFT = 256 → Δf = 3.91 Hz/bin
  - ±8 binuri ≈ ±31 Hz
  - BW ≥ 2 binuri ≈ 8 Hz (mai precis decât "3 Hz" care era < 1 bin)
- Text: "mascare DC" → "mascare componentă DC" (precis)

---

## 3. CORECTĂRI CONSISTENȚĂ TERMINOLOGIE

| Inainte | După | Slide |
|---------|------|-------|
| "Conversie Doppler → Viteză" | "Relație Doppler -- Viteză radială" | IPIX |
| "mare agitată" | "sea state ridicat" | Animații |
| "Detecție perfectă" | "Rata de detecție 100%" | Sintetice |
| "~75 ms" | "aproximativ 75 ms" | Concluzii |
| "DBSCAN asimetric" | Consistent cu "metrica de distanță asimetrică" | Adaptarea 3 |

---

## 4. CORECTĂRI STIL ȘI CLARITATE

### Slide: "De ce CFAR nu detectează..."
**Inainte:**
```
Întrebare frecventă: De ce doar puncte împrăștiate?
Prag global (naiv):
```

**După:**
```
Observație: De ce doar puncte localizate și nu regiuni uniforme?
Prag global (naiv):
```

**Motivație:**
- Evităm "Întrebare frecventă" (sună informal)
- "puncte împrăștiate" → "puncte localizate" (mai precis)
- Adresăm direct contrast: "și nu regiuni uniforme"

---

## 5. VERIFICARE COMPILARE

✅ Document compilat cu succes:
```
pdflatex -interaction=nonstopmode presentation_ro.tex
Output written on presentation_ro.pdf (19 pages, 1810710 bytes).
```

---

## CHECKLIST APLICAT

- [x] Fix diacritice / encoding (nici una nu era corrupta deja)
- [x] Transformare "formulări informale" → academice
- [x] Corectare formule și parametri (adăugare context Hz/bin)
- [x] Consistență terminologie (sea state, viteză radială, etc.)
- [x] Eliminare cuvinte "prea tari" (perfect, miracol)
- [x] Adăugare context și condiții (CPU, parametri, secvențe)
- [x] Compilare și verificare LaTeX

---

## NOTĂ FINALĂ

Prezentarea este acum:
- **Mai academică**: Evită limbajul informal și exclamațiile
- **Mai transparentă**: Specifică exact condițiile experimentale
- **Mai coherentă**: Terminologie și stil consistent
- **Mai preciză**: Parametri convertiți în unități fizice relevante
- **Ușor reproducibilă**: Condițiile sunt clare și documentate

Documentul PDF este gata pentru prezentare sau publicare.
