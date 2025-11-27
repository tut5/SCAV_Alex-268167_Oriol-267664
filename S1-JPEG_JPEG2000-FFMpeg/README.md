
# Seminari 1: S1-JPEG_JPEG2000-FFMpeg

Aquest repositori conté la implementació dels exercicis del **Primer Seminari** de
l'assignatura de _Sistemes de Codificació d'Àudio i Vídeo (SCAV)_.
El projecte consisteix en un script de Python que implementa diversos algoritmes de
compressió i manipulació d’imatges, a més de tests automatitzats.

## Funcionalitats Implementades

El fitxer principal (first_seminar.py) inclou les següents classes i mètodes:

1. **ColorTranslator**
    Conversió entre espais de color **RGB** i **YUV** (estàndard BT.601).
2. **ImageEncoder**
    Automatització d’ **FFmpeg** per a:
       ○ Redimensionament d’imatges mantenint l’aspect ratio.
       ○ Conversió a blanc i negre.
       ○ Compressió agressiva.
3. **SerpentineScanner**
    Algoritme de lectura de bytes en zig-zag (serpentine).
4. **RLEConverter**
    Implementació de l’algoritme de compressió _Run-Length Encoding_.
5. **DCTConverter**
    Transformada Discreta del Cosinus ( **DCT** ) i la seva inversa ( **IDCT** ) utilitzant scipy.
6. **DWTConverter**
    Transformada Discreta Wavelet ( **DWT** ) utilitzant _Haar_ (PyWavelets).

## Requisits Previs

Per executar aquest projecte correctament, necessites tenir instal·lat el següent programari:

### 1. Python i Llibreries

Aquest projecte utilitza **Python 3**. Les dependències externes són:
● numpy
● scipy
● PyWavelets

### 2. FFmpeg (Important!)

El codi utilitza ffmpeg i és imprescindible tenir-lo instal·lat i accessible des
del **PATH**.

## Instal·lació i Ús

### 1. Clona el repositori

### 2. Prepara les imatges de prova


L’script espera trobar dues imatges al directori arrel:
test_image.jpg
test_image2.jpg
_(Pots canviar els noms a first_seminar.py si cal.)_

### 3. Executa l’script

python first_seminar.py
Això llançarà automàticament els **Unit Tests visuals**.

## Sortida de Dades

Després de l’execució, es generaran automàticament les carpetes:

● **test_outputs/**

Imatges redimensionades i comprimides per FFmpeg.

● **test_outputs_serpentine/**

Fitxers de text amb els resultats de l’escaneig en serpentí.

## Tests Unitaris

En executar l’script, es mostra un resum detallat:

● Verificació de fórmules matemàtiques ( _RGB ↔ YUV_ ).

● Generació d’arxius amb FFmpeg.

● Comprovació d’integritat de transformades ( _DCT / DWT inverses_ ).

● Resum final d’errors i encerts.

## 📝 Autor


**[Oriol Tutusaus - 267664]**

**[Alex Alastuey - 268167]**

SCAV – Seminari 1



This is a offline tool, your data stays locally and is not send to any server!
Feedback & Bug Reports
