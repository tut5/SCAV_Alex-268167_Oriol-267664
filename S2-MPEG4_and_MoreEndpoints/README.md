# **Pràctica 2: MPEG-4 & Nous Endpoints (FFmpeg Integration)**

Aquest repositori conté la implementació de la **Pràctica 2 (S2)** de l'assignatura de **Sistemes de Codificació d'Àudio i Vídeo (SCAV)**. L'objectiu principal d'aquest seminari és estendre la API creada a la Pràctica 1 per incloure funcionalitats avançades de manipulació d'imatge (relacionades amb conceptes de MPEG-4) mitjançant la integració amb **FFmpeg** via Docker.

## **Task 1: Actualització de l'Estructura i Dependències**

Enunciat:
*“Implement new endpoints involving MPEG-4 concepts and ensure the communication between the Python API and the FFmpeg service works via Docker.”*

Hem partit de l'estructura de la Pràctica 1 i hem afegit els elements necessaris per comunicar la API amb el contenidor d'FFmpeg.

L'estructura de fitxers actualitzada és la següent:

	→ S2-MPEG4_and_MoreEndpoints/
		→ app/
			→ main.py [Nous endpoints afegits]
			→ scav_logic.py [Lògica de comunicació amb Docker]
		→ ffmpeg/ [Dockerfile del servei FFmpeg]
		→ shared_data/ [Volum compartit per imatges]
		→ docker-compose.yml [Orquestració de serveis]
		→ requirements.txt [Dependències actualitzades]

Hem actualitzat el fitxer *requirements.txt* per incloure la llibreria `docker`, necessària per controlar contenidors des de Python.

## **Task 2: Dockerització d'FFmpeg i Volums**

Per permetre la manipulació d'imatges utilitzant eines externes, hem creat un servei dedicat a FFmpeg.

El funcionament és el següent:
1.  **Dockerfile d'FFmpeg:** Hem creat una imatge lleugera basada en Alpine amb FFmpeg instal·lat a la carpeta `ffmpeg/`.
2.  **Volum Compartit:** Hem definit una carpeta `shared_data` que actua com a pont. Tant la API com el contenidor d'FFmpeg tenen accés a aquesta carpeta. Això permet que la API guardi una imatge, FFmpeg la processi, i la API la llegeixi de nou sense necessitat de transferir bytes directament per la xarxa.

## **Task 3: Nous Endpoints a la API (MPEG-4 Concepts)**

Enunciat:
*“Create endpoints to process actions involving FFmpeg interactions like resizing or chroma subsampling.”*

Hem modificat `main.py` per exposar aquestes funcionalitats. Els nous *endpoints* implementen conceptes clau de la codificació MPEG-4:

- **`POST /image/resize`**: Rep una imatge i redueix la seva resolució espacial (downscaling). Això simula la reducció de càrrega per bitrate en vídeo.
- **`POST /image/compress-bw`**: Rep una imatge i elimina la informació de crominància, deixant només la luminància (escala de grisos), un concepte fonamental en la compressió de vídeo.

## **Task 4: Lògica d'Orquestració (scav_logic.py)**

Per connectar la API amb el contenidor d'FFmpeg, hem modificat `app/scav_logic.py`.

El flux d'execució és:
1.  La API rep un `UploadFile` i el guarda a `./shared_data`.
2.  Utilitzant la llibreria `docker` de Python, la API es connecta al socket de Docker (`/var/run/docker.sock`).
3.  La API executa una comanda `ffmpeg` dins del contenidor `scav-ffmpeg`, apuntant als fitxers del volum compartit.
4.  Un cop FFmpeg acaba, la API retorna el fitxer resultant.

## **Task 5: Execució amb Docker Compose**

Enunciat:
*“Use docker-compose to launch both and make them interact.”*

Hem creat l'arxiu `docker-compose.yml` per aixecar tot l'entorn amb una sola comanda, assegurant que els volums i les xarxes estiguin ben configurats.

Per executar el projecte:

`docker-compose up --build`

![](./assets/image_compose_up.png)

Podem comprovar el funcionament mitjançant Swagger:

1.  Accedir a: [`http://localhost:8000/docs`](http://localhost:8000/docs)
2.  Provar l'endpoint **/image/resize**:
    * Pujar `test_image.jpg`.
    * Resposta: `resized_test_image.jpg` amb resolució reduïda.

3.  Provar l'endpoint **/image/compress-bw**:
    * Pujar `test_image.jpg`.
    * Resposta: `bw_test_image.jpg` (blanc i negre).

![](./assets/image_swagger_test.png)

Aquests fitxers resultants es guarden físicament a la carpeta `shared_data` del host per verificació.

## **6. Instruccions d'Ús**

El projecte és compatible amb **Windows**, **macOS** i **Linux**.

1.  Obrir terminal a la carpeta `S2-MPEG4_and_MoreEndpoints`.
2.  Executar: `docker-compose up --build`
3.  Accedir a la documentació interactiva: [http://localhost:8000/docs](http://localhost:8000/docs)
4.  Per aturar: `docker-compose down`

## **Autors**

* **[Oriol Tutusaus - 267664]**
* **[Alex Alastuey - 268167]**

SCAV – Pràctica 2 (Seminari 2)