# **Pràctica 1: API & Dockerization**

Aquest repositori conté la implementació de la **Pràctica 1** de l'assignatura de **Sistemes de Codificació d'Àudio i Vídeo (SCAV)**. El projecte consisteix en una arquitectura de microserveis basada en **Docker** que fa servir FastAPI per executar algoritmes de manipulació d’imatges i vídeo, integrant la lògica del Seminari 1 i delegant tasques a un contenidor independent d'FFmpeg.

## **Task 1: Creació de l'API i Dockerització**

Enunciat:   
*“Start a new project called practice1. You’re going to create an API. You can use Flask, FastAPI (recommended), Django, or any other python framework you’re familiar with… or any other API framework from other language (Golang? Node.JS?) Put it inside a docker.”*

Per crear la API hem escollit el framework **FastAPI**, ja que hem prioritzat la velocitat i facilitat d’ús. A més a més, com a punt a favor, ens permet generar automàticament documentació del projecte.

- **Estructura del projecte:** Hem creat la carpeta *practice1* amb el fitxer *requirements.txt* i l’estructura de l’aplicació. 

Així doncs, l’estructura de fitxers del projecte és la següent:

	→ Practice1/  
		→ app/  
			→ \_\_init\_\_.py   
			→ main.py \[Codi de la API\]  
		→ Dockerfile \[Configuració del contenidor\]  
		→ requirements.txt

![][image1]

Inicialment, el document *requirements.txt* només inclou les dues llibreries *fastapi* i *uvicorn*.

			

- **Dockerització:** Hem creat el *Dockerfile* basat en la imatge de *python:3.10-slim*. El *Dockerfile* instal·la les dependències i executa el web server *uvicorn*.

Per executar el docker hem de seguir els següents passos:

Construir la imatge del Docker

`docker build -t scav-practice-api .`

`![][image2]`

Executar el contenidor:

`docker run -d --name scav-api -p 8000:80 scav-practice-api`

`![][image3]`

Verificar que realment funciona:

- Resposta JSON: [`http://localhost:8000/`](http://localhost:8000/)

  ![][image4]

- Documentació automàtica (Swagger): [`http://localhost:8000/docs`](http://localhost:8000/docs) 

`![][image5]`

- **Resultat:** Tenim com a punt de partida pels propers exercicis una API funcional a la que podem accedir des del port 8000 que respon a peticions HTTP. 

## **Task 2: Dockerització d'FFmpeg**

Enunciat:   
*“Put ffmpeg inside a Docker”*

En lloc d’instal·lar FFmpeg dins el contenidor de la API hem creat una carpeta *ffmpeg/* amb el seu *Dockerfile*. Hem fet servir *alpine:3.19* (distro de Linux lleugera) on hem instal·lat *FFmpeg*. El contenidor pot executar comandes *ffmpeg* directament. 

Per construir la imatge hem fet servir la comanda:

`docker build -t scav-ffmpeg ./ffmpeg`

`![][image6]`

I, per executar el contenidor per veure si *FFmpeg* respon a `ffmpeg -version`:

`docker run --rm scav-ffmpeg -version`

`![][image7]`

D’aquesta manera tenim *FFmpeg* a un Docker independent, que més endavant podrem integrar amb la API per implementar les funcionalitats del *Seminari 1*. 

## **Task 3: Integració de la Lògica Prèvia (Seminari 1\)**

Enunciat:   
*“Include all your previous work inside the new API. Use the help of any AI tool to adapt the code and the unit tests”*

Per implementar aquesta tasca hem fet servir *Gemini* (tot i que hem revisat, adaptat i modificat el seu codi) per implementar la lògica del *Seminari 1* a *app/scav\_logic.py*, que contindrà tota la lògica de les funcionalitats a les que podrem accedir des de la API. Hem hagut d’importar les dependències *numpy, scipy* i *PyWavelets* que també hem inclòs a *requirements.txt*. 

Les classes que hem integrat són:

- ColorTranslator: Conversió RGB ←→ YUV.  
- SerpentineScanner: Lectura de matrius en zig-zag.  
- RLEConverter: Compressió Run-Length Encoding.  
- DCTConverter i DWTConverter: Transformades de freqüència.

Com que hem modificar el document *requirements.txt*, hem de tornar a construir la imatge del *Docker* per actualitzar-lo amb els nous mòduls.   
Per fer-ho, executem les següents comandes:

docker build \-t scav-practice-api .

docker rm \-f scav-api   → Esborrem el contenidor vell si s’estava executant

docker run \-d \--name scav-api \-p 8000:80 scav-practice-api

![][image8]

## **Task 4: Creació d'Endpoints**

Enunciat:   
*“Create at least 2 endpoints which will process some actions from the previous S1”*

Per crear els *endpoins* que ens permetran executar les funcions del *Seminari 1* des del navegador (o client HTTP) hem modificat l’arxiu *main.py*. Hem fet servir *Pydantic* per validar les dades d’entrada. Com que encara no hem connectat el contenidor d’*FFmpeg* només funcionaran els *endpoints* que facin servir Python pur, com poden ser `ColorTranslator` i `RLEConverter`. 

Tot i així, hem implementat els *endpoints* per totes les funcionalitats:

- `POST /converter/rgb-to-yuv`  
- `POST /converter/rle`  
- `POST /scanner/serpentine`  
- `POST /converter/dct/forward` (i `inverse`)

Com hem modificat l’entorn, hem de reconstruir la imatge perquè els canvis s’hi vegin reflexats:

- `docker rm -f scav-api` → Aturar i esborrar el contenidor vell  
- `docker build -t scav-practice-api .` → Reconstruir la imatge amb el nou codi  
- `docker run -d --name scav-api -p 8000:80 scav-practice-api` → Executar el nou contenidor

![][image9]

Podem provar la API seguint els següents passos:

1. Accedir a: [http://localhost:8000/docs](http://localhost:8000/docs)  
   ![][image10]  
2. Provar *POST /converter/rgb-to-yuv \-\> Try it out.*  
3. Canviar el JSON d’exemple per:

   `{`

     `"r": 255,`

     `"g": 0,`

     `"b": 0`

   `}`

4. Executar i veure si retorna el resultat esperat. 

![][image11]  
![][image12]

Podem provar cadascun dels serveis que hem implementat amb els endpoints, però és important saber que només funcionen aquells que es basin en Python (sense *FFmpeg*), ja que de moment no tenim accés a comandes *FFmpeg*. 

![][image13]  
![][image14]

## **Task 5: Orquestració amb Docker Compose**

Enunciat:   
*“Use docker-compose to launch both and make them interact (i.e., you have a method for conversion, you launch your API and it will call the FFMPEG docker)”*

Per aconseguir la comunicació entre la API i el contenidor de *FFmpeg* hem fet servir una carpeta compartida (*/shared*) dins la que la API deixarà les imatges (o recursos als que *FFmpeg* hagi d’accedir). A més a més, mitjançant un Docker Socket la API tindrà accés al Docker (*var/run/rocker.sock*) per poder enviar les ordre d’execució al contenidor de *FFmpeg*. Per llançar aquestes ordres farem servir la llibreria de Python *docker*. 

Per poder fer ús d’aquest sistema i executar-ho amb una sola comanda farem servir Docker Compose, que s’encarregarà d’unificar tot el sistema. 

El pas a pas de la implementació és el següent:

1. Actualitzar *requirements.txt* per afegir la llibreria *docker*.   
2. Modificar la lògica de *app/scav\_logic.py* perquè, en lloc d'intentar executar ffmpeg localment, es connecti al Docker que conté *FFmpeg* i llanci les ordres necessàries.   
3. Afegir els *endpoints* de les funcions del *Seminari 1* que requereixen l’ús de comandes de *FFmpeg*: *"/image/resize"* i *"/image/compress-bw"*.   
4. Crear el *Docker-Compose.yml*. Defineix els dos serveis i com es connecten entre ells.   
5. Creació de la carpeta *shared\_data* dins la carpeta arrel.   
6. Executar el servei: `docker-compose up --build`

`![][image15]`

`![][image16]`

Per comprovar la funcionalitat dels diferents *endpoints* implementats hem llançat algunes proves des de la API ([http://localhost:8000/docs](http://localhost:8000/docs)). 

![][image17]

El procés intern quan executem aquestes comandes és el següent:

1. La API rep la imatge i la guarda a `shared_data/test_image.jpg`.  
2. La API llança la comanda d’execució a `ffmpeg-service` sobre */shared/test\_image.jpg*.  
3. El contenidor d'FFmpeg executa la comanda i guarda el resultat a *shared\_data/bw\_test\_image.jpg*.  
4. La API llegeix el fitxer resultant i el mostra al navegador.

![][image18]

![][image19]

![][image20]![][image21]

![][image22]

## **6\. Instruccions d'Ús**

El projecte és compatible amb **Windows**, **macOS** i **Linux**. Per a usuaris de **Windows**, és necessari tenir activat el subsistema de Linux **WSL 2 (Windows Subsystem for Linux):** Docker Desktop a Windows requereix WSL 2 per executar contenidors Linux de manera nativa i eficient. Es pot instal·lar executant `wsl --install`.

Tot el sistema està basat en contenidors, per la qual cosa no cal instal·lar Python ni FFmpeg al host, però si el motor de contenidors **Docker Desktop**, que ha d’estar actiu. 

Per executar tot el projecte:

1. Obrir terminal a la carpeta arrel.  
2. Executar: docker-compose up \--build  
3. Accedir a la documentació interactiva: http://localhost:8000/docs  
4. Per aturar: docker-compose down

## **Autors**

* **\[Oriol Tutusaus \- 267664\]**  
* **\[Alex Alastuey \- 268167\]**

SCAV – Pràctica 1

