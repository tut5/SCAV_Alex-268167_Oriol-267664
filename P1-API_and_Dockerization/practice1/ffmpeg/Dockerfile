# 1. Fem servir una imatge base molt lleugera (Alpine Linux)
FROM alpine:3.19

# 2. Instal·lem FFmpeg
# 'apk' és el gestor de paquets d'Alpine (similar a 'apt' a Ubuntu)
RUN apk update && \
    apk add --no-cache ffmpeg

# 3. Establim un directori de treball per defecte
WORKDIR /tmp

# 4. Definim el punt d'entrada
# Això fa que quan executis el contenidor, actuï directament com la comanda 'ffmpeg'
ENTRYPOINT ["ffmpeg"]

# 5. Comanda per defecte (si no li passem arguments, mostra l'ajuda)
CMD ["--help"]