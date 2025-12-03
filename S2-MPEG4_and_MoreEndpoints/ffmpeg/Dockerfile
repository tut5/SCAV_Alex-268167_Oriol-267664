# Alpine Linux (imatge molt lleugera)
FROM alpine:3.19

# Instalacio ffmpeg
# 'apk' --> gestor de paquets d'Alpine
RUN apk update && \
    apk add --no-cache ffmpeg

# Directori de treball
WORKDIR /tmp

# Definim el punt d'entrada
ENTRYPOINT ["ffmpeg"]

# Comanda per defecte
CMD ["--help"]