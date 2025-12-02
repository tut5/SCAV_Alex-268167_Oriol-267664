# 1. Fem servir una imatge base de Python oficial i lleugera (slim)
FROM python:3.10-slim

# 2. Establim el directori de treball dins del contenidor
WORKDIR /code

# 3. Copiem el fitxer de requeriments primer (per aprofitar la cache de Docker)
COPY ./requirements.txt /code/requirements.txt

# 4. Instal·lem les dependències
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Copiem la resta del codi de l'aplicació dins del contenidor
COPY ./app /code/app

# 6. Comanda per executar l'API utilitzant Uvicorn
# host 0.0.0.0 fa que sigui accessible des de fora del contenidor
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]