# 1. Fem servir python:3.10-slim
FROM python:3.10-slim

# 2. Establim el directori de treball dins del contenidor
WORKDIR /code

# 3. Copiem el fitxer de requeriments
COPY ./requirements.txt /code/requirements.txt

# 4. Instalem les dependències
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Copiem el codi de l'aplicació dins del contenidor
COPY ./app /code/app

# 6. Executar l'API utilitzant Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]