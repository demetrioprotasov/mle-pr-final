FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY recsys_app ./recsys_app

# parquet-артефакты не кладём в образ: они весят десятки мегабайт
# и обновляются после каждого дообучения, поэтому монтируются снаружи
VOLUME /app/data

EXPOSE 8010

CMD ["uvicorn", "recsys_app.main:app", "--host", "0.0.0.0", "--port", "8010"]
