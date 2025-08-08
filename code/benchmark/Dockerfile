
FROM python:3.10-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y git && apt-get clean

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt --verbose --no-cache
RUN pip install --no-deps utilsforecast==0.2.6 --verbose --no-cache

COPY data ./data
COPY logs ./logs
COPY src ./src
COPY run.sh .

CMD ["./run.sh"]