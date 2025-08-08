FROM python:3.11.9-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y git && apt-get clean
RUN pip install --upgrade pip

COPY src /app/src
COPY run.sh /app/.
COPY requirements.txt /app/.

RUN pip install -r requirements.txt
