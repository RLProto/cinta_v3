FROM python:3.10-slim as base

EXPOSE 9999

COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9999"]
