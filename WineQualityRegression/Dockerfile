FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model.pkl scaler.pkl serve.py .

EXPOSE 9696

CMD ["python", "serve.py"]