From python3.9-slim

WORKDIR /app
COPY . /app
RUN pip install --non-cache-dir -r requirements.txt
EXPOSE 8080
RUN ['python' , 'app.py']