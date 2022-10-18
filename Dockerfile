FROM python:3.8

# Create app folder and move to it
WORKDIR /app

# copy all files
COPY * .

# run requirements installation
RUN pip install -r requirements.txt

# run the app
CMD ["python", "app.py"]

# commands to build and run
# docker build -t guess-who
# docker run guess-who

