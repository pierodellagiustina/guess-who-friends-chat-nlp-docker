FROM python:3.8

# Create app folder and move to it
WORKDIR /app

# copy all files and folders
COPY . .

# run requirements installation
RUN pip install -r requirements.txt

# run the app
CMD ["python", "app.py"]

# commands to build and run
# docker build -t guess-who
# docker run guess-who
# docker run -p 8000:8000 guess-who

# interactive shell on the image
# docker run -it image-name sh

# remove
# docker rm $(docker ps -aq)
# docker image prune -a






