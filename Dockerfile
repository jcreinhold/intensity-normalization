# Use official python runtime as a parent image
FROM python:3.6-slim-stretch
LABEL maintainer="Jacob Reinhold, jcreinhold@gmail.com"

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# 1) Install numpy first since skfuzzy requires it to be pre-installed
# 2) Install any needed packages specified in requirements.txt
# 3) Install ANTsPy which currently requires a specific path
# 4) Install this package into the container
# 5) Setup matplotlib to not pull in a GUI
RUN pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir numpy && \
    pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt && \
    pip install --no-cache-dir antspyx && \
    python setup.py install && \
    echo "backend: agg" > matplotlibrc

WORKDIR /work
