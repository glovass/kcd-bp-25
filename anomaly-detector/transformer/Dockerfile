FROM python:3.12-slim

ENV APP_HOME=/app
COPY ./ $APP_HOME
WORKDIR $APP_HOME


# workaround for psutil installation problem: command 'gcc' failed
RUN apt-get update && apt-get install -y \
  build-essential python3-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN pip install --no-binary :all: --compile psutil

RUN pip install kserve
RUN pip install joblib
RUN pip install scikit-learn

RUN chmod +x /app/transformer.py

ENTRYPOINT ["/app/transformer.py"]