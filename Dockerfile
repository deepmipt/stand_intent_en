FROM stand/docker_cuda

VOLUME /vol
WORKDIR /app
ADD . /app

RUN pip install -r requirements.txt && \
    ./download_components.sh

EXPOSE 6007

CMD python3.6 intent_en_api.py > /vol/intent_en.log 2>&1