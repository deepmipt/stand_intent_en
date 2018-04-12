FROM stand/docker_cuda

VOLUME /logs
WORKDIR /app
ADD . /app

RUN pip install -r requirements.txt && \
    ./download_components.sh

EXPOSE 6007

ENV MODEL_NAME="stand_intent_en"
ENV DEFAULT_POD_NODE="unknown_node"
ENV DEFAULT_POD_NAME="unknown_pod"

CMD if [[ $POD_NODE=="" ]]; then POD_NODE=$DEFAULT_POD_NODE; fi && \
    if [[ $POD_NAME=="" ]]; then POD_NAME=$DEFAULT_POD_NAME; fi && \
    DATE_TIME=$(date '+%Y-%m-%d_%H-%M-%S.%N') && \
    LOG_DIR="/logs/"$MODEL_NAME"/"$POD_NODE"/" && \
    LOG_FILE=$MODEL_NAME"_"$DATE_TIME"_"$POD_NAME".log" && \
    LOG_PATH=$LOG_DIR$LOG_FILE && \
    mkdir -p $LOG_DIR && \
    python3.6 intent_en_api.py > $LOG_PATH 2>&1