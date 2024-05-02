FROM python:3.9-slim

ARG GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

WORKDIR /workspace

ADD requirements.txt app.py /workspace/

COPY ./models /models

RUN pip install -r /workspace/requirements.txt

CMD ["python", "/workspace/app.py"]