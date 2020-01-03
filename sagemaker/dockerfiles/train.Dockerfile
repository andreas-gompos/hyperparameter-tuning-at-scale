FROM python:3.7.4

COPY ./requirements.txt requirements.txt
RUN pip install -U pip && \
    pip install -r requirements.txt

WORKDIR /code
COPY ./src/ /code/
ENTRYPOINT [ "python", "sagemaker_train.py"]
