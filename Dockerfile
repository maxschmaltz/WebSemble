FROM nvidia/cuda:11.3.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive


# install python, pip
RUN apt-get update &&\
    apt-get install python3.10 -y &&\
    apt-get install python3-pip -y

# making directory of app
WORKDIR /WebSemble

# copy contents elementwise to reduce layer sizes
COPY ./websemble/downstream ./websemble/downstream
COPY ./instructions ./instructions
COPY ./websemble/models/bart-base-webis22 ./websemble/models/bart-base-webis22
COPY ./websemble/models/bert-base-uncased-MNLI-webis22 ./websemble/models/bert-base-uncased-MNLI-webis22
COPY ./websemble/models/bert-large-uncased-whole-word-masking-finetuned-squad ./websemble/models/bert-large-uncased-whole-word-masking-finetuned-squad
COPY ./websemble/models/deberta-v3-base-tasksource-nli ./websemble/models/deberta-v3-base-tasksource-nli
COPY ./websemble/models/distilbert-base-cased-distilled-squad ./websemble/models/distilbert-base-cased-distilled-squad
COPY ./websemble/models/distilbert-base-uncased ./websemble/models/distilbert-base-uncased
COPY ./websemble/models/distilbert-base-uncased-webis22 ./websemble/models/distilbert-base-uncased-webis22
COPY ./websemble/models/pegasus-xsum ./websemble/models/pegasus-xsum
COPY ./websemble/models/roberta-base-squad2 ./websemble/models/roberta-base-squad2
COPY ./websemble/utils ./websemble/utils
COPY ./websemble/run.py ./websemble/run.py
COPY ./websemble/web_trainer.py ./websemble/web_trainer.py
COPY ./requirements.txt ./requirements.txt
COPY ./README.md ./README.md

# install packages
RUN pip install -r requirements.txt
RUN cd WebSemble

# make script executable
RUN chmod +x /websemble/run.py
ENTRYPOINT ["python3", "/websemble/run.py", "$inputDataset", "$outputDir"]