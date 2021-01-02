FROM conda/miniconda3
    
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python \
         nginx \
         ca-certificates \
         graphviz \
         git \
    && rm -rf /var/lib/apt/lists/*

RUN sed -i -e 's/archive.ubuntu.com\|security.ubuntu.com/old-releases.ubuntu.com/g' /etc/apt/sources.list
RUN apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get install gcc -y


RUN conda info && conda update conda

# add conda to ssh sessions (all users) as ssh scrapes entire environment when starting the shell
RUN echo "export PATH=\"/opt/conda/bin:$PATH\"" >> /etc/profile
RUN echo "export LANG=en_US.UTF-8 LANGUAGE=en_US.UTF-8 LC_ALL=en_US.UTF-8" >> /etc/profile

# update environment
ADD environment.yaml environment.yaml

# channel needs to be added here
RUN conda env update -n base -f environment.yaml


ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY . /opt/program
WORKDIR /opt/program

ENTRYPOINT ["python","main.py"]