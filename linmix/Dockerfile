FROM python:2

RUN pip install numpy==1.15.0

RUN git clone https://github.com/jmeyers314/linmix.git

RUN cd linmix && python setup.py install

WORKDIR /mcmc

COPY run_mcmc.py .

USER 1000

ENTRYPOINT [ "python", "run_mcmc.py" ]