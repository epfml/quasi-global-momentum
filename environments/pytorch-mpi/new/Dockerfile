FROM user/pytorch-mpi:latest
WORKDIR $HOME
ENV PYTHON_VERSION=3.8
ENV PATH $HOME/.openmpi/bin:$PATH
ENV LD_LIBRARY_PATH $HOME/.openmpi/lib:$LD_LIBRARY_PATH
ENV PATH $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin:$PATH

#match version to your torch https://github.com/pytorch/text
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install torchtext==0.9

RUN $HOME/conda/bin/conda clean -ya
