FROM nvidia/tensorflow:latest

ENV PATH /root/.local/bin:$PATH

RUN mkdir /workdir
RUN apt update && apt install vim -y
RUN pip install pip setuptools wheel -U && \
    pip install \
	segmentation-models \
	geopandas \
	scikit-learn \
	netcdf4 \
	shapely \
	jupyter

WORKDIR /workdir

