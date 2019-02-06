# tell docker which basic image my new image is based on
FROM centos:7

WORKDIR /tmp

COPY requirements.txt /tmp

# install packages
RUN yum -y install epel-release \ 
    && yum -y install python-pip python-devel git \
	&& yum -y install tkinter wget gcc g++ gcc-gfortran\
	&& yum -y update  \
	&& yum -y clean all

# install Python requirements
RUN pip install --upgrade pip \
    && pip install numpy \
    && pip install pytest \
    && pip install -r requirements.txt \
    && rm -rf ~/.cache ~/.pip

