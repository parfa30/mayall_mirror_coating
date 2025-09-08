FROM quay.io/jupyter/base-notebook:notebook-7.4.5

RUN pip install --no-cache-dir \
    matplotlib \
    scikit-learn \
    matplotlib \
    pandas \
    gspread \
    ipywidgets 

COPY mayall_mirror_wash_analysis.ipynb /home/jovyan/work/

