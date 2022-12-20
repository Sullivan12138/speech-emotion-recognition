FROM kpavlovsky/python3.7

COPY . /

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN cd / && pip install -r requirements.txt