FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app/meta-l2d
WORKDIR /app/meta-l2d
