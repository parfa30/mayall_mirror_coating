# mayall_mirror_coating

## Docker image

Run this image:

```sh
docker run --rm -p 8889:8888 mayall_mirror_wash:latest start-notebook.py --NotebookApp.token='my-token'

```

Access the notebook at [http://localhost:8889/lab?token=my-token](http://localhost:8889/lab?token=my-token)

Build the docker image:

```sh
docker build r -f Containerfile -t mayall_mirror_wash .

```

Build the image using Podman:

```sh
docker build --format docker -f Containerfile -t mayall_mirror_wash .

```
