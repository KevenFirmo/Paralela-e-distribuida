# Entrega CPD 2024

## Estrutura do Projeto
entrega_cpd_2024/\
    ├── 01-pthreads/\
    ├── 02-openmp/\
    ├── 03-openmpi/\
    └── README.md

## Como Utilizar

### Pré-requisitos

- **Docker** instalado na sua máquina. [Instalar Docker](https://docs.docker.com/get-docker/)
- **Docker Compose** (opcional, se necessário para orquestração).

### 01 - pthreads

Implementação utilizando **PThreads**.

#### Construir a Imagem Docker

```bash
cd 01 - pthreads
docker build -t pthreads_app .

docker run --rm -v $(pwd):/app/data pthreads_app

cat graph_metrics.json  
```
### 02 - openmp
Implementação utilizando **openMp**.


```bash
cd 02 - openmp
docker build -t openmp_app .

docker run --rm -v $(pwd):/app/data openmp_app

cat graph_metrics.json  
```

### 03 - openmpi
Implementação utilizando **openMpi**.

```bash
cd 03 - openmpi

docker-compose build

docker-compose up -d
```
Isso levantará 4 contêineres: node1, node2, node3 e node4.

Verifique se os contêineres estão ativos:
```bash
docker ps
```
Entre no contêiner node1 como mpiuser:

```bash
docker exec -it --user=mpiuser node1
```
Execute o aplicativo distribuído:

```bash
mpirun -np 4 -H node1,node2,node3,node4 ./openmpi_app /app/data input_graph.txt
```
Ao final, verifique o resultado.
```bash
cat /app/data/graph_metrics.json
```
Ou então, caso deseje ver no host, saia do bash do container e do container, e execute:
```bash
cat data/graph_metrics.json
```