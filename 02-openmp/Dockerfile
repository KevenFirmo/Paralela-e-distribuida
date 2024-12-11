# Use uma imagem base com um compilador C++ e suporte a OpenMP
FROM ubuntu:22.04

# Instalar dependências
RUN apt-get update && apt-get install -y \
    build-essential \
    libpthread-stubs0-dev \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

# Criar diretórios para a aplicação e para os dados
RUN mkdir -p /app/bin /app/data

# Definir o diretório de trabalho para a aplicação
WORKDIR /app/bin

# Copiar os arquivos de código-fonte para o diretório da aplicação
COPY openmp.cpp /app/bin/

# Compilar o código com OpenMP
RUN g++ -fopenmp -o openmp_app openmp.cpp -std=c++17

RUN chmod +x openmp_app

# Comando padrão para executar o aplicativo
CMD ["./openmp_app", "/app/data/input_graph.txt"]
