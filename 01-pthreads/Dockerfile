# Use uma imagem base com um compilador C++ e suporte a PThreads
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
COPY pthreads.cpp /app/bin/

# Compilar o código com PThreads
RUN g++ -o pthreads_app pthreads.cpp -lpthread -std=c++17

# Garantir permissões de execução
RUN chmod +x pthreads_app

# Definir o comando padrão para executar o aplicativo, apontando para o arquivo de entrada no diretório de dados
CMD ["./pthreads_app", "/app/data/input_graph.txt"]
