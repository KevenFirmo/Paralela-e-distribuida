# Use uma imagem base do Ubuntu
FROM ubuntu:22.04

# Evitar interações durante a instalação de pacotes
ENV DEBIAN_FRONTEND=noninteractive

# Atualizar os repositórios e instalar dependências
RUN apt-get update && apt-get install -y \
    build-essential \
    openmpi-bin \
    libopenmpi-dev \
    nlohmann-json3-dev \
    sudo \
    openssh-client openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Criar diretórios para a aplicação e para os dados
RUN mkdir -p /app/bin /app/data /home/mpiuser/.ssh

# Criar um usuário não-root chamado 'mpiuser'
RUN useradd -ms /bin/bash mpiuser \
    && echo "mpiuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Definir o proprietário dos diretórios para o usuário não-root
RUN chown -R mpiuser:mpiuser /app/bin /app/data /home/mpiuser/.ssh

# Definir o diretório de trabalho para a aplicação
WORKDIR /app/bin

# Copiar o código fonte
COPY openmpi.cpp /app/bin/


# Compilar o código com OpenMPI
RUN mpic++ -o openmpi_app openmpi.cpp -std=c++11

# Garantir permissões de execução
RUN chmod +x openmpi_app

# Configurar SSH: gerar chaves e configurar acesso sem senha
USER mpiuser
RUN ssh-keygen -t rsa -f /home/mpiuser/.ssh/id_rsa -q -N "" && \
    cat /home/mpiuser/.ssh/id_rsa.pub >> /home/mpiuser/.ssh/authorized_keys && \
    chmod 600 /home/mpiuser/.ssh/authorized_keys

USER root
RUN chown mpiuser:mpiuser /home/mpiuser/.ssh/authorized_keys

# Ajustar o SSHD para aceitar conexões
RUN mkdir /var/run/sshd
RUN sed -i 's/.*PasswordAuthentication.*/PasswordAuthentication no/g' /etc/ssh/sshd_config
RUN sed -i 's/.*PermitRootLogin.*/PermitRootLogin prohibit-password/g' /etc/ssh/sshd_config

EXPOSE 22

# Iniciar SSH e deixar container rodando em segundo plano
CMD ["/usr/sbin/sshd", "-D"]
