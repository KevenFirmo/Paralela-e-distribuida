A aplicação realiza O código apresentado é um programa escrito em C++ que analisa um grafo representado por um arquivo de texto (web-Google.txt) e realiza várias operações sobre ele, utilizando múltiplas threads para otimizar o processamento. O código foi desenvolvido utilizando o padrão C++11 e as bibliotecas pthread para multithreading e nlohmann::json para manipulação de JSON.

Pré-requisitos
Docker: A aplicação está configurada para ser executada dentro de um container Docker. Certifique-se de ter o Docker instalado em seu sistema. Caso não tenha o Docker, siga a documentação oficial do Docker para instalá-lo.
Como Rodar a Aplicação
1. Clonar o Repositório
Clone este repositório para o seu ambiente local:

bash
Copiar código
git clone https://github.com/seu-usuario/repo.git
cd repo
2. Construir a Imagem Docker
Execute o comando abaixo para construir a imagem Docker a partir do Dockerfile:

bash
Copiar código
docker build -t cpp-app .
3. Rodar a Aplicação
Após a construção da imagem, você pode rodar a aplicação no Docker:

bash
Copiar código
docker run --rm cpp-app
O parâmetro --rm garante que o container será removido automaticamente após a execução.

4. (Opcional) Rodar com Argumentos
Se você precisar passar argumentos para a aplicação, você pode adicionar os parâmetros após o nome da imagem Docker. Exemplo:

bash
Copiar código
docker run --rm cpp-app argumento1 argumento2
