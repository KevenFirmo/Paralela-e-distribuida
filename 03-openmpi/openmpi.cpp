// main.cpp

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <stack>
#include <cmath>
#include <algorithm>
#include <random>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Estruturas Globais
std::unordered_set<int> nodes;
std::unordered_map<int, std::vector<int>> adjList;
std::unordered_map<int, std::vector<int>> reverseAdjList;
unsigned long long edges = 0;
std::vector<std::pair<int, int>> graphData;

// Lista de adjacência não direcionada
std::unordered_map<int, std::unordered_set<int>> undirectedAdjList;

// Função para carregar parte do grafo
void loadGraph(const std::vector<std::pair<int, int>>& localGraphData) {
    for (const auto& edge : localGraphData) {
        int from = edge.first;
        int to = edge.second;

        nodes.insert(from);
        nodes.insert(to);
        edges++;

        adjList[from].push_back(to);
        reverseAdjList[to].push_back(from);

        // Construir a lista de adjacência não direcionada
        undirectedAdjList[from].insert(to);
        undirectedAdjList[to].insert(from);
    }
}

// Função para calcular o maior componente fracamente conectado (WCC)
std::pair<size_t, size_t> calculateWCC() {
    std::unordered_set<int> visited;
    size_t largestWCCNodes = 0, largestWCCEdges = 0;

    for (int node : nodes) {
        if (visited.find(node) != visited.end()) continue;

        std::queue<int> q;
        q.push(node);
        visited.insert(node);

        size_t componentNodes = 0, componentEdges = 0;
        while (!q.empty()) {
            int curr = q.front();
            q.pop();
            componentNodes++;

            for (int neighbor : adjList[curr]) {
                componentEdges++;
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    q.push(neighbor);
                }
            }
            for (int neighbor : reverseAdjList[curr]) {
                componentEdges++;
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    q.push(neighbor);
                }
            }
        }

        if (componentNodes > largestWCCNodes) {
            largestWCCNodes = componentNodes;
            largestWCCEdges = componentEdges / 2; // Cada aresta é contada duas vezes
        }
    }

    return {largestWCCNodes, largestWCCEdges};
}

// Função para calcular o maior componente fortemente conectado (SCC) usando Kosaraju
std::pair<size_t, size_t> calculateSCC() {
    std::unordered_set<int> visited;
    std::stack<int> finishStack;

    // 1ª Passagem: Ordenar nós por tempos de término (DFS no grafo original)
    for (int node : nodes) {
        if (visited.find(node) == visited.end()) {
            std::stack<int> dfsStack;
            dfsStack.push(node);

            while (!dfsStack.empty()) {
                int curr = dfsStack.top();
                dfsStack.pop();

                if (visited.find(curr) != visited.end()) continue;
                visited.insert(curr);

                for (int neighbor : adjList[curr]) {
                    if (visited.find(neighbor) == visited.end()) {
                        dfsStack.push(neighbor);
                    }
                }

                finishStack.push(curr);
            }
        }
    }

    // 2ª Passagem: Processar o grafo reverso
    visited.clear();
    size_t largestSCCNodes = 0, largestSCCEdges = 0;
    std::unordered_set<int> largestSCCSet;

    while (!finishStack.empty()) {
        int node = finishStack.top();
        finishStack.pop();

        if (visited.find(node) == visited.end()) {
            std::stack<int> dfsStack;
            dfsStack.push(node);

            size_t componentNodes = 0;
            std::unordered_set<int> componentSet;
            size_t componentEdges = 0;

            while (!dfsStack.empty()) {
                int curr = dfsStack.top();
                dfsStack.pop();

                if (visited.find(curr) != visited.end()) continue;
                visited.insert(curr);
                componentSet.insert(curr);
                componentNodes++;

                for (int neighbor : reverseAdjList[curr]) {
                    if (componentSet.find(neighbor) != componentSet.end()) {
                        componentEdges++;
                    }
                    if (visited.find(neighbor) == visited.end()) {
                        dfsStack.push(neighbor);
                    }
                }
            }

            // Atualizar o maior SCC
            if (componentNodes > largestSCCNodes) {
                largestSCCNodes = componentNodes;
                largestSCCEdges = componentEdges;
                largestSCCSet = componentSet;
            }
        }
    }

    // Recontar as arestas internas ao SCC para evitar contagem duplicada
    size_t actualEdgesInSCC = 0;
    for (int node : largestSCCSet) {
        for (int neighbor : adjList[node]) {
            if (largestSCCSet.find(neighbor) != largestSCCSet.end()) {
                actualEdgesInSCC++;
            }
        }
    }

    largestSCCEdges = actualEdgesInSCC;

    return {largestSCCNodes, largestSCCEdges};
}

// Função para calcular o número de triângulos em um grafo não direcionado
std::pair<size_t, double> calculateTriangles() {
    size_t triangles = 0;

    for (const auto& [u, neighbors_u] : undirectedAdjList) {
        for (int v : neighbors_u) {
            if (v <= u) continue; // Garantir que cada par seja considerado apenas uma vez
            const std::unordered_set<int>& neighbors_v = undirectedAdjList[v];
            for (int w : neighbors_u) {
                if (w <= v) continue; // Garantir w > v > u para evitar duplicatas
                if (neighbors_v.find(w) != neighbors_v.end()) {
                    // Triângulo encontrado entre u, v, w
                    triangles++;
                }
            }
        }
    }

    // Calcular o número de tripletos conectados (abertos e fechados)
    size_t triplets = 0;
    for (const auto& [node, neighbors] : undirectedAdjList) {
        size_t degree = neighbors.size();
        if (degree >= 2) {
            triplets += degree * (degree - 1) / 2;
        }
    }

    double closedTriangleFraction = (triplets > 0) ? (double)(3 * triangles) / triplets : 0.0;

    return {triangles, closedTriangleFraction};
}

// Função para calcular o coeficiente de agrupamento médio
double calculateAverageClusteringCoefficient() {
    double totalClusteringCoefficient = 0.0;
    size_t nodeCount = 0;

    for (const auto& [node, neighbors] : undirectedAdjList) {
        size_t degree = neighbors.size();
        if (degree < 2) continue;
        nodeCount++;

        size_t connectedNeighborPairs = 0;
        for (auto it1 = neighbors.begin(); it1 != neighbors.end(); ++it1) {
            for (auto it2 = std::next(it1); it2 != neighbors.end(); ++it2) {
                if (undirectedAdjList[*it1].find(*it2) != undirectedAdjList[*it1].end()) {
                    connectedNeighborPairs++;
                }
            }
        }
        double clusteringCoefficient = (2.0 * connectedNeighborPairs) / (degree * (degree - 1));
        totalClusteringCoefficient += clusteringCoefficient;
    }

    return (nodeCount > 0) ? totalClusteringCoefficient / nodeCount : 0.0;
}

// Função para calcular o diâmetro e o diâmetro efetivo
std::pair<int, double> calculateDiameters() {
    int diameter = 0;
    std::vector<int> shortestPaths;

    // Realizar BFS a partir de um subconjunto de nós aleatórios (para desempenho e representatividade)
    int numSamples = 1000; // Aumentado para 1000 para melhor precisão
    std::vector<int> sampleNodes;

    // Selecionar nós aleatoriamente
    if (!nodes.empty()) {
        std::vector<int> allNodes(nodes.begin(), nodes.end());
        size_t totalNodes = allNodes.size();
        numSamples = std::min(static_cast<int>(totalNodes), numSamples); // Ajustar caso o grafo tenha menos nós

        // Gerador de números aleatórios
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, totalNodes - 1);

        std::unordered_set<int> selectedIndices;
        while (sampleNodes.size() < static_cast<size_t>(numSamples)) {
            int idx = dis(gen);
            if (selectedIndices.find(idx) == selectedIndices.end()) {
                selectedIndices.insert(idx);
                sampleNodes.push_back(allNodes[idx]);
            }
        }
    }

    for (int startNode : sampleNodes) {
        std::unordered_map<int, int> distances;
        std::queue<int> q;
        distances[startNode] = 0;
        q.push(startNode);

        while (!q.empty()) {
            int curr = q.front();
            q.pop();

            for (int neighbor : undirectedAdjList[curr]) {
                if (distances.find(neighbor) == distances.end()) {
                    distances[neighbor] = distances[curr] + 1;
                    q.push(neighbor);
                    diameter = std::max(diameter, distances[neighbor]);
                    shortestPaths.push_back(distances[neighbor]);
                }
            }
        }
    }

    // Calcular o diâmetro efetivo (90º percentil) com interpolação
    double effectiveDiameter = 0.0;
    if (!shortestPaths.empty()) {
        std::sort(shortestPaths.begin(), shortestPaths.end());
        size_t N = shortestPaths.size();
        double rank = 0.9 * (N - 1);
        size_t idx = static_cast<size_t>(std::floor(rank));
        double frac = rank - idx;

        if (idx + 1 < N) {
            effectiveDiameter = shortestPaths[idx] * (1.0 - frac) + shortestPaths[idx + 1] * frac;
        } else {
            effectiveDiameter = shortestPaths[idx];
        }
    }

    return {diameter, effectiveDiameter};
}

// Função para calcular as métricas
json calculateMetrics() {
    auto [largestWCCNodes, largestWCCEdges] = calculateWCC();
    auto [largestSCCNodes, largestSCCEdges] = calculateSCC();
    auto [triangleCount, closedTriangleFraction] = calculateTriangles();
    double averageClusteringCoefficient = calculateAverageClusteringCoefficient();
    auto [diameter, effectiveDiameter] = calculateDiameters();

    return json{
        {"graph_metrics", {
            {"nodes", nodes.size()},
            {"edges", edges},
            {"largest_wcc", {
                {"nodes", largestWCCNodes},
                {"fraction_of_total_nodes", nodes.size() > 0 ? static_cast<double>(largestWCCNodes) / nodes.size() : 0.0},
                {"edges", largestWCCEdges},
                {"fraction_of_total_edges", edges > 0 ? static_cast<double>(largestWCCEdges) / edges : 0.0}
            }},
            {"largest_scc", {
                {"nodes", largestSCCNodes},
                {"fraction_of_total_nodes", nodes.size() > 0 ? static_cast<double>(largestSCCNodes) / nodes.size() : 0.0},
                {"edges", largestSCCEdges},
                {"fraction_of_total_edges", edges > 0 ? static_cast<double>(largestSCCEdges) / edges : 0.0}
            }},
            {"average_clustering_coefficient", averageClusteringCoefficient},
            {"triangles", triangleCount},
            {"fraction_of_closed_triangles", closedTriangleFraction},
            {"diameter", diameter},
            {"effective_diameter_90_percentile", effectiveDiameter}
        }}
    };
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size;
    int world_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Mensagem de início
    std::cout << "Processo " << world_rank << " de " << world_size << " iniciado." << std::endl << std::flush;

    if (argc < 2) {
        if (world_rank == 0) {
            std::cerr << "Uso: " << argv[0] << " <arquivo_de_entrada>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Leitura e distribuição dos dados
    std::vector<std::pair<int, int>> localGraphData;
    std::vector<std::pair<int, int>> graphData_send;
    std::vector<int> flatGraphData;

    if (world_rank == 0) {
        std::ifstream infile(argv[1]);
        if (!infile.is_open()) {
            std::cerr << "Erro: Não foi possível abrir o arquivo " << argv[1] << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::string line;
        while (std::getline(infile, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            int from, to;
            if (iss >> from >> to) {
                graphData_send.emplace_back(from, to);
            }
        }
        infile.close();
        std::cout << "Processo " << world_rank << ": Arquivo de entrada lido com " << graphData_send.size() << " arestas." << std::endl;

        // Flatten graphData_send into flatGraphData
        flatGraphData.reserve(graphData_send.size() * 2);
        for (const auto& edge : graphData_send) {
            flatGraphData.push_back(edge.first);
            flatGraphData.push_back(edge.second);
        }
    }

    // Broadcast o tamanho do graphData
    unsigned long long totalEdges;
    if (world_rank == 0) {
        totalEdges = graphData_send.size();
    }
    MPI_Bcast(&totalEdges, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    // Alocar espaço para todos os processos
    std::vector<int> flatGraphData_recv;
    if (world_rank != 0) {
        flatGraphData_recv.resize(totalEdges * 2);
    }

    // Broadcast de todo graphData
    if (world_rank == 0) {
        MPI_Bcast(flatGraphData.data(), totalEdges * 2, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(flatGraphData_recv.data(), totalEdges * 2, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Reconstruir localGraphData
    if (world_rank != 0) {
        for (unsigned long long i = 0; i < totalEdges; ++i) {
            localGraphData.emplace_back(flatGraphData_recv[2 * i], flatGraphData_recv[2 * i + 1]);
        }
    } else {
        for (unsigned long long i = 0; i < totalEdges; ++i) {
            localGraphData.emplace_back(flatGraphData[i * 2], flatGraphData[i * 2 + 1]);
        }
    }

    // Dividir o graphData entre os processos
    unsigned long long chunkSize = totalEdges / world_size;
    unsigned long long start = world_rank * chunkSize;
    unsigned long long end = (world_rank == world_size - 1) ? totalEdges : start + chunkSize;

    // Slice localGraphData para cada processo
    std::vector<std::pair<int, int>> slicedGraphData;
    if (start < end) {
        for (unsigned long long i = start; i < end; ++i) {
            slicedGraphData.emplace_back(localGraphData[i]);
        }
    }

    std::cout << "Processo " << world_rank << ": Processando arestas de " << start << " até " << end << "." << std::endl;

    // Cada processo carrega sua parte do grafo
    loadGraph(slicedGraphData);

    // Contagem de nós e arestas locais
    unsigned long long localNodeCount = nodes.size();
    unsigned long long localEdgeCount = edges;

    // Reduzir para obter a contagem total
    unsigned long long totalNodeCount = 0;
    unsigned long long totalEdgeCount = 0;
    MPI_Reduce(&localNodeCount, &totalNodeCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localEdgeCount, &totalEdgeCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Processo raiz recebe as listas de adjacência
    if (world_rank == 0) {
        // Agregar listas de adjacência de todos os processos
        for (int p = 1; p < world_size; ++p) {
            // Receber adjList
            unsigned long long adjListSize;
            MPI_Recv(&adjListSize, 1, MPI_UNSIGNED_LONG_LONG, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Processo 0: Recebido adjListSize = " << adjListSize << " do processo " << p << std::endl;

            for (unsigned long long i = 0; i < adjListSize; i++) {
                int node, neighborCount;
                MPI_Recv(&node, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&neighborCount, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::vector<int> neighbors(neighborCount);
                MPI_Recv(neighbors.data(), neighborCount, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                adjList[node].insert(adjList[node].end(), neighbors.begin(), neighbors.end());
            }

            // Receber reverseAdjList
            MPI_Recv(&adjListSize, 1, MPI_UNSIGNED_LONG_LONG, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Processo 0: Recebido reverseAdjListSize = " << adjListSize << " do processo " << p << std::endl;
            for (unsigned long long i = 0; i < adjListSize; i++) {
                int node, neighborCount;
                MPI_Recv(&node, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&neighborCount, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::vector<int> neighbors(neighborCount);
                MPI_Recv(neighbors.data(), neighborCount, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                reverseAdjList[node].insert(reverseAdjList[node].end(), neighbors.begin(), neighbors.end());
            }

            // Receber undirectedAdjList
            MPI_Recv(&adjListSize, 1, MPI_UNSIGNED_LONG_LONG, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Processo 0: Recebido undirectedAdjListSize = " << adjListSize << " do processo " << p << std::endl;
            for (unsigned long long i = 0; i < adjListSize; i++) {
                int node, neighborCount;
                MPI_Recv(&node, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&neighborCount, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::vector<int> neighbors(neighborCount);
                MPI_Recv(neighbors.data(), neighborCount, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                undirectedAdjList[node].insert(neighbors.begin(), neighbors.end());
            }
        }

        // Reconstruir o conjunto 'nodes' a partir de 'undirectedAdjList'
        nodes.clear();
        for (const auto& [node, neighbors] : undirectedAdjList) {
            nodes.insert(node);
            for (int neighbor : neighbors) {
                nodes.insert(neighbor);
            }
        }

        // Atualizar 'edges' para refletir o total de arestas
        edges = totalEdgeCount;

        // Mensagens de Depuração
        std::cout << "Processo 0: Total de nós reconstruídos = " << nodes.size() << std::endl;
        std::cout << "Processo 0: Total de arestas = " << edges << std::endl;

        // Calcular as métricas
        std::cout << "Processo 0: Calculando métricas do grafo." << std::endl;
        json result = calculateMetrics();
        std::ofstream outfile("/app/data/graph_metrics.json");
        outfile << result.dump(4);
        outfile.close();

        std::cout << "Processo 0: Métricas do grafo salvas em graph_metrics.json" << std::endl;
    } else {
        // Enviar adjList
        unsigned long long adjListSize = adjList.size();
        std::cout << "Processo " << world_rank << ": Enviando adjListSize = " << adjListSize << std::endl;
        MPI_Send(&adjListSize, 1, MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD);
        for (const auto& [node, neighbors] : adjList) {
            int node_id = node;
            int neighbor_count = neighbors.size();
            MPI_Send(&node_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&neighbor_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(neighbors.data(), neighbor_count, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }

        // Enviar reverseAdjList
        unsigned long long reverseAdjListSize = reverseAdjList.size();
        std::cout << "Processo " << world_rank << ": Enviando reverseAdjListSize = " << reverseAdjListSize << std::endl;
        MPI_Send(&reverseAdjListSize, 1, MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD);
        for (const auto& [node, neighbors] : reverseAdjList) {
            int node_id = node;
            int neighbor_count = neighbors.size();
            MPI_Send(&node_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&neighbor_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(neighbors.data(), neighbor_count, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }

        // Enviar undirectedAdjList
        unsigned long long undirectedAdjListSize = undirectedAdjList.size();
        std::cout << "Processo " << world_rank << ": Enviando undirectedAdjListSize = " << undirectedAdjListSize << std::endl;
        MPI_Send(&undirectedAdjListSize, 1, MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD);
        for (const auto& [node, neighbors] : undirectedAdjList) {
            int node_id = node;
            int neighbor_count = neighbors.size();
            MPI_Send(&node_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&neighbor_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            std::vector<int> neighbor_vec(neighbors.begin(), neighbors.end());
            MPI_Send(neighbor_vec.data(), neighbor_count, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }

        std::cout << "Processo " << world_rank << ": Listas de adjacência enviadas para processo raiz." << std::endl;
    }

    // Mensagem de finalização
    std::cout << "Processo " << world_rank << " finalizando." << std::endl;

    MPI_Finalize();
    return 0;
}
