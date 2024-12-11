#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <stack>
#include <cmath>
#include <omp.h>
#include <nlohmann/json.hpp>
#include <algorithm> // Para std::sort

using json = nlohmann::json;

// Estruturas globais
std::unordered_set<int> nodes;
std::unordered_map<int, std::vector<int>> adjList;
std::unordered_map<int, std::vector<int>> reverseAdjList;
size_t edges = 0;
std::vector<std::pair<int, int>> graphData;

// Lista de adjacência não direcionada
std::unordered_map<int, std::unordered_set<int>> undirectedAdjList;

// Função para carregar o grafo
void loadGraph(size_t start, size_t end) {
    std::unordered_set<int> localNodes;
    size_t localEdges = 0;

    for (size_t i = start; i < end; ++i) {
        int from = graphData[i].first;
        int to = graphData[i].second;

        localNodes.insert(from);
        localNodes.insert(to);
        localEdges++;

        #pragma omp critical
        {
            adjList[from].push_back(to);
            reverseAdjList[to].push_back(from);
            undirectedAdjList[from].insert(to);
            undirectedAdjList[to].insert(from);
        }
    }

    #pragma omp critical
    {
        nodes.insert(localNodes.begin(), localNodes.end());
        edges += localEdges;
    }
}

// Função para calcular o maior componente fracamente conectado (WCC)
std::pair<size_t, size_t> calculateWCC() {
    std::unordered_set<int> visited;
    size_t largestWCCNodes = 0, largestWCCEdges = 0;

    std::vector<int> nodeList(nodes.begin(), nodes.end());

    for (size_t i = 0; i < nodeList.size(); ++i) {
        int node = nodeList[i];
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
            largestWCCEdges = componentEdges / 2;
        }
    }

    return {largestWCCNodes, largestWCCEdges};
}

// Função para calcular o maior componente fortemente conectado (SCC) usando Kosaraju
std::pair<size_t, size_t> calculateSCC() {
    std::unordered_set<int> visited;
    std::stack<int> finishStack;

    std::vector<int> nodeList(nodes.begin(), nodes.end());

    // 1ª Passagem: DFS no grafo original
    for (size_t i = 0; i < nodeList.size(); ++i) {
        int node = nodeList[i];
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

    // 2ª Passagem: DFS no grafo reverso
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

            if (componentNodes > largestSCCNodes) {
                largestSCCNodes = componentNodes;
                largestSCCEdges = componentEdges;
                largestSCCSet = componentSet;
            }
        }
    }

    // Recontar as arestas internas ao SCC
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

// Função para calcular o número de triângulos e a fração de triângulos fechados
std::pair<size_t, double> calculateTriangles() {
    size_t triangles = 0;

    // Converter as chaves do unordered_map para um vetor para facilitar a paralelização
    std::vector<int> nodeList;
    nodeList.reserve(undirectedAdjList.size());
    for (const auto& pair : undirectedAdjList) {
        nodeList.push_back(pair.first);
    }

    size_t numNodes = nodeList.size();

    // Paralelizar a contagem de triângulos
    #pragma omp parallel for reduction(+:triangles) schedule(dynamic)
    for (size_t i = 0; i < numNodes; ++i) {
        int u = nodeList[i];
        const auto& neighbors_u = undirectedAdjList[u];

        for (int v : neighbors_u) {
            if (v <= u) continue; // Garantir que cada par seja considerado apenas uma vez
            const auto& neighbors_v = undirectedAdjList[v];
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

    #pragma omp parallel for reduction(+:triplets) schedule(dynamic)
    for (size_t i = 0; i < numNodes; ++i) {
        int node = nodeList[i];
        size_t degree = undirectedAdjList[node].size();
        if (degree >= 2) {
            triplets += degree * (degree - 1) / 2;
        }
    }

    double closedTriangleFraction = (triplets > 0) ? (static_cast<double>(3 * triangles) / triplets) : 0.0;

    return {triangles, closedTriangleFraction};
}

// Função para calcular o coeficiente de agrupamento médio
double calculateAverageClusteringCoefficient() {
    double totalClusteringCoefficient = 0.0;
    size_t nodeCount = 0;

    // Converter as chaves do unordered_map para um vetor para facilitar a paralelização
    std::vector<int> nodeList;
    nodeList.reserve(undirectedAdjList.size());
    for (const auto& pair : undirectedAdjList) {
        nodeList.push_back(pair.first);
    }

    size_t numNodes = nodeList.size();

    #pragma omp parallel for reduction(+:totalClusteringCoefficient, nodeCount) schedule(dynamic)
    for (size_t i = 0; i < numNodes; ++i) {
        int node = nodeList[i];
        const auto& neighbors = undirectedAdjList[node];
        size_t degree = neighbors.size();
        if (degree < 2) continue;

        size_t connectedNeighborPairs = 0;
        // Converter os neighbors para um vetor para facilitar a iteração
        std::vector<int> neighborsVec(neighbors.begin(), neighbors.end());

        for (size_t j = 0; j < neighborsVec.size(); ++j) {
            for (size_t k = j + 1; k < neighborsVec.size(); ++k) {
                int neighbor1 = neighborsVec[j];
                int neighbor2 = neighborsVec[k];
                if (undirectedAdjList[neighbor1].find(neighbor2) != undirectedAdjList[neighbor1].end()) {
                    connectedNeighborPairs++;
                }
            }
        }

        double clusteringCoefficient = (2.0 * connectedNeighborPairs) / (degree * (degree - 1));
        totalClusteringCoefficient += clusteringCoefficient;
        nodeCount++;
    }

    return (nodeCount > 0) ? (totalClusteringCoefficient / nodeCount) : 0.0;
}

// Função para calcular o diâmetro e o diâmetro efetivo
std::pair<int, double> calculateDiameters() {
    int diameter = 0;
    std::vector<int> shortestPaths;

    // Realizar BFS a partir de um subconjunto de nós (para desempenho)
    int numSamples = 100;
    std::vector<int> sampleNodes;
    {
        // Selecionar até numSamples nós
        size_t count = 0;
        for (int node : nodes) {
            sampleNodes.push_back(node);
            if (++count >= numSamples) break;
        }
    }

    size_t numSampleNodes = sampleNodes.size();

    // Criar vetores para armazenar os resultados parciais
    std::vector<int> localDiameters(numSampleNodes, 0);
    std::vector<std::vector<int>> localShortestPaths(numSampleNodes);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < numSampleNodes; ++i) {
        int startNode = sampleNodes[i];
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
                    // Atualizar o diâmetro local
                    if (distances[neighbor] > localDiameters[i]) {
                        localDiameters[i] = distances[neighbor];
                    }
                    localShortestPaths[i].push_back(distances[neighbor]);
                }
            }
        }
    }

    // Agregar os resultados
    for (size_t i = 0; i < numSampleNodes; ++i) {
        diameter = std::max(diameter, localDiameters[i]);
        shortestPaths.insert(shortestPaths.end(), localShortestPaths[i].begin(), localShortestPaths[i].end());
    }

    // Calcular o diâmetro efetivo (90º percentil)
    double effectiveDiameter = 0.0;
    if (!shortestPaths.empty()) {
        std::sort(shortestPaths.begin(), shortestPaths.end());
        size_t idx = static_cast<size_t>(0.9 * shortestPaths.size());
        if(idx >= shortestPaths.size()) idx = shortestPaths.size() - 1;
        effectiveDiameter = shortestPaths[idx];
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
                {"fraction_of_total_nodes", (nodes.size() > 0) ? static_cast<double>(largestWCCNodes) / nodes.size() : 0.0},
                {"edges", largestWCCEdges},
                {"fraction_of_total_edges", (edges > 0) ? static_cast<double>(largestWCCEdges) / edges : 0.0}
            }},
            {"largest_scc", {
                {"nodes", largestSCCNodes},
                {"fraction_of_total_nodes", (nodes.size() > 0) ? static_cast<double>(largestSCCNodes) / nodes.size() : 0.0},
                {"edges", largestSCCEdges},
                {"fraction_of_total_edges", (edges > 0) ? static_cast<double>(largestSCCEdges) / edges : 0.0}
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
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <arquivo_de_entrada>" << std::endl;
        return 1;
    }

    std::ifstream infile(argv[1]);
    if (!infile.is_open()) {
        std::cerr << "Erro: Não foi possível abrir o arquivo " << argv[1] << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        int from, to;
        if (iss >> from >> to) {
            graphData.emplace_back(from, to);
        }
    }
    infile.close();

    size_t numThreads = 8; // Ajuste conforme o número de núcleos disponíveis
    size_t chunkSize = graphData.size() / numThreads;

    // Paralelizar o carregamento do grafo
    #pragma omp parallel num_threads(numThreads)
    {
        int thread_id = omp_get_thread_num();
        size_t start = thread_id * chunkSize;
        size_t end = (thread_id == numThreads - 1) ? graphData.size() : start + chunkSize;
        loadGraph(start, end);
    }

    json result = calculateMetrics();
    std::ofstream outfile("/app/data/graph_metrics.json"); // Ajuste o caminho conforme necessário
    if (!outfile.is_open()) {
        std::cerr << "Erro: Não foi possível criar o arquivo de saída." << std::endl;
        return 1;
    }
    outfile << result.dump(4);
    outfile.close();

    std::cout << "Métricas do grafo salvas em graph_metrics.json" << std::endl;

    return 0;
}
