#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <stack>
#include <cmath>
#include <pthread.h>
#include <mutex>
#include <nlohmann/json.hpp>
#include <algorithm>

using json = nlohmann::json;

// Estrutura para armazenar dados locais de cada thread durante o carregamento
struct LocalGraph {
    std::unordered_set<int> localNodes;
    std::unordered_map<int, std::vector<int>> localAdjList;
    std::unordered_map<int, std::vector<int>> localReverseAdjList;
    std::unordered_map<int, std::unordered_set<int>> localUndirectedAdjList;
    size_t localEdges = 0;
};

// Estrutura para argumentos das threads de carregamento do grafo
struct LoadArgs {
    size_t start;
    size_t end;
    LocalGraph* localGraph;
};

// Lista de adjacência global
std::unordered_set<int> nodes;
std::unordered_map<int, std::vector<int>> adjList;
std::unordered_map<int, std::vector<int>> reverseAdjList;
std::unordered_map<int, std::unordered_set<int>> undirectedAdjList;
size_t edges = 0;
std::vector<std::pair<int, int>> graphData;

// Mutex para sincronização durante a mesclagem (não é mais necessário para o carregamento)
std::mutex mtx;

// Função para carregar o grafo em paralelo usando Pthreads
void* loadGraph(void* arg) {
    LoadArgs* args = static_cast<LoadArgs*>(arg);
    size_t start = args->start;
    size_t end = args->end;
    LocalGraph* localGraph = args->localGraph;

    for (size_t i = start; i < end; ++i) {
        int from = graphData[i].first;
        int to = graphData[i].second;

        localGraph->localNodes.insert(from);
        localGraph->localNodes.insert(to);
        localGraph->localEdges++;

        localGraph->localAdjList[from].push_back(to);
        localGraph->localReverseAdjList[to].push_back(from);
        localGraph->localUndirectedAdjList[from].insert(to);
        localGraph->localUndirectedAdjList[to].insert(from);
    }

    pthread_exit(nullptr);
}

// Estrutura para argumentos das threads de contagem de triângulos
struct TriangleArgs {
    size_t start;
    size_t end;
    size_t localTriangles;
    const std::vector<int>* nodeList;
    const std::unordered_map<int, std::unordered_set<int>>* localUndirectedAdjList;
};

// Função executada por cada thread para contar triângulos
void* triangleThread(void* arg) {
    TriangleArgs* args = static_cast<TriangleArgs*>(arg);
    size_t localTriangles = 0;

    for (size_t i = args->start; i < args->end; ++i) {
        int u = (*(args->nodeList))[i];
        const auto& neighbors_u = args->localUndirectedAdjList->at(u);

        for (int v : neighbors_u) {
            if (v <= u) continue; // Garantir u < v
            const auto& neighbors_v = args->localUndirectedAdjList->at(v);
            for (int w : neighbors_u) {
                if (w <= v) continue; // Garantir v < w
                if (neighbors_v.find(w) != neighbors_v.end()) {
                    // Triângulo encontrado entre u, v, w
                    localTriangles++;
                }
            }
        }
    }

    args->localTriangles = localTriangles;
    pthread_exit(nullptr);
}

// Estrutura para argumentos das threads de cálculo do coeficiente de agrupamento
struct ClusteringArgs {
    size_t start;
    size_t end;
    double localClusteringSum;
    const std::vector<int>* nodeList;
    const std::unordered_map<int, std::unordered_set<int>>* localUndirectedAdjList;
};

// Função executada por cada thread para calcular o coeficiente de agrupamento
void* clusteringThread(void* arg) {
    ClusteringArgs* args = static_cast<ClusteringArgs*>(arg);
    double localSum = 0.0;

    for (size_t i = args->start; i < args->end; ++i) {
        int node = (*(args->nodeList))[i];
        const auto& neighbors = args->localUndirectedAdjList->at(node);
        size_t degree = neighbors.size();
        if (degree < 2) continue;

        size_t connectedPairs = 0;
        // Converter os vizinhos para um vetor para facilitar a iteração
        std::vector<int> neighborsVec(neighbors.begin(), neighbors.end());

        for (size_t j = 0; j < neighborsVec.size(); ++j) {
            for (size_t k = j + 1; k < neighborsVec.size(); ++k) {
                int neighbor1 = neighborsVec[j];
                int neighbor2 = neighborsVec[k];
                if (args->localUndirectedAdjList->at(neighbor1).find(neighbor2) != args->localUndirectedAdjList->at(neighbor1).end()) {
                    connectedPairs++;
                }
            }
        }

        double clusteringCoefficient = (2.0 * connectedPairs) / (degree * (degree - 1));
        localSum += clusteringCoefficient;
    }

    args->localClusteringSum = localSum;
    pthread_exit(nullptr);
}

// Estrutura para argumentos das threads de cálculo do diâmetro
struct DiameterArgs {
    int startNode;
    int localDiameter;
    std::vector<int> localShortestPaths;
};

// Função executada por cada thread para calcular o diâmetro
void* diameterThread(void* arg) {
    DiameterArgs* args = static_cast<DiameterArgs*>(arg);
    int startNode = args->startNode;
    std::unordered_map<int, int> distances;
    std::queue<int> q;

    distances[startNode] = 0;
    q.push(startNode);

    while (!q.empty()) {
        int curr = q.front();
        q.pop();

        for (int neighbor : undirectedAdjList.at(curr)) {
            if (distances.find(neighbor) == distances.end()) {
                distances[neighbor] = distances[curr] + 1;
                q.push(neighbor);
                args->localDiameter = std::max(args->localDiameter, distances[neighbor]);
                args->localShortestPaths.push_back(distances[neighbor]);
            }
        }
    }

    pthread_exit(nullptr);
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
            largestWCCEdges = componentEdges / 2;
        }
    }

    return {largestWCCNodes, largestWCCEdges};
}

// Função para calcular o maior componente fortemente conectado (SCC) usando Kosaraju
std::pair<size_t, size_t> calculateSCC() {
    std::unordered_set<int> visited;
    std::stack<int> finishStack;

    // 1ª Passagem: DFS no grafo original
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

// Função para calcular o número de triângulos e a fração de triângulos fechados
std::pair<size_t, double> calculateTriangles() {
    size_t triangles = 0;

    // Preparar a lista de nós
    std::vector<int> nodeListVec(nodes.begin(), nodes.end());
    size_t numNodes = nodeListVec.size();

    // Definir o número de threads
    int numThreads = 8;
    pthread_t threads[numThreads];
    TriangleArgs args[numThreads];

    // Calcular o tamanho do chunk para cada thread
    size_t chunkSize = numNodes / numThreads;

    // Criar threads
    for(int i = 0; i < numThreads; ++i){
        args[i].start = i * chunkSize;
        args[i].end = (i == numThreads -1) ? numNodes : (i +1) * chunkSize;
        args[i].localTriangles = 0;
        args[i].nodeList = &nodeListVec;
        args[i].localUndirectedAdjList = &undirectedAdjList;
        pthread_create(&threads[i], nullptr, triangleThread, &args[i]);
    }

    // Aguardar todas as threads terminarem e acumular os resultados
    for(int i = 0; i < numThreads; ++i){
        pthread_join(threads[i], nullptr);
        triangles += args[i].localTriangles;
    }

    // Calcular o número de tripletos conectados
    size_t triplets = 0;
    for (const auto& [node, neighbors] : undirectedAdjList) {
        size_t degree = neighbors.size();
        if (degree >=2) {
            triplets += degree * (degree -1) / 2;
        }
    }

    double closedTriangleFraction = (triplets > 0) ? (static_cast<double>(3 * triangles) / triplets) : 0.0;

    return {triangles, closedTriangleFraction};
}

// Função para calcular o coeficiente de agrupamento médio
double calculateAverageClusteringCoefficient() {
    double totalClusteringCoefficient = 0.0;
    size_t nodeCount = 0;

    // Preparar a lista de nós
    std::vector<int> nodeListVec(nodes.begin(), nodes.end());
    size_t numNodes = nodeListVec.size();

    // Definir o número de threads
    int numThreads = 8;
    pthread_t threads[numThreads];
    ClusteringArgs args[numThreads];

    // Calcular o tamanho do chunk para cada thread
    size_t chunkSize = numNodes / numThreads;

    // Criar threads
    for(int i = 0; i < numThreads; ++i){
        args[i].start = i * chunkSize;
        args[i].end = (i == numThreads -1) ? numNodes : (i +1) * chunkSize;
        args[i].localClusteringSum = 0.0;
        args[i].nodeList = &nodeListVec;
        args[i].localUndirectedAdjList = &undirectedAdjList;
        pthread_create(&threads[i], nullptr, clusteringThread, &args[i]);
    }

    // Aguardar todas as threads terminarem e acumular os resultados
    for(int i = 0; i < numThreads; ++i){
        pthread_join(threads[i], nullptr);
        totalClusteringCoefficient += args[i].localClusteringSum;

        // Calcular o número de nós com grau >= 2
        for(size_t j = args[i].start; j < args[i].end; ++j){
            int node = nodeListVec[j];
            size_t degree = undirectedAdjList.at(node).size();
            if(degree >=2){
                nodeCount++;
            }
        }
    }

    return (nodeCount > 0) ? (totalClusteringCoefficient / nodeCount) : 0.0;
}

// Função para calcular o diâmetro e o diâmetro efetivo
std::pair<int, double> calculateDiameters() {
    int diameter = 0;
    std::vector<int> shortestPaths;

    // Selecionar um subconjunto de nós para amostragem
    int numSamples = 100;
    std::vector<int> sampleNodes;
    size_t count =0;
    for(int node : nodes){
        sampleNodes.push_back(node);
        if(++count >= numSamples) break;
    }

    size_t numThreadTasks = sampleNodes.size();
    pthread_t threads[numThreadTasks];
    DiameterArgs args[numThreadTasks];

    // Criar threads para cada nó de amostra
    for(size_t i =0; i < numThreadTasks; ++i){
        args[i].startNode = sampleNodes[i];
        args[i].localDiameter = 0;
        pthread_create(&threads[i], nullptr, diameterThread, &args[i]);
    }

    // Aguardar todas as threads terminarem e acumular os resultados
    for(size_t i =0; i < numThreadTasks; ++i){
        pthread_join(threads[i], nullptr);
        diameter = std::max(diameter, args[i].localDiameter);
        shortestPaths.insert(shortestPaths.end(), args[i].localShortestPaths.begin(), args[i].localShortestPaths.end());
    }

    // Calcular o diâmetro efetivo (90º percentil)
    double effectiveDiameter =0.0;
    if(!shortestPaths.empty()){
        std::sort(shortestPaths.begin(), shortestPaths.end());
        size_t idx = static_cast<size_t>(0.9 * shortestPaths.size());
        if(idx >= shortestPaths.size()) idx = shortestPaths.size()-1;
        effectiveDiameter = shortestPaths[idx];
    }

    return {diameter, effectiveDiameter};
}

// Função para calcular o maior componente fracamente conectado (WCC)
std::pair<size_t, size_t> calculateWCC_single() { // Renomeado para evitar conflito
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
            largestWCCEdges = componentEdges / 2;
        }
    }

    return {largestWCCNodes, largestWCCEdges};
}

// Função para calcular o maior componente fortemente conectado (SCC) usando Kosaraju
std::pair<size_t, size_t> calculateSCC_single() { // Renomeado para evitar conflito
    std::unordered_set<int> visited;
    std::stack<int> finishStack;

    // 1ª Passagem: DFS no grafo original
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

// Função para calcular as métricas
json calculateMetrics() {
    auto [largestWCCNodes, largestWCCEdges] = calculateWCC_single();
    auto [largestSCCNodes, largestSCCEdges] = calculateSCC_single();
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

    int numThreads = 8;
    pthread_t loadThreads[numThreads];
    LoadArgs loadArgs[numThreads];
    LocalGraph localGraphs[numThreads];
    size_t chunkSize = graphData.size() / numThreads;

    // Criar threads para carregar o grafo
    for(int i =0; i < numThreads; ++i){
        loadArgs[i].start = i * chunkSize;
        loadArgs[i].end = (i == numThreads -1) ? graphData.size() : (i +1) * chunkSize;
        loadArgs[i].localGraph = &localGraphs[i];
        pthread_create(&loadThreads[i], nullptr, loadGraph, &loadArgs[i]);
    }

    // Aguardar todas as threads de carregamento terminarem
    for(int i =0; i < numThreads; ++i){
        pthread_join(loadThreads[i], nullptr);
    }

    // Mesclar os dados locais nas estruturas globais
    for(int i =0; i < numThreads; ++i){
        // Mesclar nós
        nodes.insert(localGraphs[i].localNodes.begin(), localGraphs[i].localNodes.end());

        // Mesclar adjList
        for(auto& [node, neighbors] : localGraphs[i].localAdjList){
            adjList[node].insert(adjList[node].end(), neighbors.begin(), neighbors.end());
        }

        // Mesclar reverseAdjList
        for(auto& [node, neighbors] : localGraphs[i].localReverseAdjList){
            reverseAdjList[node].insert(reverseAdjList[node].end(), neighbors.begin(), neighbors.end());
        }

        // Mesclar undirectedAdjList
        for(auto& [node, neighbors] : localGraphs[i].localUndirectedAdjList){
            undirectedAdjList[node].insert(neighbors.begin(), neighbors.end());
        }

        // Mesclar arestas
        edges += localGraphs[i].localEdges;
    }

    // Calcular as métricas
    json result = calculateMetrics();

    // Salvar as métricas em um arquivo JSON
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
