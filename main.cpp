#include <iostream>
#include <fstream>
#include <pthread.h>
#include <vector>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <queue>
#include <stack>
#include <cmath>
#include <chrono>
#include <sys/resource.h>
#include <sstream>

using json = nlohmann::json;
pthread_mutex_t mutex_edges;
long long total_edges = 0;
long long total_triangles = 0;

struct ThreadData {
    int thread_id;
    long long start;
    long long end;
    std::unordered_map<int, std::set<int>>* adj_list;
};

void* process_graph(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    std::ifstream file("web-small.txt");
    if (!file.is_open()) {
        std::cerr << "Erro ao abrir o arquivo" << std::endl;
        return NULL;
    }

    file.seekg(data->start, std::ios::beg);
    std::string line;
    long long local_edges = 0;
    std::unordered_map<int, std::set<int>> local_adj_list;

    // Processa a parte do arquivo designada à thread
    for (long long i = 0; i < data->end - data->start; ++i) {
        if (std::getline(file, line)) {
            if (line[0] != '#') { // Ignora comentários
                int u, v;
                sscanf(line.c_str(), "%d %d", &u, &v);
                local_adj_list[u].insert(v);
                local_adj_list[v].insert(u); // Grafo não direcionado
                ++local_edges;
            }
        }
    }

    file.close();

    // Sincroniza as listas de adjacência globais
    pthread_mutex_lock(&mutex_edges);
    for (const auto& entry : local_adj_list) {
        (*data->adj_list)[entry.first].insert(entry.second.begin(), entry.second.end());
    }
    total_edges += local_edges;
    pthread_mutex_unlock(&mutex_edges);

    return NULL;
}

long long get_memory_usage_kb() {
    std::ifstream file("/proc/self/status");
    if (!file.is_open()) return 0;

    std::string line;
    while (std::getline(file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            long long memory;
            std::istringstream iss(line.substr(7));
            iss >> memory;
            file.close();
            return memory;
        }
    }

    file.close();
    return 0;
}

std::pair<int, int> largest_scc(const std::unordered_map<int, std::set<int>>& adj_list) {
    // Implementação de Tarjan ou Kosaraju para SCC
    // Adicione código específico aqui
    return {0, 0}; // Retorna nós e arestas no maior SCC
}
std::pair<int, int> largest_wcc(const std::unordered_map<int, std::set<int>>& adj_list) {
    std::unordered_map<int, bool> visited;
    int max_nodes = 0, max_edges = 0;

    for (const auto& node : adj_list) {
        if (!visited[node.first]) {
            int nodes = 0, edges = 0;
            std::queue<int> queue;
            queue.push(node.first);
            visited[node.first] = true;

            while (!queue.empty()) {
                int u = queue.front();
                queue.pop();
                ++nodes;
                edges += adj_list.at(u).size();

                for (int v : adj_list.at(u)) {
                    if (!visited[v]) {
                        visited[v] = true;
                        queue.push(v);
                    }
                }
            }

            max_nodes = std::max(max_nodes, nodes);
            max_edges = std::max(max_edges, edges / 2);
        }
    }

    return {max_nodes, max_edges};
}

// Função para calcular o coeficiente de agrupamento médio
double average_clustering_coefficient(const std::unordered_map<int, std::set<int>>& adj_list) {
    double total_coefficient = 0.0;
    int total_nodes = 0;

    for (const auto& node : adj_list) {
        int degree = node.second.size();
        if (degree < 2) continue;

        int triangles = 0;
        for (int neighbor1 : node.second) {
            for (int neighbor2 : node.second) {
                if (neighbor1 != neighbor2 && adj_list.at(neighbor1).count(neighbor2)) {
                    ++triangles;
                }
            }
        }
        total_coefficient += (triangles / 2.0) / (degree * (degree - 1) / 2.0);
        ++total_nodes;
    }

    return total_nodes ? total_coefficient / total_nodes : 0.0;
}

// Funções para diâmetro e diâmetro efetivo (90%)
// Implementação da busca BFS para encontrar distâncias máximas
int bfs_diameter(const std::unordered_map<int, std::set<int>>& adj_list, bool effective = false, double percentile = 0.9) {
    int max_diameter = 0;
    std::vector<int> all_distances;

    for (const auto& node : adj_list) {
        std::queue<std::pair<int, int>> queue;
        std::unordered_map<int, bool> visited;
        queue.push({node.first, 0});
        visited[node.first] = true;

        while (!queue.empty()) {
            auto [u, dist] = queue.front();
            queue.pop();
            all_distances.push_back(dist);
            max_diameter = std::max(max_diameter, dist);

            for (int v : adj_list.at(u)) {
                if (!visited[v]) {
                    visited[v] = true;
                    queue.push({v, dist + 1});
                }
            }
        }
    }

    if (effective) {
        std::sort(all_distances.begin(), all_distances.end());
        return all_distances[(int)(percentile * all_distances.size())];
    }

    return max_diameter;
}


// Função para calcular os triângulos
long long count_triangles(const std::unordered_map<int, std::set<int>>& adj_list) {
    long long local_triangles = 0;

    // Percorre todas as arestas
    for (const auto& node : adj_list) {
        int u = node.first;
        for (int v : node.second) {
            // Verifica se existe um vizinho comum entre u e v
            for (int w : adj_list.at(v)) {
                if (adj_list.at(u).count(w)) {
                    ++local_triangles; // Triângulo detectado
                }
            }
        }
    }
    return local_triangles / 3; // Cada triângulo será contado 3 vezes
}

int main() {
    int num_threads = 8;
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    long long total_lines = 0;
    std::string line;

    std::unordered_map<int, std::set<int>> adj_list;

    // Conta o número total de linhas no arquivo
    std::ifstream file("web-small.txt");
    if (!file.is_open()) {
        std::cerr << "Erro ao abrir o arquivo" << std::endl;
        return 1;
    }

    while (std::getline(file, line)) {
        ++total_lines;
    }
    file.close();

    long long lines_per_thread = total_lines / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].thread_id = i;
        thread_data[i].start = i * lines_per_thread;
        thread_data[i].end = (i == num_threads - 1) ? total_lines : (i + 1) * lines_per_thread;
        thread_data[i].adj_list = &adj_list;
        pthread_create(&threads[i], NULL, process_graph, (void *)&thread_data[i]);
    }

    // Espera todas as threads terminarem
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    // Conta os triângulos no grafo
    total_triangles = count_triangles(adj_list);

    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);

    long long memory_usage_kb = get_memory_usage_kb();
    double user_cpu_time = (usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6);
    double system_cpu_time = (usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6);
    double execution_time = (std::chrono::high_resolution_clock::now() - std::chrono::high_resolution_clock::now()).count() / 1e9;

    auto [nodes_scc, edges_scc] = largest_scc(adj_list);
    auto [nodes_wcc, edges_wcc] = largest_wcc(adj_list);
    double avg_clustering = average_clustering_coefficient(adj_list);
    int diameter = bfs_diameter(adj_list);
    int effective_diameter = bfs_diameter(adj_list, true, 0.9);
    double fraction_triangles_closed = total_triangles ? (3.0 * total_triangles / total_edges) : 0;

    // Gera JSON com os resultados
    json output = {
    {"largest_scc", {{"nodes", nodes_scc}, {"edges", edges_scc}}},
    {"largest_wcc", {{"nodes", nodes_wcc}, {"edges", edges_wcc}}},
    {"average_clustering_coefficient", avg_clustering},
    {"diameter", diameter},
    {"effective_diameter_90_percent", effective_diameter},
    {"fraction_triangles_closed", fraction_triangles_closed},
    {"number_of_triangles", total_triangles}
};

    std::ofstream json_file("results.json");
    if (json_file.is_open()) {
        json_file << output.dump(4); // Indentação de 4 espaços
        json_file.close();
    } else {
        std::cerr << "Erro ao abrir o arquivo JSON" << std::endl;
    }


    std::cout << "Total de arestas no grafo: " << total_edges << std::endl;
    std::cout << "Total de triângulos no grafo: " << total_triangles << std::endl;
    std::cout << "Tempo de execução: " << execution_time << " segundos" << std::endl;
    std::cout << "Tempo de CPU (usuário): " << user_cpu_time << " segundos" << std::endl;
    std::cout << "Tempo de CPU (sistema): " << system_cpu_time << " segundos" << std::endl;
    std::cout << "Consumo de memória: " << memory_usage_kb << " kB" << std::endl;

    return 0;
}
