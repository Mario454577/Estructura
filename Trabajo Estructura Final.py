import csv
import heapq
from collections import deque
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
#Presentado por: Carlos, Mario y Valeria
def build_graph_from_csv(filename):
    """
    Construye un grafo no dirigido a partir del archivo CSV.
    Formato: {origen: {destino: {'distancia': km, 'tiempo': min}}}
    """
    graph = {}
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)  
        for row in reader:
            origen = row[0].strip()
            destino = row[1].strip()
            distancia = float(row[2].strip())
            tiempo = int(row[3].strip())

            if origen not in graph:
                graph[origen] = {}
            graph[origen][destino] = {'distancia': distancia, 'tiempo': tiempo}

            if destino not in graph:
                graph[destino] = {}
            graph[destino][origen] = {'distancia': distancia, 'tiempo': tiempo}
    return graph

def build_flow_graph(undirected_graph, capacity=150):
    """
    Construye un grafo dirigido para flujo máximo con capacidades en ambos sentidos
    """
    flow_graph = {}
    for u in undirected_graph:
        for v in undirected_graph[u]:
            if u not in flow_graph:
                flow_graph[u] = {}
            flow_graph[u][v] = capacity

            if v not in flow_graph:
                flow_graph[v] = {}
            flow_graph[v][u] = capacity
    return flow_graph

def dijkstra(graph, start, criterion='distancia'):
    dist = {node: float('inf') for node in graph}
    prev = {node: None for node in graph}
    dist[start] = 0
    
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_dist, current_node = heapq.heappop(priority_queue)
        
        if current_dist > dist[current_node]:
            continue
            
        for neighbor, attrs in graph[current_node].items():
            weight = attrs[criterion]
            new_dist = current_dist + weight
            
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = current_node
                heapq.heappush(priority_queue, (new_dist, neighbor))
    
    return dist, prev

def bellman_ford(graph, start, criterion='distancia'):
    dist = {node: float('inf') for node in graph}
    prev = {node: None for node in graph}
    dist[start] = 0
    
    nodes = list(graph.keys())
    for _ in range(len(nodes) - 1):
        for u in graph:
            for v, attrs in graph[u].items():
                weight = attrs[criterion]
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    prev[v] = u

    for u in graph:
        for v, attrs in graph[u].items():
            weight = attrs[criterion]
            if dist[u] + weight < dist[v]:
                raise ValueError("El grafo contiene un ciclo de peso negativo")
    
    return dist, prev

def floyd_warshall(graph, criterion='distancia'):
    nodes = sorted(graph.keys())
    node_index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
  
    dist = np.full((n, n), float('inf'))
    next_node = np.full((n, n), -1, dtype=np.int32)  # -1 representa None

    np.fill_diagonal(dist, 0)
    
    for u in graph:
        for v, attrs in graph[u].items():
            i = node_index[u]
            j = node_index[v]
            weight = attrs[criterion]
            dist[i, j] = float(weight)
            next_node[i, j] = j

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
                    next_node[i, j] = next_node[i, k]
    
    return dist, next_node, node_index

def reconstruct_path(prev, start, end):
    path = []
    current = end
    while current != start:
        if current is None:
            return []
        path.append(current)
        current = prev[current]
    path.append(start)
    path.reverse()
    return path

def reconstruct_path_fw(start_idx, end_idx, next_node, nodes):
    if next_node[start_idx][end_idx] is None:
        return []
    
    path = [nodes[start_idx]]
    while start_idx != end_idx:
        start_idx = next_node[start_idx][end_idx]
        path.append(nodes[start_idx])
    return path

def bfs_flow(graph, source, sink, parent):
    visited = {node: False for node in graph}
    queue = deque([source])
    visited[source] = True
    parent[source] = None
    
    while queue:
        u = queue.popleft()
        for v, capacity in graph.get(u, {}).items():
            if not visited[v] and capacity > 0:
                visited[v] = True
                parent[v] = u
                queue.append(v)
                if v == sink:
                    return True
    return False

def edmonds_karp(graph, source, sink):

    residual_graph = {u: {v: cap for v, cap in neighbors.items()} for u, neighbors in graph.items()}
    
    parent = {}
    max_flow = 0
    
    while bfs_flow(residual_graph, source, sink, parent):
        path_flow = float('inf')
        s = sink
        while s != source:
            u = parent[s]
            path_flow = min(path_flow, residual_graph[u][s])
            s = u
        
        v = sink
        while v != source:
            u = parent[v]
            residual_graph[u][v] -= path_flow
            if u not in residual_graph.get(v, {}):
                if v not in residual_graph:
                    residual_graph[v] = {}
                residual_graph[v][u] = 0
            residual_graph[v][u] += path_flow
            v = u
        
        max_flow += path_flow
    
    return max_flow