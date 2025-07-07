import networkx as nx
from collections import deque
import heapq
from typing import Dict, List, Tuple, Optional, Any, Union, TypeVar, Generic

T = TypeVar('T')

class Matrix(Generic[T]):
    def __init__(self, rows: int, cols: int, default_value: T):
        self.data: List[List[T]] = [[default_value for _ in range(cols)] for _ in range(rows)]
        self.rows = rows
        self.cols = cols

    def __getitem__(self, key: Tuple[int, int]) -> T:
        i, j = key
        return self.data[i][j]

    def __setitem__(self, key: Tuple[int, int], value: T) -> None:
        i, j = key
        self.data[i][j] = value

class RouteCalculator:
    def __init__(self, graph_data: Dict[str, Dict[str, Dict[str, float]]]):
        """
        Inicializa el calculador de rutas con los datos del grafo
        graph_data: diccionario con formato {origen: {destino: {'distancia': float, 'tiempo': int}}}
        """
        self.graph = graph_data
        self.flow_graph: Dict[str, Dict[str, float]] = {}
        self.FLOW_CAPACITY = 150.0

    def dijkstra(self, start: str, end: str, criterion: str = 'distancia') -> Tuple[Optional[List[str]], float]:
        """Implementación mejorada de Dijkstra"""
        if start not in self.graph or end not in self.graph:
            return None, float('inf')

        dist: Dict[str, float] = {node: float('inf') for node in self.graph}
        prev: Dict[str, Optional[str]] = {node: None for node in self.graph}
        dist[start] = 0
        
        pq: List[Tuple[float, str]] = [(0, start)]
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current == end:
                break
                
            if current_dist > dist[current]:
                continue
                
            for neighbor, attrs in self.graph[current].items():
                weight = attrs[criterion]
                distance = current_dist + weight
                
                if distance < dist[neighbor]:
                    dist[neighbor] = distance
                    prev[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        # Reconstruir ruta
        if dist[end] == float('inf'):
            return None, float('inf')
            
        path: List[str] = []
        current = end
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()
        
        return path, dist[end]

    def bellman_ford(self, start: str, end: str, criterion: str = 'distancia') -> Tuple[Optional[List[str]], float]:
        """Implementación mejorada de Bellman-Ford"""
        if start not in self.graph or end not in self.graph:
            return None, float('inf')

        dist: Dict[str, float] = {node: float('inf') for node in self.graph}
        prev: Dict[str, Optional[str]] = {node: None for node in self.graph}
        dist[start] = 0
        
        # Relajación de aristas
        for _ in range(len(self.graph) - 1):
            for u in self.graph:
                for v, attrs in self.graph[u].items():
                    weight = attrs[criterion]
                    if dist[u] + weight < dist[v]:
                        dist[v] = dist[u] + weight
                        prev[v] = u
        
        # Detección de ciclos negativos
        for u in self.graph:
            for v, attrs in self.graph[u].items():
                weight = attrs[criterion]
                if dist[u] + weight < dist[v]:
                    raise ValueError("El grafo contiene un ciclo de peso negativo")
        
        # Reconstruir ruta
        if dist[end] == float('inf'):
            return None, float('inf')
            
        path: List[str] = []
        current = end
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()
        
        return path, dist[end]

    def floyd_warshall(self, start: str, end: str, criterion: str = 'distancia') -> Tuple[Optional[List[str]], float]:
        """Implementación de Floyd-Warshall usando una clase Matrix personalizada"""
        if start not in self.graph or end not in self.graph:
            return None, float('inf')

        # Crear diccionario de índices
        nodes = list(self.graph.keys())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        
        # Inicializar matrices usando la clase Matrix
        dist = Matrix[float](n, n, float('inf'))
        next_node = Matrix[Optional[int]](n, n, None)
        
        # Configurar distancias iniciales
        for i in range(n):
            dist[(i, i)] = 0.0
            
        for u in self.graph:
            for v, attrs in self.graph[u].items():
                i, j = node_to_idx[u], node_to_idx[v]
                dist[(i, j)] = float(attrs[criterion])
                next_node[(i, j)] = j
        
        # Algoritmo principal
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[(i, k)] + dist[(k, j)] < dist[(i, j)]:
                        dist[(i, j)] = dist[(i, k)] + dist[(k, j)]
                        next_node[(i, j)] = next_node[(i, k)]
        
        # Reconstruir ruta
        start_idx = node_to_idx[start]
        end_idx = node_to_idx[end]
        
        if dist[(start_idx, end_idx)] == float('inf'):
            return None, float('inf')
            
        path: List[str] = []
        current = start_idx
        
        while current != end_idx:
            next_idx = next_node[(current, end_idx)]
            if next_idx is None:
                return None, float('inf')
            path.append(nodes[current])
            current = next_idx
        path.append(nodes[end_idx])
        
        return path, dist[(start_idx, end_idx)]

    def build_flow_graph(self, start: str, end: str) -> Dict[str, Dict[str, float]]:
        """
        Construye un grafo de flujo dirigido basado en el origen y destino seleccionados.
        Los arcos se redirigen para favorecer el flujo desde el origen hasta el destino.
        """
        self.flow_graph = {}
        
        # Primero, crear todas las aristas con capacidad estándar
        for u in self.graph:
            if u not in self.flow_graph:
                self.flow_graph[u] = {}
            for v in self.graph[u]:
                if v not in self.flow_graph:
                    self.flow_graph[v] = {}
                # Inicialmente todas las aristas son bidireccionales
                self.flow_graph[u][v] = float(self.FLOW_CAPACITY)
                self.flow_graph[v][u] = float(self.FLOW_CAPACITY)
        
        # Usar Dijkstra para encontrar el camino más corto
        path, _ = self.dijkstra(start, end)
        
        if path:
            # Favorecer las aristas en el camino más corto
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                # Aumentar capacidad en la dirección del flujo
                self.flow_graph[u][v] = float(self.FLOW_CAPACITY * 2)
                # Reducir capacidad en la dirección opuesta
                self.flow_graph[v][u] = float(self.FLOW_CAPACITY / 2)
        
        return self.flow_graph

    def edmonds_karp(self, start: str, end: str) -> Tuple[List[Tuple[List[str], float]], float]:
        """
        Implementación mejorada de Edmonds-Karp con visualización de flujo
        """
        if not self.flow_graph:
            self.build_flow_graph(start, end)
            
        if start not in self.flow_graph or end not in self.flow_graph:
            return [], 0.0
            
        def bfs() -> Optional[Dict[str, Optional[str]]]:
            visited: Dict[str, bool] = {node: False for node in self.flow_graph}
            parent: Dict[str, Optional[str]] = {node: None for node in self.flow_graph}
            visited[start] = True
            queue: deque = deque([start])
            
            while queue and not visited[end]:
                u = queue.popleft()
                for v, capacity in self.flow_graph[u].items():
                    if not visited[v] and capacity > 0:
                        visited[v] = True
                        parent[v] = u
                        queue.append(v)
            
            return parent if visited[end] else None
        
        max_flow = 0.0
        flow_paths: List[Tuple[List[str], float]] = []
        
        while True:
            parent = bfs()
            if not parent:
                break
                
            path_flow = float('inf')
            path: List[str] = []
            v = end
            
            while v != start:
                u = parent[v]
                if u is not None:  # Verificación de tipo
                    path_flow = min(path_flow, self.flow_graph[u][v])
                    path.append(v)
                    v = u
            path.append(start)
            path.reverse()
            
            flow_paths.append((path, path_flow))
            max_flow += path_flow
            
            v = end
            while v != start:
                u = parent[v]
                if u is not None:  # Verificación de tipo
                    self.flow_graph[u][v] -= path_flow
                    self.flow_graph[v][u] += path_flow
                    v = u
        
        return flow_paths, max_flow

    def calculate_route(self, start: str, end: str, algorithm: str, criterion: str = 'distancia') -> Tuple[Union[List[str], List[Tuple[List[str], float]]], float, Optional[float]]:
        """
        Calcula la ruta usando el algoritmo especificado
        Retorna: (ruta/rutas_flujo, distancia/flujo, tiempo si aplica)
        """
        if algorithm == 'edmonds_karp':
            flow_paths, max_flow = self.edmonds_karp(start, end)
            return flow_paths, max_flow, None
            
        path: Optional[List[str]] = None
        distance: float = float('inf')
        
        if algorithm == 'dijkstra':
            path, distance = self.dijkstra(start, end, criterion)
        elif algorithm == 'bellman_ford':
            path, distance = self.bellman_ford(start, end, criterion)
        elif algorithm == 'floyd_warshall':
            path, distance = self.floyd_warshall(start, end, criterion)
        else:
            raise ValueError(f"Algoritmo no soportado: {algorithm}")
            
        if path is None:
            return [], float('inf'), None
            
        time = self.get_route_time(path)
        return path, distance, time

    def get_route_time(self, path: List[str]) -> float:
        """Calcula el tiempo total de una ruta"""
        if not path or len(path) < 2:
            return 0.0
            
        total_time = 0.0
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            total_time += float(self.graph[u][v]['tiempo'])
        return total_time 