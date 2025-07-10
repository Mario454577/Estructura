import networkx as nx
from collections import deque
import heapq
from typing import Dict, List, Set, Tuple, Optional, Any, Union, TypeVar, Generic, TypedDict, cast, Mapping
import math
from collections import defaultdict

T = TypeVar('T')

class Matrix(Generic[T]):
    def __init__(self, rows: int, cols: int, default_value: T):
        self.rows = rows
        self.cols = cols
        self.data = [[default_value for _ in range(cols)] for _ in range(rows)]
    
    def __getitem__(self, key: Tuple[int, int]) -> T:
        i, j = key
        if 0 <= i < self.rows and 0 <= j < self.cols:
            return self.data[i][j]
        raise IndexError("Matrix index out of range")
    
    def __setitem__(self, key: Tuple[int, int], value: T) -> None:
        i, j = key
        if 0 <= i < self.rows and 0 <= j < self.cols:
            self.data[i][j] = value
        else:
            raise IndexError("Matrix index out of range")

class FlowPath(TypedDict):
    path: List[str]
    flow: float

class Graph:
    def __init__(self):
        self.graph: Dict[str, List[Tuple[str, float, float]]] = defaultdict(list)
        self.vertices: Set[str] = set()
        
    def add_edge(self, u: str, v: str, weight: float, time: float) -> None:
        """Añade una arista al grafo con peso (distancia) y tiempo"""
        # Validar los datos de entrada
        if not isinstance(weight, (int, float)) or not isinstance(time, (int, float)):
            raise ValueError(f"Peso y tiempo deben ser números. Recibidos: peso={type(weight)}, tiempo={type(time)}")
        
        if weight < 0 or time < 0:
            raise ValueError(f"Peso y tiempo deben ser positivos. Recibidos: peso={weight}, tiempo={time}")
            
        # Verificar si la conexión ya existe
        exists = False
        for i, (dest, w, t) in enumerate(self.graph[u]):
            if dest == v:
                exists = True
                # Actualizar con el menor peso/tiempo si ya existe
                if weight < w or (weight == w and time < t):
                    self.graph[u][i] = (v, weight, time)
                break
        
        if not exists:
            self.graph[u].append((v, weight, time))
            
        self.vertices.add(u)
        self.vertices.add(v)
        
    def get_vertices(self) -> Set[str]:
        """Retorna el conjunto de vértices del grafo"""
        return self.vertices
        
    def get_neighbors(self, vertex: str) -> List[Tuple[str, float, float]]:
        """Retorna la lista de vecinos de un vértice con sus pesos y tiempos"""
        return self.graph.get(vertex, [])

class RouteCalculator:
    def __init__(self, graph: Graph):
        self.graph = graph

    def calculate_shortest_path(self, start: str, end: str, algorithm: str = 'dijkstra', criterion: str = 'distancia') -> Tuple[List[str], float, float]:
        """
        Calcula la ruta más corta usando el algoritmo especificado
        """
        if algorithm == 'dijkstra':
            return self._dijkstra(start, end, criterion)
        elif algorithm == 'bellman_ford':
            return self._bellman_ford(start, end, criterion)
        elif algorithm == 'floyd_warshall':
            return self._floyd_warshall(start, end, criterion)
        elif algorithm == 'a_star':
            return self._a_star(start, end, criterion)
        elif algorithm == 'johnson':
            return self._johnson(start, end, criterion)
        else:
            raise ValueError(f"Algoritmo no soportado: {algorithm}")

    def calculate_max_flow(self, source: str, sink: str, algorithm: str = 'edmonds_karp') -> Tuple[float, List[FlowPath]]:
        """Calcula el flujo máximo entre dos puntos usando el algoritmo especificado."""
        if algorithm == 'ford_fulkerson':
            return self._ford_fulkerson(source, sink)
        elif algorithm == 'edmonds_karp':
            return self._edmonds_karp(source, sink)
        elif algorithm == 'push_relabel':
            return self._push_relabel(source, sink)
        else:
            raise ValueError(f'Algoritmo de flujo máximo no válido: {algorithm}')

    def _dijkstra(self, start: str, end: str, criterion: str = 'distancia') -> Tuple[List[str], float, float]:
        """Implementación del algoritmo de Dijkstra"""
        # Inicializar estructuras de datos
        distances: Dict[str, float] = {vertex: float('infinity') for vertex in self.graph.vertices}
        times: Dict[str, float] = {vertex: float('infinity') for vertex in self.graph.vertices}
        distances[start] = 0
        times[start] = 0
        pq: List[Tuple[float, str]] = [(0, start)]
        previous: Dict[str, Optional[str]] = {vertex: None for vertex in self.graph.vertices}
        visited = set()

        while pq:
            current_value, current_vertex = heapq.heappop(pq)
            
            if current_vertex in visited:
                continue
                
            visited.add(current_vertex)

            if current_vertex == end:
                break

            if current_vertex not in self.graph.graph:
                continue

            for neighbor, weight, time in self.graph.graph[current_vertex]:
                if neighbor in visited:
                    continue
                    
                if criterion == 'distancia':
                    new_distance = distances[current_vertex] + weight
                    corresponding_time = times[current_vertex] + time
                    
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        times[neighbor] = corresponding_time
                        previous[neighbor] = current_vertex
                        heapq.heappush(pq, (new_distance, neighbor))
                else:  # criterion == 'tiempo'
                    new_time = times[current_vertex] + time
                    corresponding_distance = distances[current_vertex] + weight
                    
                    if new_time < times[neighbor]:
                        times[neighbor] = new_time
                        distances[neighbor] = corresponding_distance
                        previous[neighbor] = current_vertex
                        heapq.heappush(pq, (new_time, neighbor))

        # Reconstruir el camino
        path: List[str] = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()

        if not path or path[0] != start:
            return [], float('infinity'), float('infinity')

        return path, distances[end], times[end]

    def _bellman_ford(self, start: str, end: str, criterion: str = 'distancia') -> Tuple[List[str], float, float]:
        """Implementación del algoritmo Bellman-Ford"""
        distances: Dict[str, float] = {vertex: float('infinity') for vertex in self.graph.vertices}
        times: Dict[str, float] = {vertex: float('infinity') for vertex in self.graph.vertices}
        distances[start] = 0
        times[start] = 0
        previous: Dict[str, Optional[str]] = {vertex: None for vertex in self.graph.vertices}

        for _ in range(len(self.graph.vertices) - 1):
            for u in self.graph.vertices:
                for v, weight, time in self.graph.graph[u]:
                    if criterion == 'distancia':
                        if distances[u] + weight < distances[v]:
                            distances[v] = distances[u] + weight
                            times[v] = times[u] + time
                            previous[v] = u
                    else:  # criterion == 'tiempo'
                        if times[u] + time < times[v]:
                            distances[v] = distances[u] + weight
                            times[v] = times[u] + time
                            previous[v] = u

        for u in self.graph.vertices:
            for v, weight, time in self.graph.graph[u]:
                if criterion == 'distancia':
                    if distances[u] + weight < distances[v]:
                        raise ValueError("El grafo contiene un ciclo negativo")
                else:
                    if times[u] + time < times[v]:
                        raise ValueError("El grafo contiene un ciclo negativo")

        path: List[str] = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()

        return path, distances[end], times[end]

    def _floyd_warshall(self, start: str, end: str, criterion: str = 'distancia') -> Tuple[List[str], float, float]:
        """Implementación del algoritmo Floyd-Warshall"""
        vertices = list(self.graph.vertices)
        n = len(vertices)
        vertex_to_index = {vertex: i for i, vertex in enumerate(vertices)}
        

        dist_matrix: List[List[float]] = [[float('infinity')] * n for _ in range(n)]
        time_matrix: List[List[float]] = [[float('infinity')] * n for _ in range(n)]
        next_vertex: List[List[Optional[int]]] = [[None] * n for _ in range(n)]

        for i in range(n):
            dist_matrix[i][i] = 0.0
            time_matrix[i][i] = 0.0
            next_vertex[i][i] = i

        for u in self.graph.vertices:
            for v, weight, time in self.graph.graph[u]:
                i, j = vertex_to_index[u], vertex_to_index[v]
                dist_matrix[i][j] = float(weight)
                time_matrix[i][j] = float(time)
                next_vertex[i][j] = j

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if criterion == 'distancia':
                        if dist_matrix[i][k] + dist_matrix[k][j] < dist_matrix[i][j]:
                            dist_matrix[i][j] = dist_matrix[i][k] + dist_matrix[k][j]
                            time_matrix[i][j] = time_matrix[i][k] + time_matrix[k][j]
                            next_vertex[i][j] = next_vertex[i][k]
                    else: 
                        if time_matrix[i][k] + time_matrix[k][j] < time_matrix[i][j]:
                            dist_matrix[i][j] = dist_matrix[i][k] + dist_matrix[k][j]
                            time_matrix[i][j] = time_matrix[i][k] + time_matrix[k][j]
                            next_vertex[i][j] = next_vertex[i][k]

        start_idx = vertex_to_index[start]
        end_idx = vertex_to_index[end]
        
        if next_vertex[start_idx][end_idx] is None:
            return [], float('infinity'), float('infinity')

        path: List[str] = [start]
        current = start_idx
        while current != end_idx:
            next_idx = next_vertex[current][end_idx]
            if next_idx is None:
                return [], float('infinity'), float('infinity')
            current = next_idx
            path.append(vertices[current])

        return path, dist_matrix[start_idx][end_idx], time_matrix[start_idx][end_idx]

    def _a_star(self, start: str, end: str, criterion: str = 'distancia') -> Tuple[List[str], float, float]:
        """Implementación del algoritmo A*"""
        def heuristic(node1: str, node2: str) -> float:
            # Heurística simple basada en la distancia directa entre nodos
            return 0  # Podríamos mejorar esto con coordenadas reales

        open_set: Set[str] = {start}
        closed_set: Set[str] = set()
        
        g_score: Dict[str, float] = {vertex: float('infinity') for vertex in self.graph.vertices}
        g_score[start] = 0
        
        f_score: Dict[str, float] = {vertex: float('infinity') for vertex in self.graph.vertices}
        f_score[start] = heuristic(start, end)
        
        came_from: Dict[str, Optional[str]] = {vertex: None for vertex in self.graph.vertices}
        times: Dict[str, float] = {vertex: float('infinity') for vertex in self.graph.vertices}
        times[start] = 0

        while open_set:
            current = min(open_set, key=lambda x: f_score[x])
            
            if current == end:
                path: List[str] = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path, g_score[end], times[end]

            open_set.remove(current)
            closed_set.add(current)

            for neighbor, weight, time in self.graph.graph[current]:
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score[current] + (weight if criterion == 'distancia' else time)
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score[neighbor]:
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                times[neighbor] = times[current] + time
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)

        return [], float('infinity'), float('infinity')

    def _johnson(self, start: str, end: str, criterion: str = 'distancia') -> Tuple[List[str], float, float]:
        """Implementación del algoritmo de Johnson"""
        # Añadir un nuevo vértice q y conectarlo a todos los demás con peso 0
        q = "q_temp"
        original_vertices = self.graph.vertices.copy()
        self.graph.vertices.add(q)
        for v in original_vertices:
            self.graph.graph[q].append((v, 0, 0))

        # Ejecutar Bellman-Ford desde q para obtener los potenciales h(v)
        h: Dict[str, float] = {vertex: float('infinity') for vertex in self.graph.vertices}
        h[q] = 0

        # Relajación de aristas
        for _ in range(len(self.graph.vertices) - 1):
            for u in self.graph.vertices:
                for v, weight, _ in self.graph.graph[u]:
                    if h[u] + weight < h[v]:
                        h[v] = h[u] + weight

        # Verificar ciclos negativos
        for u in self.graph.vertices:
            for v, weight, _ in self.graph.graph[u]:
                if h[u] + weight < h[v]:
                    raise ValueError("El grafo contiene un ciclo negativo")

        # Eliminar el vértice q
        self.graph.vertices.remove(q)
        del self.graph.graph[q]

        # Recalcular pesos usando los potenciales
        reweighted_graph = Graph()
        for u in original_vertices:
            for v, weight, time in self.graph.graph[u]:
                new_weight = weight + h[u] - h[v]
                reweighted_graph.add_edge(u, v, new_weight, time)

        # Ejecutar Dijkstra con los nuevos pesos
        temp_calculator = RouteCalculator(reweighted_graph)
        path, distance, time = temp_calculator._dijkstra(start, end, criterion)

        # Ajustar la distancia final
        if criterion == 'distancia':
            distance = distance - h[start] + h[end]

        return path, distance, time

    def _ford_fulkerson(self, source: str, sink: str) -> Tuple[float, List[FlowPath]]:
        """Implementación del algoritmo Ford-Fulkerson"""
        def find_path(residual_graph: Dict[str, Dict[str, float]], source: str, sink: str) -> Tuple[bool, Dict[str, Optional[str]]]:
            visited: Set[str] = set()
            queue: deque[str] = deque([source])
            parent: Dict[str, Optional[str]] = {vertex: None for vertex in self.graph.vertices}
            visited.add(source)

            while queue and sink not in visited:
                u = queue.popleft()
                for v in residual_graph[u]:
                    if v not in visited and residual_graph[u][v] > 0:
                        queue.append(v)
                        visited.add(v)
                        parent[v] = u

            return sink in visited, parent

        # Inicializar grafo residual
        residual_graph: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for u in self.graph.vertices:
            for v, weight, _ in self.graph.graph[u]:
                residual_graph[u][v] = float(weight)

        max_flow = 0
        flow_paths: List[FlowPath] = []

        while True:
            path_exists, parent = find_path(residual_graph, source, sink)
            if not path_exists:
                break

            path_flow = float('infinity')
            s = sink
            path: List[str] = []
            while s != source:
                path.append(s)
                p = parent[s]
                if p is not None:
                    path_flow = min(path_flow, residual_graph[p][s])
                s = p if p is not None else source
            path.append(source)
            path.reverse()

            s = sink
            while s != source:
                p = parent[s]
                if p is not None:
                    residual_graph[p][s] -= path_flow
                    residual_graph[s][p] += path_flow
                s = p if p is not None else source

            max_flow += path_flow
            flow_paths.append({"path": path, "flow": path_flow})

        return max_flow, flow_paths

    def _edmonds_karp(self, source: str, sink: str) -> Tuple[float, List[FlowPath]]:
        """Implementación del algoritmo de Edmonds-Karp"""
        def bfs(residual_graph: Dict[str, Dict[str, float]], source: str, sink: str) -> Tuple[bool, Dict[str, Optional[str]]]:
            visited: Set[str] = set()
            queue: deque[str] = deque([source])
            parent: Dict[str, Optional[str]] = {vertex: None for vertex in self.graph.vertices}
            visited.add(source)

            while queue and sink not in visited:
                u = queue.popleft()
                for v in residual_graph[u]:
                    if v not in visited and residual_graph[u][v] > 0:
                        queue.append(v)
                        visited.add(v)
                        parent[v] = u

            return sink in visited, parent

        # Inicializar grafo residual
        residual_graph: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for u in self.graph.vertices:
            for v, weight, _ in self.graph.graph[u]:
                residual_graph[u][v] = float(weight)

        max_flow = 0
        flow_paths: List[FlowPath] = []

        while True:
            path_exists, parent = bfs(residual_graph, source, sink)
            if not path_exists:
                break

            path_flow = float('infinity')
            s = sink
            path: List[str] = []
            while s != source:
                path.append(s)
                p = parent[s]
                if p is not None:
                    path_flow = min(path_flow, residual_graph[p][s])
                s = p if p is not None else source
            path.append(source)
            path.reverse()

            s = sink
            while s != source:
                p = parent[s]
                if p is not None:
                    residual_graph[p][s] -= path_flow
                    residual_graph[s][p] += path_flow
                s = p if p is not None else source

            max_flow += path_flow
            flow_paths.append({"path": path, "flow": path_flow})

        return max_flow, flow_paths

    def _push_relabel(self, source: str, sink: str) -> Tuple[float, List[FlowPath]]:
        """Implementación del algoritmo Push-Relabel"""
        def initialize_preflow() -> Tuple[Dict[str, int], Dict[str, float], Dict[str, Dict[str, float]]]:
            height: Dict[str, int] = {vertex: 0 for vertex in self.graph.vertices}
            height[source] = len(self.graph.vertices)
            
            excess: Dict[str, float] = {vertex: 0.0 for vertex in self.graph.vertices}
            
            flow: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
            
            # Inicializar el flujo desde la fuente
            for v, weight, _ in self.graph.graph[source]:
                flow[source][v] = float(weight)
                flow[v][source] = -float(weight)
                excess[v] = float(weight)
                excess[source] -= float(weight)
            
            return height, excess, flow

        def push(u: str, v: str, height: Dict[str, int], excess: Dict[str, float], 
                flow: Dict[str, Dict[str, float]], capacity: Dict[str, Dict[str, float]]) -> bool:
            if height[u] <= height[v]:
                return False
            
            residual = capacity[u][v] - flow[u][v]
            if residual <= 0 or excess[u] <= 0:
                return False
                
            delta = min(excess[u], residual)
            flow[u][v] += delta
            flow[v][u] -= delta
            excess[u] -= delta
            excess[v] += delta
            return True

        def relabel(u: str, height: Dict[str, int], flow: Dict[str, Dict[str, float]], 
                   capacity: Dict[str, Dict[str, float]]) -> bool:
            if excess[u] <= 0:
                return False
                
            min_height = float('infinity')
            for v, weight, _ in self.graph.graph[u]:
                if capacity[u][v] - flow[u][v] > 0:
                    min_height = min(min_height, height[v])
            
            if min_height == float('infinity'):
                return False
                
            height[u] = int(min_height + 1)  # Convertir a int
            return True

        # Inicializar capacidades
        capacity: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for u in self.graph.vertices:
            for v, weight, _ in self.graph.graph[u]:
                capacity[u][v] = float(weight)

        # Inicializar preflow
        height, excess, flow = initialize_preflow()

        # Proceso principal
        vertices = list(self.graph.vertices - {source, sink})
        i = 0
        while i < len(vertices):
            u = vertices[i]
            old_excess = excess[u]
            
            for v, _, _ in self.graph.graph[u]:
                if excess[u] > 0:
                    push(u, v, height, excess, flow, capacity)
            
            if excess[u] == old_excess and excess[u] > 0:
                relabel(u, height, flow, capacity)
                i = 0
            else:
                i += 1

        # Reconstruir las rutas de flujo
        max_flow = sum(flow[source][v] for v, _, _ in self.graph.graph[source])
        
        # Encontrar las rutas de flujo
        flow_paths: List[FlowPath] = []
        visited: Set[str] = set()
        
        def find_path(u: str, current_path: List[str], current_flow: float) -> None:
            if u == sink:
                if current_flow > 0:
                    flow_paths.append({"path": current_path.copy(), "flow": current_flow})
                return
            
            visited.add(u)
            for v, _, _ in self.graph.graph[u]:
                if v not in visited and flow[u][v] > 0:
                    current_path.append(v)
                    find_path(v, current_path, min(current_flow, flow[u][v]))
                    current_path.pop()
            visited.remove(u)
        
        find_path(source, [source], float('infinity'))
        
        return max_flow, flow_paths 