from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_from_directory, Response
from flask_cors import CORS
import pandas as pd
import folium
import json
from route_algorithms import RouteCalculator, Graph
from pathlib import Path
import os
from werkzeug.utils import secure_filename
from typing import Dict, List, Tuple, Optional, Any, Union
from flask.typing import ResponseReturnValue
import csv
#Presentado por: Carlos, Mario y Valeria
app = Flask(__name__)
app.secret_key = '1111' 
CORS(app)

# Coordenadas de los municipios, exclusivo para el mapa del programa
COORDS_DATA = {
    "Achi": {"lat": 8.60283045, "lon": -74.4586880450204},
    "Altos del Rosario": {"lat": 8.79139, "lon": -74.1636},
    "Arenal": {"lat": 8.4593893, "lon": -73.942994},
    "Arjona": {"lat": 10.25444, "lon": -75.34389},
    "Arroyohondo": {"lat": 10.24402435, "lon": -75.0283270068316},
    "Barranco de Loba": {"lat": 8.94556, "lon": -74.1058},
    "Calamar": {"lat": 10.2526122, "lon": -74.9146691},
    "Cantagallo": {"lat": 7.3780007, "lon": -73.9150265},
    "Cartagena de Indias": {"lat": 10.39972, "lon": -75.51444},
    "Cicuco": {"lat": 9.2778539, "lon": -74.643809},
    "Clemencia": {"lat": 10.56180475, "lon": -75.3295539371322},
    "Cordoba": {"lat": 9.5244165, "lon": -74.8780482},
    "El Carmen de Bolivar": {"lat": 9.7178841, "lon": -75.1241376},
    "El Guamo": {"lat": 10.0312607, "lon": -74.9754413},
    "El Penon": {"lat": 8.9895346, "lon": -73.949864},
    "Hatillo de Loba": {"lat": 8.9568235, "lon": -74.0770262},
    "Magangue": {"lat": 9.2412097, "lon": -74.7567413},
    "Mahates": {"lat": 10.234262, "lon": -75.186894},
    "Margarita": {"lat": 9.05747505, "lon": -74.2209201246511},
    "Maria La Baja": {"lat": 9.9863081, "lon": -75.3963004},
    "Montecristo": {"lat": 7.93057835, "lon": -74.4074396536722},
    "Morales": {"lat": 8.2763913, "lon": -73.8680272},
    "Norosi": {"lat": 8.52611, "lon": -74.0378},
    "Pinillos": {"lat": 8.915, "lon": -74.4619},
    "Regidor": {"lat": 8.66639, "lon": -73.8222},
    "Rio Viejo": {"lat": 8.583, "lon": -73.85},
    "San Cristobal": {"lat": 10.3925, "lon": -75.0631},
    "San Estanislao": {"lat": 10.3978, "lon": -75.1514},
    "San Fernando": {"lat": 9.21194, "lon": -74.3231},
    "San Jacinto": {"lat": 9.83111, "lon": -75.1219},
    "San Jacinto del Cauca": {"lat": 8.24972, "lon": -74.72},
    "San Juan Nepomuceno": {"lat": 9.95222, "lon": -75.0811},
    "San Martin de Loba": {"lat": 8.93889, "lon": -74.0392},
    "San Pablo": {"lat": 7.47639, "lon": -73.9231},
    "Santa Catalina": {"lat": 10.6039, "lon": -75.2878},
    "Santa Cruz de Mompox": {"lat": 9.233, "lon": -74.417},
    "Santa Rosa": {"lat": 10.4456, "lon": -75.3686},
    "Santa Rosa del Sur": {"lat": 7.96333, "lon": -74.0533},
    "Simiti": {"lat": 7.95639, "lon": -73.9461},
    "Soplaviento": {"lat": 10.3908, "lon": -75.1367},
    "Talaigua Nuevo": {"lat": 9.30278, "lon": -74.5678},
    "Tiquisio": {"lat": 8.55778, "lon": -74.2639},
    "Turbaco": {"lat": 10.3319, "lon": -75.4142},
    "Turbana": {"lat": 10.283, "lon": -75.45},
    "Villanueva": {"lat": 10.4442, "lon": -75.2747},
    "Zambrano": {"lat": 9.74472, "lon": -74.8172}
}

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
CALCULATOR: Optional[RouteCalculator] = None  # Se inicializa la calculadora como NONE

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_graph_data(file_path: str) -> bool:
    """Carga los datos del grafo desde un archivo CSV."""
    global CALCULATOR  
    try:
        graph = Graph()
        
        # Diccionario para almacenar todas las conexiones para verificación
        connections = {}
        
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=';')
            headers = next(reader)
            
            required_headers = ['Origen', 'Destino', 'Distancia', 'Duracion']
            if not all(header in headers for header in required_headers):
                return False
            
            origin_idx = headers.index('Origen')
            dest_idx = headers.index('Destino')
            dist_idx = headers.index('Distancia')
            time_idx = headers.index('Duracion')
            
            # Primera pasada: recolectar todas las conexiones
            for row in reader:
                origin = row[origin_idx].strip()
                destination = row[dest_idx].strip()
                try:
                    # Convertir valores con coma decimal a punto decimal, en caso de ser necesario
                    distance = float(row[dist_idx].strip().replace(',', '.'))
                    duration = float(row[time_idx].strip().replace(',', '.'))
                    
                 
                    if origin not in connections:
                        connections[origin] = set()
                    if destination not in connections:
                        connections[destination] = set()
                    
                    connections[origin].add(destination)
                    connections[destination].add(origin)
                    
                    # Agregar la conexión al grafo en ambas direcciones
                    graph.add_edge(origin, destination, distance, duration)
                    graph.add_edge(destination, origin, distance, duration)
                    
                except ValueError as e:
                    continue
        
        # Verificar conectividad
        def find_path(start, end, visited=None):
            if visited is None:
                visited = set()
            if start == end:
                return True
            visited.add(start)
            for next_node in connections.get(start, []):
                if next_node not in visited and find_path(next_node, end, visited):
                    return True
            return False
        
        # Verificar que haya un camino entre Cartagena y todos los demás municipios

        
        CALCULATOR = RouteCalculator(graph)
        return True
        
    except Exception as e:
        print(f"Error al cargar el grafo: {str(e)}")
        return False

def create_base_map():
    """Crear mapa base centrado en Bolívar"""
    return folium.Map(
        location=[9.0, -74.8],  
        zoom_start=8,
        tiles='OpenStreetMap'
    )

@app.route('/')
def index():
    """Ruta principal que maneja la redirección entre la página de carga y la interfaz principal."""
    if 'graph_loaded' not in session or not session['graph_loaded']:
        return redirect(url_for('upload_page'))
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    """Página de carga de archivos."""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file() -> ResponseReturnValue:
    """Maneja la carga de archivos CSV."""

    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No se encontró el archivo'
        })
    
    file = request.files['file']
    

    if not file or not file.filename:
        return jsonify({
            'status': 'error',
            'message': 'No se seleccionó ningún archivo'
        })
    

    if not allowed_file(str(file.filename)):
        return jsonify({
            'status': 'error',
            'message': 'Tipo de archivo no permitido. Solo se aceptan archivos CSV.'
        })
    
    try:
  
        filename = secure_filename(str(file.filename))
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        if load_graph_data(file_path):
            session['graph_loaded'] = True
            session['graph_file'] = file_path
            
            return jsonify({
                'status': 'success',
                'message': 'Archivo cargado exitosamente'
            })
        else:
           
            if os.path.exists(file_path):
                os.remove(file_path)
                
            return jsonify({
                'status': 'error',
                'message': 'Error al procesar el archivo. Verifique que el formato sea correcto: Origen;Destino;Distancia;Duracion'
            })
            
    except Exception as e:
       
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
            
        return jsonify({
            'status': 'error',
            'message': f'Error al procesar el archivo: {str(e)}'
        })

@app.route('/get_municipalities')
def get_municipalities():
    """Obtener lista de municipios"""
    if 'graph_loaded' not in session:
        return jsonify([])
    return jsonify(list(COORDS_DATA.keys()))

@app.route('/calculate_route', methods=['POST'])
def calculate_route() -> ResponseReturnValue:
    """Calcula la ruta entre dos puntos usando el algoritmo especificado."""
    print("\n=== Iniciando cálculo de ruta ===")  # Debug log
    
    if not CALCULATOR:
        print("Error: No hay calculadora inicializada")  # Debug log
        return jsonify({
            'status': 'error',
            'message': 'No se han cargado los datos del grafo'
        }), 400

    try:
  
        content_type = request.headers.get('Content-Type', '')
        print(f"Content-Type recibido: {content_type}")  # Debug log
        
        if not content_type.startswith('application/json'):
            print(f"Error: Content-Type incorrecto: {content_type}")  # Debug log
            return jsonify({
                'status': 'error',
                'message': 'El Content-Type debe ser application/json'
            }), 400

        
        try:
            raw_data = request.get_data(as_text=True)
            print(f"Datos raw recibidos: {raw_data}")  # Debug log
            
            data = request.get_json(force=True)
            print(f"Datos JSON parseados: {data}")  # Debug log
        except Exception as e:
            print(f"Error al parsear JSON: {str(e)}")  # Debug log
            return jsonify({
                'status': 'error',
                'message': f'Error al parsear JSON: {str(e)}'
            }), 400

        if not isinstance(data, dict):
            print(f"Error: datos no son un diccionario: {type(data)}")  # Debug log
            return jsonify({
                'status': 'error',
                'message': 'Formato de datos inválido'
            }), 400

        origin = str(data.get('origin', '')).strip()
        destination = str(data.get('destination', '')).strip()
        algorithm = str(data.get('algorithm', '')).strip()
        criterion = str(data.get('criterion', 'distancia')).strip()

        print(f"Datos procesados:")  # Debug log
        print(f"- Origen: '{origin}'")
        print(f"- Destino: '{destination}'")
        print(f"- Algoritmo: '{algorithm}'")
        print(f"- Criterio: '{criterion}'")

       
        if not origin or not destination or not algorithm:
            missing = []
            if not origin: missing.append('origen')
            if not destination: missing.append('destino')
            if not algorithm: missing.append('algoritmo')
            print(f"Error: Faltan datos requeridos: {missing}")  # Debug log
            return jsonify({
                'status': 'error',
                'message': f'Faltan datos requeridos: {", ".join(missing)}'
            }), 400

       
        if origin not in COORDS_DATA:
            print(f"Error: Municipio de origen no encontrado: '{origin}'")  # Debug log
            return jsonify({
                'status': 'error',
                'message': f'Municipio de origen "{origin}" no encontrado'
            }), 404
            
        if destination not in COORDS_DATA:
            print(f"Error: Municipio de destino no encontrado: '{destination}'")  # Debug log
            return jsonify({
                'status': 'error',
                'message': f'Municipio de destino "{destination}" no encontrado'
            }), 404

        try:
       
            valid_algorithms = ['dijkstra', 'bellman_ford', 'floyd_warshall', 'a_star', 'johnson',
                              'ford_fulkerson', 'edmonds_karp', 'push_relabel']
            if algorithm not in valid_algorithms:
                print(f"Error: Algoritmo no válido: '{algorithm}'")  # Debug log
                return jsonify({
                    'status': 'error',
                    'message': f'Algoritmo "{algorithm}" no válido'
                }), 400

   
            if algorithm in ['edmonds_karp', 'push_relabel', 'ford_fulkerson']:
                print(f"\nCalculando flujo máximo con {algorithm}...")  # Debug log
                try:
                    result = None
                    if algorithm == 'ford_fulkerson':
                        result = CALCULATOR._ford_fulkerson(origin, destination)
                    elif algorithm == 'edmonds_karp':
                        result = CALCULATOR._edmonds_karp(origin, destination)
                    elif algorithm == 'push_relabel':
                        result = CALCULATOR._push_relabel(origin, destination)
                    
                    if result is None:
                        print("Error: No se pudo calcular el flujo máximo")  # Debug log
                        return jsonify({
                            'status': 'error',
                            'message': 'No se pudo calcular el flujo máximo'
                        }), 500
                        
                    max_flow, flow_paths = result
                    print(f"Flujo máximo calculado: {max_flow}")  # Debug log
                    print(f"Número de rutas: {len(flow_paths)}")  # Debug log
                    
                    # Preparar datos para el frontend
                    route_data = {
                        'type': 'flow',
                        'paths': flow_paths,
                        'coords': COORDS_DATA
                    }
                    
                    response_data = {
                        'status': 'success',
                        'route_data': route_data,
                        'details': {
                            'max_flow': float(max_flow),  # Asegurar que sea JSON serializable
                            'num_paths': len(flow_paths),
                            'flow_paths': flow_paths
                        }
                    }
                    
                    print("\nEnviando respuesta:")  # Debug log
                    print(response_data)
                    
                    return jsonify(response_data)
                    
                except Exception as e:
                    print(f"Error al calcular el flujo máximo con {algorithm}: {str(e)}")  # Debug log
                    return jsonify({
                        'status': 'error',
                        'message': f'Error al calcular el flujo máximo con {algorithm}: {str(e)}'
                    }), 500
            else:
                print(f"\nCalculando ruta más corta con {algorithm}...")  # Debug log
                try:
                    result = CALCULATOR.calculate_shortest_path(origin, destination, algorithm, criterion)
                    if result is None:
                        print("Error: No se pudo calcular la ruta")  
                        return jsonify({
                            'status': 'error',
                            'message': 'No se pudo calcular la ruta'
                        }), 500
                        
                    path, distance, time = result
                    print(f"Ruta calculada:")  # Debug log
                    print(f"- Camino: {path}")
                    print(f"- Distancia: {distance}")
                    print(f"- Tiempo: {time}")
                    
                    # Preparar datos para el frontend
                    route_data = {
                        'type': 'path',
                        'path': path,
                        'coords': COORDS_DATA
                    }
                    
                    response_data = {
                        'status': 'success',
                        'route_data': route_data,
                        'details': {
                            'path': ' → '.join(path),
                            'distance': float(distance),  # Asegurar que sea JSON serializable
                            'time': float(time)  # Asegurar que sea JSON serializable
                        }
                    }
                    
                    print("\nEnviando respuesta:")  # Debug log
                    print(response_data)
                    
                    return jsonify(response_data)
                    
                except Exception as e:
                    print(f"Error al calcular la ruta con {algorithm}: {str(e)}")  # Debug log
                    return jsonify({
                        'status': 'error',
                        'message': f'Error al calcular la ruta con {algorithm}: {str(e)}'
                    }), 500

        except Exception as e:
            print(f"Error al calcular la ruta: {str(e)}")  # Debug log
            import traceback
            traceback.print_exc()  # Imprimir stack trace completo
            return jsonify({
                'status': 'error',
                'message': f'Error al calcular la ruta: {str(e)}'
            }), 500

    except Exception as e:
        print(f"Error al procesar la solicitud: {str(e)}")  # Debug log
        import traceback
        traceback.print_exc()  # Imprimir stack trace completo
        return jsonify({
            'status': 'error',
            'message': f'Error al procesar la solicitud: {str(e)}'
        }), 400
#Cierre de sesion o reinicio del programa en este caso
@app.route('/logout', methods=['POST'])
def logout() -> ResponseReturnValue:
    """Cierra la sesión actual y limpia los datos cargados."""
    global CALCULATOR
    
    
    CALCULATOR = None
    
 
    session.clear()

    if 'graph_file' in session:
        try:
            file_path = session['graph_file']
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
    
    return jsonify({
        'status': 'success',
        'message': 'Sesión cerrada exitosamente'
    })

if __name__ == '__main__':
    app.run(debug=True) 