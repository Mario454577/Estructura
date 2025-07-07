from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_from_directory, Response
from flask_cors import CORS
import pandas as pd
import folium
import json
from route_algorithms import RouteCalculator
from pathlib import Path
import os
from werkzeug.utils import secure_filename
from typing import Dict, List, Tuple, Optional, Any, Union
from flask.typing import ResponseReturnValue
import csv

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_aqui'  # Necesario para flash messages y session
CORS(app)

# Cargar datos de coordenadas
COORDS_DATA = {
    "Achí": {"lat": 8.60283045, "lon": -74.4586880450204},
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
    "Córdoba": {"lat": 9.5244165, "lon": -74.8780482},
    "El Carmen de Bolívar": {"lat": 9.7178841, "lon": -75.1241376},
    "El Guamo": {"lat": 10.0312607, "lon": -74.9754413},
    "El Peñón": {"lat": 8.9895346, "lon": -73.949864},
    "Hatillo de Loba": {"lat": 8.9568235, "lon": -74.0770262},
    "Magangué": {"lat": 9.2412097, "lon": -74.7567413},
    "Mahates": {"lat": 10.234262, "lon": -75.186894},
    "Margarita": {"lat": 9.05747505, "lon": -74.2209201246511},
    "María La Baja": {"lat": 9.9863081, "lon": -75.3963004},
    "Montecristo": {"lat": 7.93057835, "lon": -74.4074396536722},
    "Morales": {"lat": 8.2763913, "lon": -73.8680272},
    "Norosí": {"lat": 8.52611, "lon": -74.0378},
    "Pinillos": {"lat": 8.915, "lon": -74.4619},
    "Regidor": {"lat": 8.66639, "lon": -73.8222},
    "Río Viejo": {"lat": 8.583, "lon": -73.85},
    "San Cristóbal": {"lat": 10.3925, "lon": -75.0631},
    "San Estanislao": {"lat": 10.3978, "lon": -75.1514},
    "San Fernando": {"lat": 9.21194, "lon": -74.3231},
    "San Jacinto": {"lat": 9.83111, "lon": -75.1219},
    "San Jacinto del Cauca": {"lat": 8.24972, "lon": -74.72},
    "San Juan Nepomuceno": {"lat": 9.95222, "lon": -75.0811},
    "San Martín de Loba": {"lat": 8.93889, "lon": -74.0392},
    "San Pablo": {"lat": 7.47639, "lon": -73.9231},
    "Santa Catalina": {"lat": 10.6039, "lon": -75.2878},
    "Santa Cruz de Mompox": {"lat": 9.233, "lon": -74.417},
    "Santa Rosa": {"lat": 10.4456, "lon": -75.3686},
    "Santa Rosa del Sur": {"lat": 7.96333, "lon": -74.0533},
    "Simití": {"lat": 7.95639, "lon": -73.9461},
    "Soplaviento": {"lat": 10.3908, "lon": -75.1367},
    "Talaigua Nuevo": {"lat": 9.30278, "lon": -74.5678},
    "Tiquisio": {"lat": 8.55778, "lon": -74.2639},
    "Turbaco": {"lat": 10.3319, "lon": -75.4142},
    "Turbaná": {"lat": 10.283, "lon": -75.45},
    "Villanueva": {"lat": 10.4442, "lon": -75.2747},
    "Zambrano": {"lat": 9.74472, "lon": -74.8172}
}

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
CALCULATOR: Optional[RouteCalculator] = None  # Inicializar como None

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_graph_data(file_path: str) -> bool:
    """Carga los datos del grafo desde un archivo CSV."""
    global CALCULATOR  # Declarar que vamos a modificar la variable global
    try:
        # Leer el archivo CSV y construir el grafo
        graph_data: Dict[str, Dict[str, Dict[str, float]]] = {}
        
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=';')
            headers = next(reader)
            
            # Verificar encabezados
            required_headers = ['Origen', 'Destino', 'Distancia', 'Duracion']
            if not all(header in headers for header in required_headers):
                return False
            
            # Obtener índices de las columnas
            origin_idx = headers.index('Origen')
            dest_idx = headers.index('Destino')
            dist_idx = headers.index('Distancia')
            time_idx = headers.index('Duracion')
            
            # Leer datos
            for row in reader:
                origin = row[origin_idx].strip()
                destination = row[dest_idx].strip()
                try:
                    distance = float(row[dist_idx].strip())
                    duration = float(row[time_idx].strip())
                except ValueError:
                    continue
                
                # Agregar al grafo (ambas direcciones)
                if origin not in graph_data:
                    graph_data[origin] = {}
                if destination not in graph_data:
                    graph_data[destination] = {}
                
                # Agregar arista origen -> destino
                graph_data[origin][destination] = {
                    'distancia': distance,
                    'tiempo': duration
                }
                
                # Agregar arista destino -> origen (grafo no dirigido)
                graph_data[destination][origin] = {
                    'distancia': distance,
                    'tiempo': duration
                }
        
        # Crear nueva instancia del calculador
        CALCULATOR = RouteCalculator(graph_data)
        return True
        
    except Exception as e:
        print(f"Error al cargar el grafo: {str(e)}")
        return False

def create_base_map():
    """Crear mapa base centrado en Bolívar"""
    return folium.Map(
        location=[9.0, -74.8],  # Centro aproximado de Bolívar
        zoom_start=8,
        tiles='OpenStreetMap'
    )

@app.route('/')
def index():
    """Ruta principal que maneja la redirección entre la página de carga y la interfaz principal."""
    if 'graph_loaded' not in session or not session['graph_loaded']:
        return render_template('upload.html')
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file() -> ResponseReturnValue:
    """Maneja la carga de archivos CSV."""
    # Verificar si hay un archivo en la solicitud
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No se encontró el archivo'
        })
    
    file = request.files['file']
    
    # Verificar si se seleccionó un archivo
    if not file or not file.filename:
        return jsonify({
            'status': 'error',
            'message': 'No se seleccionó ningún archivo'
        })
    
    # Verificar si es un archivo CSV
    if not allowed_file(str(file.filename)):
        return jsonify({
            'status': 'error',
            'message': 'Tipo de archivo no permitido. Solo se aceptan archivos CSV.'
        })
    
    try:
        # Guardar el archivo
        filename = secure_filename(str(file.filename))
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Intentar cargar el grafo
        if load_graph_data(file_path):
            session['graph_loaded'] = True
            session['graph_file'] = file_path
            
            return jsonify({
                'status': 'success',
                'message': 'Archivo cargado exitosamente'
            })
        else:
            # Si falla la carga del grafo, eliminar el archivo
            if os.path.exists(file_path):
                os.remove(file_path)
                
            return jsonify({
                'status': 'error',
                'message': 'Error al procesar el archivo. Verifique que el formato sea correcto: Origen;Destino;Distancia;Duracion'
            })
            
    except Exception as e:
        # En caso de cualquier error, asegurar que se elimine el archivo si existe
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
    """Calcula y visualiza la ruta entre dos municipios."""
    if 'graph_loaded' not in session:
        return jsonify({
            'status': 'error',
            'message': 'Primero debe cargar un archivo CSV'
        })
    
    if not CALCULATOR:
        return jsonify({
            'status': 'error',
            'message': 'El calculador de rutas no está inicializado'
        })
    
    data = request.get_json()
    if not data:
        return jsonify({
            'status': 'error',
            'message': 'Datos inválidos'
        })
        
    origin = str(data.get('origin', ''))
    destination = str(data.get('destination', ''))
    algorithm = str(data.get('algorithm', ''))
    criterion = str(data.get('criterion', 'distancia'))
    
    if not all([origin, destination, algorithm]):
        return jsonify({
            'status': 'error',
            'message': 'Faltan datos requeridos'
        })
    
    try:
        if algorithm == 'edmonds_karp':
            # Construir grafo de flujo y calcular flujo máximo
            CALCULATOR.build_flow_graph(origin, destination)
            flow_paths, max_flow = CALCULATOR.edmonds_karp(origin, destination)
            
            if not flow_paths:
                return jsonify({
                    'status': 'error',
                    'message': 'No se encontró una ruta válida'
                })
            
            # Crear mapa con las rutas de flujo
            m = create_base_map()
            
            # Verificar y obtener coordenadas
            try:
                origin_coords = COORDS_DATA.get(origin, {})
                dest_coords = COORDS_DATA.get(destination, {})
                
                # Convertir coordenadas a float y validar
                origin_lat = float(origin_coords.get('lat', 0))
                origin_lon = float(origin_coords.get('lon', 0))
                dest_lat = float(dest_coords.get('lat', 0))
                dest_lon = float(dest_coords.get('lon', 0))
                
                if origin_lat != 0 and origin_lon != 0 and dest_lat != 0 and dest_lon != 0:
                    # Agregar marcadores para origen y destino
                    folium.Marker(
                        location=[origin_lat, origin_lon],
                        popup=f'Origen: {origin}',
                        icon=folium.Icon(color='green')
                    ).add_to(m)
                    
                    folium.Marker(
                        location=[dest_lat, dest_lon],
                        popup=f'Destino: {destination}',
                        icon=folium.Icon(color='red')
                    ).add_to(m)
                    
                    # Dibujar cada ruta de flujo con un color diferente
                    colors = ['blue', 'purple', 'orange', 'darkred', 'darkgreen', 'cadetblue']
                    flow_path_details = []
                    
                    for i, path_data in enumerate(flow_paths):
                        if not isinstance(path_data, tuple) or len(path_data) != 2:
                            continue
                            
                        path, flow = path_data
                        if not isinstance(path, list) or not isinstance(flow, (int, float)):
                            continue
                            
                        # Verificar y recolectar coordenadas para cada ciudad en la ruta
                        points = []
                        valid_path = True
                        
                        for city in path:
                            city_str = str(city)
                            city_coords = COORDS_DATA.get(city_str, {})
                            
                            lat = float(city_coords.get('lat', 0))
                            lon = float(city_coords.get('lon', 0))
                            
                            if lat != 0 and lon != 0:
                                points.append([lat, lon])
                            else:
                                valid_path = False
                                break
                        
                        if valid_path and points:
                            color = colors[i % len(colors)]
                            weight = 2.0 + float(flow)/50.0
                            
                            folium.PolyLine(
                                locations=points,
                                weight=weight,
                                color=color,
                                popup=f'Flujo: {flow}'
                            ).add_to(m)
                            
                            flow_path_details.append({
                                'path': [str(city) for city in path],
                                'flow': float(flow)
                            })
                    
                    return jsonify({
                        'status': 'success',
                        'map': m._repr_html_(),
                        'details': {
                            'max_flow': float(max_flow),
                            'num_paths': len(flow_path_details),
                            'flow_paths': flow_path_details
                        }
                    })
                
                return jsonify({
                    'status': 'error',
                    'message': 'Coordenadas no válidas para origen o destino'
                })
            except (ValueError, TypeError) as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Error al procesar coordenadas: {str(e)}'
                })
        else:
            result = CALCULATOR.calculate_route(origin, destination, algorithm, criterion)
            if not result or len(result) != 3:
                return jsonify({
                    'status': 'error',
                    'message': 'Error al calcular la ruta'
                })
                
            path, distance, time = result
            
            if not path:
                return jsonify({
                    'status': 'error',
                    'message': 'No se encontró una ruta válida'
                })
            
            # Crear mapa con la ruta
            m = create_base_map()
            
            try:
                # Verificar y obtener coordenadas
                points = []
                valid_path = True
                
                for city in [origin, destination] + path:
                    city_str = str(city)
                    city_coords = COORDS_DATA.get(city_str, {})
                    
                    lat = float(city_coords.get('lat', 0))
                    lon = float(city_coords.get('lon', 0))
                    
                    if lat != 0 and lon != 0:
                        if city == origin:
                            folium.Marker(
                                location=[lat, lon],
                                popup=f'Origen: {city}',
                                icon=folium.Icon(color='green')
                            ).add_to(m)
                        elif city == destination:
                            folium.Marker(
                                location=[lat, lon],
                                popup=f'Destino: {city}',
                                icon=folium.Icon(color='red')
                            ).add_to(m)
                        
                        points.append([lat, lon])
                    else:
                        valid_path = False
                        break
                
                if valid_path and points:
                    # Dibujar la ruta
                    folium.PolyLine(
                        locations=points,
                        weight=3,
                        color='blue',
                        popup=f'Distancia: {float(distance):.2f} km, Tiempo: {int(time)} min'
                    ).add_to(m)
                    
                    return jsonify({
                        'status': 'success',
                        'map': m._repr_html_(),
                        'details': {
                            'distance': float(distance),
                            'time': int(time),
                            'path': ' → '.join(str(city) for city in path)
                        }
                    })
                
                return jsonify({
                    'status': 'error',
                    'message': 'Coordenadas no válidas para algunas ciudades en la ruta'
                })
            except (ValueError, TypeError) as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Error al procesar coordenadas: {str(e)}'
                })
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/logout', methods=['POST'])
def logout() -> ResponseReturnValue:
    """Cierra la sesión actual y limpia los datos cargados."""
    global CALCULATOR
    
    # Limpiar variables globales
    CALCULATOR = None
    
    # Limpiar la sesión
    session.clear()
    
    # Eliminar archivo CSV si existe
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