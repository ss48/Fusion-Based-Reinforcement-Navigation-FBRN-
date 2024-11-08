import xml.etree.ElementTree as ET
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Function to parse KML file and extract coordinates
def extract_coordinates_from_kml(kml_file):
    tree = ET.parse(kml_file)
    root = tree.getroot()

    # Find all <coordinates> tags in the KML file
    namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
    coordinates = []
    
    for placemark in root.findall(".//kml:Placemark/kml:Point/kml:coordinates", namespaces):
        coords_text = placemark.text.strip()
        lon, lat, _ = map(float, coords_text.split(","))
        coordinates.append((lat, lon))
    
    return coordinates

# Haversine formula to calculate distance between two lat/lon points
def haversine(coord1, coord2):
    R = 6371  # Radius of the Earth in kilometers
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c  # in kilometers
    return distance

# Create a distance matrix using the Haversine formula
def create_distance_matrix(coords):
    num_points = len(coords)
    distance_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(i + 1, num_points):
            distance = haversine(coords[i], coords[j])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    return distance_matrix

# Generate a TSP file from the distance matrix
def generate_tsp_file(coords, distance_matrix, tsp_filename="problem.tsp"):
    num_points = len(coords)

    with open(tsp_filename, 'w') as f:
        f.write(f"NAME: KML_TSP\nTYPE: TSP\nDIMENSION: {num_points}\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
        
        for row in distance_matrix:
            f.write(" ".join(map(str, map(int, row))) + "\n")
        
        f.write("EOF\n")

# Main function to convert KML to TSP
def kml_to_tsp(kml_file, tsp_file="problem.tsp"):
    coords = extract_coordinates_from_kml(kml_file)
    distance_matrix = create_distance_matrix(coords)
    generate_tsp_file(coords, distance_matrix, tsp_file)
    print(f"TSP file '{tsp_file}' generated successfully.")

# Example usage
kml_file = "Essex_Hospitals_NoFlyZones.kml"
tsp_file = "problem.tsp"
kml_to_tsp(kml_file, tsp_file)
