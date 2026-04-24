import csv
import math

from tqdm import tqdm


def calculate_distance(lat1, lon1, lat2, lon2):
    radius = 6371  # Earth's radius in kilometers

    # Convert latitude and longitude to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Calculate the differences between coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Calculate the Great Circle Distance
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = radius * c

    # Convert distance to meters
    distance *= 1000

    return distance


def analyze_dataset(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header

        num_trips = 0
        num_unique_points = set()
        total_points = 0
        max_points = -1
        min_points = float('inf')

        max_distance = -1
        min_distance = float('inf')
        total_distance=0
        with tqdm(reader,'===',total=1710671) as tq:
            for row in tq:
                polyline = eval(row[8])
                distance = 0
                if len(polyline)<=20:
                    continue
                num_trips += 1
                #print(polyline)
                num_points = len(polyline)
                total_points += num_points
                max_points = max(max_points, num_points)
                min_points = min(min_points, num_points)
                num_unique_points.add(str(polyline[0]))
                for i in range(1, num_points):
                    num_unique_points.add(str(polyline[i]))
                    lat1, lon1 = polyline[i-1]
                    lat2, lon2 = polyline[i]
                    d = calculate_distance(lat1, lon1, lat2, lon2)
                    distance += d
                max_distance = max(max_distance, distance)
                min_distance = min(min_distance, distance)
                total_distance+=distance
        avg_points = total_points / num_trips
        avg_distance = total_distance / num_trips

        return num_trips, len(num_unique_points), avg_points, max_points, min_points, avg_distance, max_distance, min_distance


# Replace 'your_file_path.csv' with the actual file path
result = analyze_dataset('porto.csv')

# Print the results
print("1. Number of trips:", result[0])
print("2. Number of unique points:", result[1])
print("3. Average points per trip:", result[2])
print("4. Maximum points per trip:", result[3])
print("   Minimum points per trip:", result[4])
print("5. Average distance per trip:", result[5], "m")
print("6. Maximum distance per trip:", result[6], "m")
print("   Minimum distance per trip:", result[7], "m")