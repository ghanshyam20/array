import heapq

def dijkstra(graph, start):
    # Priority queue to store (distance, vertex)
    priority_queue = [(0, start)]
    # Dictionary to store the shortest distance to each vertex
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        # Skip processing if we find a longer distance in the queue
        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight

            # If a shorter path is found
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

# Example usage
if __name__ == "__main__":
    # Graph represented as an adjacency list
    graph = {
        'A': {'B': 1, 'C': 4},
        'B': {'A': 1, 'C': 2, 'D': 6},
        'C': {'A': 4, 'B': 2, 'D': 3},
        'D': {'B': 6, 'C': 3}
    }
    start_vertex = 'A'
    shortest_distances = dijkstra(graph, start_vertex)
    print(f"Shortest distances from vertex {start_vertex}: {shortest_distances}")


    