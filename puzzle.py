import numpy as np
import matplotlib.pyplot as plt
import math

def main():
    plot_grid()

grid_size = 19
final_grid = []

node_list = [{'node': 38, 'x': -1, 'y': -1},
             {'node': 6, 'x': -5, 'y': 0},
             {'node': 3, 'x': 5, 'y' : 0},
             {'node': 4, 'x': 2, 'y': 1},
             {'node': 8, 'x': -1, 'y': 2},
             {'node': 1, 'x': -7, 'y': 3},
             {'node': 6, 'x': 5, 'y': 4},
             {'node': 3, 'x': 3, 'y': 6},
             {'node': 7, 'x': 0, 'y': 8},
             {'node': 6, 'x': 1, 'y': -2},
             {'node': 11, 'x': 8, 'y': -3},
             {'node': 6, 'x': -5, 'y': -4},
             {'node': 19, 'x': -3, 'y': -6},
             {'node': 6, 'x': 0, 'y': - 8}]

for i in range(grid_size):

    diff = abs(9 - i)
    new = [0 for i in range(grid_size - diff)]

    for j in range(len(new)):
        if i % 2 == 0:
            new[j] = {'x': j + 1 - round(len(new)/2), 'y': i - 9}

        
        else:
            new[j] = {'x': j - (math.floor(len(new)/2)), 'y': i - 9}
    
    final_grid.append(new)
    

def plot_grid():

    x_values = [node['x'] for node in node_list]
    y_values = [node['y'] for node in node_list]
    node_values = [node['node'] for node in node_list]

# Plotting the hexagonal grid and nodes
    plt.figure(figsize=(12, 8))

    for row in final_grid:
        for point in row:
            if (point['x'] >= -4 and point['x'] <= 4 and point['y'] == -9) or (point['x'] >= -4 and point['x'] <= 3 and point['y'] == -8) or (point['x'] >= -3 and point['x'] <= 3 and point['y'] == -7) or (point['x'] >= -3 and point['x'] <= 2 and point['y'] == -6) or (point['x'] >= -2 and point['x'] <= 2 and point['y'] == -5) or (point['x'] >= -2 and point['x'] <= 1 and point['y'] == -4) or (point['x'] >= -1 and point['x'] <= 1 and point['y'] == -3) or (point['x'] >= -1 and point['x'] <= 0 and point['y'] == -2) or (point['x'] == -0 and point['y'] == -1):
                plt.scatter(point['x'], point['y'], marker='h', s=300, edgecolors='black', color='red')

            elif (point['x'] == 5 and point['y'] == -9) or (point['x'] >= 4 and point['x'] <= 5 and point['y'] == -8) or (point['x'] >= 4 and point['x'] <= 6 and point['y'] == -7) or (point['x'] >= 3 and point['x'] <= 6 and point['y'] == -6) or (point['x'] >= 3 and point['x'] <= 7 and point['y'] == -5) or (point['x'] >= 2 and point['x'] <= 7 and point['y'] == -4) or (point['x'] >= 2 and point['x'] <= 8 and point['y'] == -3) or (point['x'] >= 1 and point['x'] <= 8 and point['y'] == -2) or (point['x'] >= 1 and point['x'] <= 9 and point['y'] == -1):
                plt.scatter(point['x'], point['y'], marker='h', s=300, edgecolors='black', color='orange')

            elif (point['x'] == -5 and point['y'] == -8) or (point['x'] >= -5 and point['x'] <= -4 and point['y'] == -7) or (point['x'] >= -6 and point['x'] <= -3 and point['y'] == -6) or (point['x'] >= -6 and point['x'] <= -3 and point['y'] == -5) or (point['x'] >= -7 and point['x'] <= -3 and point['y'] == -4) or (point['x'] >= -7 and point['x'] <= -2 and point['y'] == -3) or (point['x'] >= -8 and point['x'] <= -2 and point['y'] == -2) or (point['x'] >= -8 and point['x'] <= -1 and point['y'] == -1) or (point['x'] >= -9 and point['x'] <= -1 and point['y'] == 0):
                plt.scatter(point['x'], point['y'], marker='h', s=300, edgecolors='black', color='purple')

            elif (point['x'] >= 1 and point['x'] <= 9 and point['y'] == 0) or (point['x'] >= 2 and point['x'] <= 9 and point['y'] == 1) or (point['x'] >= 2 and point['x'] <= 8 and point['y'] == 2) or (point['x'] >= 3 and point['x'] <= 8 and point['y'] == 3) or (point['x'] >= 3 and point['x'] <= 7 and point['y'] == 4) or (point['x'] >= 4 and point['x'] <= 7 and point['y'] == 5) or (point['x'] >= 4 and point['x'] <= 6 and point['y'] == 6) or (point['x'] >= 5 and point['x'] <= 6 and point['y'] == 7) or (point['x'] == 5 and point['y'] == 8):
                plt.scatter(point['x'], point['y'], marker='h', s=300, edgecolors='black', color='yellow')

            elif (point['x'] >= -3 and point['x'] <= 5 and point['y'] == 9) or (point['x'] >= -3 and point['x'] <= 4 and point['y'] == 8) or (point['x'] >= -2 and point['x'] <= 4 and point['y'] == 7) or (point['x'] >= -2 and point['x'] <= 3 and point['y'] == 6) or (point['x'] >= -1 and point['x'] <= 3 and point['y'] == 5) or (point['x'] >= -1 and point['x'] <= 2 and point['y'] == 4) or (point['x'] >= 0 and point['x'] <= 2 and point['y'] == 3) or (point['x'] >= 0 and point['x'] <= 1 and point['y'] == 2) or (point['x'] == 1 and point['y'] == 1):
                plt.scatter(point['x'], point['y'], marker='h', s=300, edgecolors='black', color='g')

            elif (point['x'] >= -8 and point['x'] <= 0 and point['y'] == 1) or (point['x'] >= -8 and point['x'] <= -1 and point['y'] == 2) or (point['x'] >= -7 and point['x'] <= -1 and point['y'] == 3) or (point['x'] >= -7 and point['x'] <= -2 and point['y'] == 4) or (point['x'] >= -6 and point['x'] <= -2 and point['y'] == 5) or (point['x'] >= -6 and point['x'] <= -3 and point['y'] == 6) or (point['x'] >= -5 and point['x'] <= -3 and point['y'] == 7) or (point['x'] >= -5 and point['x'] <= -4 and point['y'] == 8) or (point['x'] == 4 and point['y'] == 9):
                plt.scatter(point['x'], point['y'], marker='h', s=300, edgecolors='black', color='b')

    plt.scatter(x_values, y_values, marker='h', s=800, edgecolors='black', color='skyblue')

    # Annotate each node with its value
    for x, y, node in zip(x_values, y_values, node_values):
        plt.text(x, y, str(node), ha='center', va='center', fontsize=12, color='black', weight='bold')

# Customize the plot
    plt.title('Hexagonal Grid ')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(False)  

# Show the plot
    plt.show()

if __name__ == "__main__":
    main()