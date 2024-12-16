import numpy as np

def main():
    point_grid = generate_grid(1, 2, 3)

    grid = point_grid.copy()
    for row in grid:
        for square in row:  
            length = len(square)
            for i in range(4):

                if i >1:
                    del square[2]
    
    movement_combination = generate_combinations_n(16, grid)
    movement_combination_2 = generate_combinations_n_2(16, grid)

    test_grid = generate_grid(1,2,3)

    array_1 = code_breaker_1(movement_combination)
    array_2 = code_breaker_2(movement_combination_2)
    find_common_solutions(array_1, array_2)



def generate_grid(a, b ,c):
    grid = []
    for i in range(1,7):
        array = []
        for j in range(1,7):
            if i == 1 or (i == 2 and j >= 1 and j <= 4) or (i == 3 and j >=1 and j <= 2):
                temp = [j, i, a, 1]
            
            elif (i == 2 and j >=5 and j <= 6) or (i == 3 and j >= 3 and j <=6) or (i == 4 and j >=1 and j <= 4) or (i == 5 and j >= 1 and j <=2) :
                temp = [j, i, b, 2]
            
            else:
                temp = [j, i, c, 3]

            array.append(temp)
        grid.append(array)
    return grid


def not_equal_array(overall_array, current_array):
    for arr in overall_array:
        if np.array_equal(arr, current_array):
            return False
    
    return True

def generate_movements(array):
    possible_movement_list = []

    position = array[-1]


    movement_1 = [position[0] + 2, position[1] + 1]
    if movement_1[0] > 0 and movement_1[1] > 0 and movement_1[0] < 7 and movement_1[1] < 7 and not_equal_array(array, movement_1):
        possible_movement_list.append(movement_1)

    movement_2 = [position[0] + 2, position[1] - 1]
    if movement_2[0] > 0 and movement_2[1] > 0 and movement_2[0] < 7 and movement_2[1] < 7 and not_equal_array(array, movement_2):
        possible_movement_list.append(movement_2)
    
    movement_3 = [position[0] + 1, position[1] + 2]
    if movement_3[0] > 0 and movement_3[1] > 0 and movement_3[0] < 7 and movement_3[1] < 7 and not_equal_array(array, movement_3):
        possible_movement_list.append(movement_3)
    
    movement_4 = [position[0] - 1, position[1] + 2]
    if movement_4[0] > 0 and movement_4[1] > 0 and movement_4[0] < 7 and movement_4[1] < 7 and not_equal_array(array, movement_4):
        possible_movement_list.append(movement_4)
    
    movement_5 = [position[0] - 2, position[1] + 1]
    if movement_5[0] > 0 and movement_5[1] > 0 and movement_5[0] < 7 and movement_5[1] < 7 and not_equal_array(array, movement_5):
        possible_movement_list.append(movement_5)
    
    movement_6 = [position[0] - 2, position[1] - 1]
    if movement_6[0] > 0 and movement_6[1] > 0 and movement_6[0] < 7 and movement_6[1] < 7 and not_equal_array(array, movement_6):
        possible_movement_list.append(movement_6)

    movement_7 = [position[0] - 1, position[1] - 2]
    if movement_7[0] > 0 and movement_7[1] > 0 and movement_7[0] < 7 and movement_7[1] < 7 and not_equal_array(array, movement_7):
        possible_movement_list.append(movement_7)

    movement_8 = [position[0] + 1, position[1] - 2]
    if movement_8[0] > 0 and movement_8[1] > 0 and movement_8[0] < 7 and movement_8[1] < 7 and not_equal_array(array, movement_8):
        possible_movement_list.append(movement_8)

    return possible_movement_list
    

def generate_combinations_n(n, grid):
    movement_combinations = []
    array = []
    current_position = grid[0][0]
    for i in range(n):
        if i == 0:
            movement_combinations = []

            for mov in generate_movements([current_position]):
                movement_combinations.append([mov])

        else:
            array = []
            print(len(movement_combinations))
            for elem in movement_combinations:
                possible_movements = generate_movements(elem)

                for movement in possible_movements:
                    temp = elem.copy()
                    temp.append(movement)
                    array.append(temp)
                #print(array)

            movement_combinations = array.copy()

        #print(movement_combinations)
        #print(array)
    #print(movement_combinations)
    movement_combinations = [elem for elem in movement_combinations if elem[-1] == [6,6]]
    #print(len(movement_combinations))
    return movement_combinations

def generate_combinations_n_2(n, grid):
    movement_combinations = []
    array = []
    current_position = grid[0][5]
    for i in range(n):
        if i == 0:
            movement_combinations = []

            for mov in generate_movements([current_position]):
                movement_combinations.append([mov])

        else:
            print(len(movement_combinations))
            array = []
            for elem in movement_combinations:
                possible_movements = generate_movements(elem)

                for movement in possible_movements:
                    temp = elem.copy()
                    temp.append(movement)
                    array.append(temp)

            movement_combinations = array.copy()


    movement_combinations = [elem for elem in movement_combinations if elem[-1] == [1,6]]
    #print(len(movement_combinations))
    return movement_combinations


def calculate_score(move_set, grid):
    sum = grid[0][0][2]
    previous_const = 0
    previous_position = [1,1]
    for move in move_set:
        column = move[1]
        row = move[0]

        if previous_const == 0:
            previous_const = grid[column-1][row-1][3]
            previous_position = [grid[column-1][row-1][0], grid[column-1][row-1][1]]
            sum += grid[column-1][row-1][2]
        
        else:
            if grid[column-1][row-1][3] == previous_const:
                sum += grid[column-1][row-1][2]
            else:
                sum = sum * grid[column-1][row-1][2]
                previous_const = grid[column-1][row-1][3]
    return sum

def calculate_score_2(move_set, grid):
    sum = grid[0][5][2]
    previous_const = grid[0][5][3]
    for move in move_set:
        column = move[1]
        row = move[0]

        if previous_const == 0:
            previous_const = grid[column-1][row-1][3]
            sum += grid[column-1][row-1][2]
        
        else:
            if grid[column-1][row-1][3] == previous_const:
                sum += grid[column-1][row-1][2]
            else:
                sum = sum * grid[column-1][row-1][2]
                previous_const = grid[column-1][row-1][3]
    return sum

def code_breaker_1(movement_combination):
    counter = 0
    solutions = []
    for x1 in range(1,4):
        print(x1)
        for x2 in range(1, 6 - x1):
            for x3 in range(1, 6 - x1 - x2):
                prototype_grid = generate_grid(x1, x2, x3)
                for elem in movement_combination:
                    sum = calculate_score(elem, prototype_grid)
                    if sum == 2024:
                        counter += 1
                        dic = {'movement': elem, 'a': x1, 'b': x2, 'c': x3}
                        solutions.append(dic)
                        #print(sum)
                        #print(f"Solution found at {elem} with a {x1}, b {x2}, c {x3}")

    print(counter)
    return solutions

def code_breaker_2(movement_combination):
    counter = 0
    solutions = []
    for x1 in range(1,4):
        print(x1)
        for x2 in range(1, 6 - x1):
            for x3 in range(1, 6 - x1 - x2):
                prototype_grid = generate_grid(x1, x2, x3)
                for elem in movement_combination:
                    sum = calculate_score_2(elem, prototype_grid)
                    if sum == 2024:
                        counter += 1
                        dic = {'movement': elem, 'a': x1, 'b': x2, 'c': x3}
                        solutions.append(dic)
                        #print(sum)
                        #print(f"Solution found at {elem} with a {x1}, b {x2}, c {x3}")

    print(counter)
    return solutions

def find_common_solutions(array_1, array_2):
    set_a_1 = set()
    set_a_2 = set()

    for elem in array_1:
        set_a_1.add((elem['a'], elem['b'], elem['c']))

    for elem in array_2:
        set_a_2.add((elem['a'], elem['b'], elem['c']))

    solutions = set_a_1 & set_a_2
    print(len(set_a_1 & set_a_2))

    solutions_array = []
    for tupple in solutions:
        a,b,c = tupple
        first = 1
        second = 1
        for elem in array_1:
            if a == elem['a'] and b == elem['b'] and c == elem['c'] and first == 1:
                first = 0
                dic = {'movement_1': elem['movement'], 'a': a, 'b': b, 'c': c}
                solutions_array.append(dic)
        
        for elem in array_2:
            if a == elem['a'] and b == elem['b'] and c == elem['c'] and second == 1:
                second = 0
                solutions_array[len(solutions_array) - 1]['movement_2'] = elem['movement']     

    with open('solutions_14n.txt', 'w') as file:
        for item in solutions_array:
            file.write(f"{item}\n")   

if __name__ == "__main__":
    main()