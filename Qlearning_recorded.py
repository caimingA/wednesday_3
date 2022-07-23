import numpy as np
import matplotlib.pyplot as plt

# maze = np.array(
#         [
#             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#             [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
#             [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
#             [1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1],
#             [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
#             [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
#             [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1],
#             [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
#             [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1],
#             [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
#             [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
#             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#         ]
#     )


maze = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 1],
            [1, 1, 1, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1]
        ]
    )



def init_Q_table(maze):
    Q_table = dict()
    for row in range(1, len(maze) - 1):
        for col in range(1, len(maze[0]) - 1):
            if maze[row][col] == 0:
                Q_table[(row, col)] = [0, 0, 0, 0] # up, left, down, right
                for i in range(4):
                    if i == 0:
                        if maze[row - 1][col] == 1:
                            Q_table[(row, col)][0] = -1000.0
                    elif i == 1:
                        if maze[row][col - 1] == 1:
                            Q_table[(row, col)][1] = -1000.0
                    elif i == 2:
                        if maze[row + 1][col] == 1:
                            Q_table[(row, col)][2] = -1000.0
                    else:
                        if maze[row][col + 1] == 1:
                            Q_table[(row, col)][3] = -1000.0

    return Q_table


# checked
def get_reward(x, y, goal_x, goal_y):
    if (x == goal_x) and (y == goal_y):
        return 1
    return 0


# checked
def is_passable(maze, x, y):
    return maze[y][x] != 1


# checked
# TD difference
def get_TD(Q_table, reward, x, y, direction, gama, next_x, next_y):
      return (gama * max(Q_table[(next_y, next_x)])) + reward - Q_table[(y, x)][direction]

# checked
def renew_value(Q_table, TD, alpha, x, y, direction):
      return Q_table[(y, x)][direction] + (alpha * TD)


# [up, left, down, right] = [0, 1, 2, 3]
def random_search_once(maze, Q_table, x, y, gama, alpha, goal_x, goal_y):
    next_x = x
    next_y = y
    direction = 0
    select = np.array([0, 1, 2, 3])
    np.random.shuffle(select)   
    
    for i in select:
        if i == 0:
            if is_passable(maze, x, y-1):
                next_y -= 1
                direction = 0
                break
        elif i == 1:
            if is_passable(maze, x-1, y):
                next_x -= 1
                direction = 1
                break
        elif i == 2:
            if is_passable(maze, x, y+1):
                next_y += 1
                direction = 2
                break
        else:
            if is_passable(maze, x+1, y):
                next_x += 1
                direction = 3
                break
        
  
    reward = get_reward(next_x, next_y, goal_x, goal_y)
    if next_x == goal_x and next_y == goal_y:
        TD = reward
    else:
        TD = get_TD(Q_table, reward, x, y, direction, gama, next_x, next_y)
    new_value = renew_value(Q_table, TD, alpha, x, y, direction)
    Q_table[(y,x)][direction] = new_value
    # print([next_x, next_y], "->")
    return [next_x, next_y]


# checked
def Q_search_once(maze, Q_table, x, y, gama, alpha, goal_x, goal_y):
    next_x = x
    next_y = y
    direction = 0
    TD_max = -1000
    TD = -1000

    select = np.array([0, 1, 2, 3])
    np.random.shuffle(select)
    
    for i in select:
        if i == 0:
            if is_passable(maze, x, y-1):
                direction = 0
                reward = get_reward(x, y-1, goal_x, goal_y)
                TD = get_TD(Q_table, reward, x, y, direction, gama, x, y-1)
                if x == goal_x and (y - 1) == goal_y:
                    next_x = x
                    next_y = y - 1
                    TD_max = reward
                    break

                if TD > TD_max:
                    next_x = x
                    next_y = y - 1
                    TD_max = TD
        elif i == 1:
            if is_passable(maze, x-1, y):
                direction = 1
                reward = get_reward(x-1, y, goal_x, goal_y)
                TD = get_TD(Q_table, reward, x, y, direction, gama, x-1, y)
                if (x - 1) == goal_x and y == goal_y:
                    next_x = x - 1
                    next_y = y
                    TD_max = reward
                    break

                if TD > TD_max:
                    next_x = x - 1
                    next_y = y
                    TD_max = TD
        elif i == 2:
            if is_passable(maze, x, y+1):
                direction = 2
                reward = get_reward(x, y+1, goal_x, goal_y)
                TD = get_TD(Q_table, reward, x, y, direction, gama, x, y+1)
                if x == goal_x and (y + 1) == goal_y:
                    next_x = x
                    next_y = y + 1
                    TD_max = reward
                    break

                if TD > TD_max:
                    next_x = x
                    next_y = y + 1
                    TD_max = TD
        else:
            if is_passable(maze, x+1, y):
                direction = 3
                reward = get_reward(x+1, y, goal_x, goal_y)
                TD = get_TD(Q_table, reward, x, y, direction, gama, x+1, y)
                if (x + 1) == goal_x and y == goal_y:
                    next_x = x + 1
                    next_y = y
                    TD_max = reward
                    break

                if TD > TD_max:
                    next_x = x + 1
                    next_y = y
                    TD_max = TD
    
    new_value = renew_value(Q_table, TD_max, alpha, x, y, direction)
    Q_table[(y, x)][direction] = new_value
    print("(", next_x, ", ", next_y, ")->", end=" ")
    return [next_x, next_y]


def epsilon_greedy(start_x, start_y, goal_x, goal_y, maze, Q_table, gama, alpha, epsilon):
    pos_x = start_x
    pos_y = start_y
    policy_list = list()
    pos_list = list()
    step = 0
    pos_list.append([1, 1])
    while pos_x != goal_x or pos_y != goal_y:
        select = np.random.rand()
        if select >= epsilon:
            [new_pos_x, new_pos_y] = Q_search_once(maze, Q_table, pos_x, pos_y, gama, alpha, goal_x, goal_y)
            policy_list.append(1)
            # print("policy", 1)
        else:
            [new_pos_x, new_pos_y] = random_search_once(maze, Q_table, pos_x, pos_y, gama, alpha, goal_x, goal_y)
            policy_list.append(0)
            # print("policy", 0)
        pos_x = new_pos_x
        pos_y = new_pos_y
        
        # print(pos_x, pos_y)
    
        pos_list.append([new_pos_x, new_pos_y])
        
#         print("Step{:d}".format(step+1))
        step += 1
    return Q_table, pos_list, policy_list


if __name__ == '__main__':
    Q_table = init_Q_table(maze)
    print(Q_table)
    pos_list_full = list()
    policy_list_full = list()
    for i in range(60):
        print("\n========Epoch{:d}========".format(i+1))
        Q_table, pos_list, policy_list = epsilon_greedy(1, 1, 5, 5, maze, Q_table, 0.95, 0.2, 0)
        # print(V_map)
        pos_list_full.append(pos_list)
        policy_list_full.append(policy_list)
        
    print(Q_table)
    print(pos_list_full[10 - 1])
    print(policy_list_full[10 - 1])

