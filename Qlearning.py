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


# maze = np.array(
#         [
#             [1, 1, 1, 1, 1, 1, 1],
#             [1, 0, 0, 0, 0, 0, 1],
#             [1, 0, 0, 1, 1, 0, 1],
#             [1, 1, 1, 1, 0, 0, 1],
#             [1, 0, 0, 0, 0, 0, 1],
#             [1, 0, 0, 0, 0, 0, 1],
#             [1, 1, 1, 1, 1, 1, 1]
#         ]
#     )



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
    Q_max = -1000
    Q = -1000

    select = np.array([0, 1, 2, 3])
    np.random.shuffle(select)
    
    for i in select:
        if i == 0:
            if is_passable(maze, x, y-1):
                Q = Q_table[(y, x)][0]
                if Q > Q_max:
                    next_x = x
                    next_y = y - 1
                    Q_max = Q
                    direction = 0
        elif i == 1:
            if is_passable(maze, x-1, y):
                Q = Q_table[(y, x)][1]
                if Q > Q_max:
                    next_x = x - 1
                    next_y = y
                    Q_max = Q
                    direction = 1
        elif i == 2:
            if is_passable(maze, x, y+1):    
                Q = Q_table[(y, x)][2]
                if Q > Q_max:
                    next_x = x
                    next_y = y + 1
                    Q_max = Q
                    direction = 2
        else:
            if is_passable(maze, x+1, y):
                Q = Q_table[(y, x)][3]
                if Q > Q_max:
                    next_x = x + 1
                    next_y = y
                    Q_max = Q
                    direction = 3

    reward = get_reward(next_x, next_y, goal_x, goal_y)
    TD = get_TD(Q_table, reward, x, y, direction, gama, next_x, next_y)
    
    new_value = renew_value(Q_table, TD, alpha, x, y, direction)
    Q_table[(y, x)][direction] = new_value
    # print("(", next_x, ", ", next_y, ")->", end=" ")
    return [next_x, next_y]


def epsilon_greedy(start_x, start_y, goal_x, goal_y, maze, Q_table, gama, alpha, epsilon):
    pos_x = start_x
    pos_y = start_y
    policy_list = list()
    pos_list = list()
    value = 0
    step = 0
    pos_list.append([1, 1])
    while pos_x != goal_x or pos_y != goal_y:
        value += max(Q_table[(pos_y, pos_x)])
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
    return Q_table, pos_list, policy_list, value / len(policy_list)


def epsilon_greedy_decay(start_x, start_y, goal_x, goal_y, maze, Q_table, gama, alpha, epsilon_start, epsilon_end, decay_index):
    pos_x = start_x
    pos_y = start_y
    policy_list = list()
    pos_list = list()
    step = 0
    pos_list.append([1, 1])
    count = 0
    while pos_x != goal_x or pos_y != goal_y:
        epsilon = epsilon_end+(epsilon_start - epsilon_end) * np.exp(- count/decay_index)
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
        count += 1
        
#         print("Step{:d}".format(step+1))
        step += 1
    return Q_table, pos_list, policy_list


if __name__ == '__main__':
    maze = list()
    for line in open("test.txt","r"): #设置文件对象并读取每一行文件
        
        # print(line)
        temp = list()
        for i in line:
            if i != ' ' and i != '\n':
                # print(i)
                temp.append(int(i))
        maze.append(temp)
        
    
    # print(maze)
    maze = np.array(maze)

    height, width = maze.shape

    plt.figure(figsize=(10, 10))
    
    plt.imshow(maze, cmap="binary")

    plt.xticks(np.arange(width), np.arange(width))

    plt.yticks(np.arange(height), np.arange(height))
    
    for i in range(len(maze) - 1):
        plt.axhline(y=i+0.5,c='r',ls='--')
        plt.axvline(x=i+0.5,c='r',ls='--')


    Q_table = init_Q_table(maze)

    # print(V_map)
    pos_list_full = list()
    policy_list_full = list()
    value_list = list()
    pos_list_times = list()
    policy_list_times = list()
    value_list_times = list()
    V_map_list_times = list()


    end_pos_x = len(maze) - 2
    end_pos_y = len(maze) - 2
    
    times = 10
    epoches = 1000
    epsilon_start = 0.9
    epsilon_end = 0.01
    decay_index = 100.0
    count = 0.0

    for i in range(epoches):
        print("\n========Epoch{:d}========".format(i+1))

        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(- count/decay_index)
        # epsilon = 0.3
        # print(epsilon)
        # epsilon_list.append(epsilon)
        Q_table, pos_list, policy_list, value = epsilon_greedy(1, 1, end_pos_x, end_pos_x, maze, Q_table, 0.95, 0.2, epsilon)
        # Q_table, pos_list, policy_list = epsilon_greedy_decay(1, 1, 5, 5, maze, Q_table, 0.95, 0.2, 0.9, 0.01, 10)
        # print(V_map)
        pos_list_full.append(pos_list)
        policy_list_full.append(policy_list)
        value_list.append(value)
        count += 1
        
    for i in range(1, len(pos_list_full[epoches - 1])):
        bias = 0
        if pos_list_full[epoches - 1][i][0] - pos_list_full[epoches - 1][i - 1][0] == 0:
            if pos_list_full[epoches - 1][i][1] - pos_list_full[epoches - 1][i - 1][1] > 0:
                bias = -0.4
            else:
                bias = 0.4

            plt.arrow(pos_list_full[epoches - 1][i - 1][0]
            , pos_list_full[epoches - 1][i - 1][1]
            , pos_list_full[epoches - 1][i][0] - pos_list_full[epoches - 1][i - 1][0] 
            , pos_list_full[epoches - 1][i][1] - pos_list_full[epoches - 1][i - 1][1] + bias
            , head_width=0.25
            )
        else:
            if pos_list_full[epoches - 1][i][0] - pos_list_full[epoches - 1][i - 1][0] > 0:
                bias = -0.4
            else:
                bias = 0.4
            plt.arrow(pos_list_full[epoches - 1][i - 1][0]
            , pos_list_full[epoches - 1][i - 1][1]
            , pos_list_full[epoches - 1][i][0] - pos_list_full[epoches - 1][i - 1][0] + bias
            , pos_list_full[epoches - 1][i][1] - pos_list_full[epoches - 1][i - 1][1] 
            , head_width=0.25
            )


    plt.show()
    plt.close()

    value_sum = 0
    # value_list_times =list()
    value_list_temp = list()
    mean_list = list()
    err_list = list()
    for i in range(len(value_list)):
        value_list_temp.append(value_list[i])
        if i % 5 == 4:
            # value_list_times.append(value_sum)
            mean_list.append(np.mean(value_list_temp))
            err_list.append(np.std(value_list_temp))
            value_list_temp = list()


    print(value_list)
    mean_list = np.array(mean_list)
    err_list = np.array(err_list)
    print(len(mean_list))

    plt.figure(figsize=(10, 10))
    plt.plot(np.arange(len(mean_list)) * 5, mean_list)
    plt.fill_between(np.arange(len(mean_list)) * 5, mean_list - err_list, mean_list + err_list, alpha=0.2)
    plt.show()

    print(pos_list_full[epoches - 1])
