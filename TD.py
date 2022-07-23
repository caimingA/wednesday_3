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



def get_V_map(maze):
    v_map = maze.astype(np.float)
    for row in range(len(maze)):
        for col in range(len(maze[0])):
            if maze[row][col] == 1:
                v_map[row][col] = -100.0
    return v_map


def get_reward(x, y, goal_x, goal_y):
    if (x == goal_x) and (y == goal_y):
        return 1
    return 0


def is_passable(maze, x, y):
    return maze[y][x] != 1


# TD difference
def get_TD(V_map, reward, x, y, gama, next_x, next_y):
      return (gama * V_map[next_y][next_x]) + reward - V_map[y][x]


def renew_value(V_map, TD, alpha, x, y):
      return V_map[y][x] + (alpha * TD)


# [up, left, down, right] = [0, 1, 2, 3]
def random_search_once(maze, V_map, x, y, gama, alpha, goal_x, goal_y):
    next_x = x
    next_y = y
    select = np.array([0, 1, 2, 3])
    np.random.shuffle(select)   
    
    for i in select:
        if i == 0:
            if is_passable(maze, x, y-1):
                next_y -= 1
                break
        elif i == 1:
            if is_passable(maze, x-1, y):
                next_x -= 1
                break
        elif i == 2:
            if is_passable(maze, x, y+1):
                next_y += 1
                break
        else:
            if is_passable(maze, x+1, y):
                next_x += 1
                break

  
    reward = get_reward(next_x, next_y, goal_x, goal_y)
    TD = get_TD(V_map, reward, x, y, gama, next_x, next_y)
    new_value = renew_value(V_map, TD, alpha, x, y)
    V_map[y][x] = new_value
    # print([next_x, next_y], "->")
    return [next_x, next_y]


# [up, left, down, right] = [0, 1, 2, 3]
def greedy_search_once(maze, V_map, x, y, gama, alpha, goal_x, goal_y):
    next_x = x
    next_y = y
    TD_max = -100
    TD = -100

    select = np.array([0, 1, 2, 3])
    np.random.shuffle(select)
    for i in select:

        if i == 0:
            if is_passable(maze, x, y-1):
                reward = get_reward(x, y-1, goal_x, goal_y)
                TD = get_TD(V_map, reward, x, y, gama, x, y-1)
                if(TD > TD_max):
                    next_x = x
                    next_y = y - 1
                    TD_max = TD
        elif i == 1:
            if is_passable(maze, x-1, y):
                reward = get_reward(x-1, y, goal_x, goal_y)
                TD = get_TD(V_map, reward, x, y, gama, x-1, y)
                if(TD > TD_max):
                    next_x = x - 1
                    next_y = y
                    TD_max = TD
        elif i == 2:
            if is_passable(maze, x, y+1):
                reward = get_reward(x, y+1, goal_x, goal_y)
                TD = get_TD(V_map, reward, x, y, gama, x, y+1)
                if(TD > TD_max):
                    next_x = x
                    next_y = y + 1
                    TD_max = TD
        else:
            if is_passable(maze, x+1, y):
                reward = get_reward(x+1, y, goal_x, goal_y)
                TD = get_TD(V_map, reward, x, y, gama, x+1, y)
                if(TD > TD_max):
                    next_x = x + 1
                    next_y = y
                    TD_max = TD
    
    new_value = renew_value(V_map, TD_max, alpha, x, y)
    V_map[y][x] = new_value
    return [next_x, next_y]


def epsilon_greedy(start_x, start_y, goal_x, goal_y, maze, V_map, gama, alpha, epsilon):
    pos_x = start_x
    pos_y = start_y
    policy_list = list()
    pos_list = list()
    # value_list = list()
    value = 0
    step = 0
    pos_list.append([1, 1])
    while pos_x != goal_x or pos_y != goal_y:
        value += V_map[pos_y][pos_x]
        select = np.random.rand()
        if select >= epsilon:
            [new_pos_x, new_pos_y] = greedy_search_once(maze, V_map, pos_x, pos_y, gama, alpha, goal_x, goal_y)
            policy_list.append(1)
            # print("policy", 1)
        else:
            [new_pos_x, new_pos_y] = random_search_once(maze, V_map, pos_x, pos_y, gama, alpha, goal_x, goal_y)
            policy_list.append(0)
            # print("policy", 0)
        pos_x = new_pos_x
        pos_y = new_pos_y
        # print(pos_x, pos_y)
    
        pos_list.append([new_pos_x, new_pos_y])
        
#         print("Step{:d}".format(step+1))
        step += 1
    return V_map, pos_list, policy_list, value/len(policy_list)


if __name__ == '__main__':
    maze = list()
    for line in open("test.txt","r"):
        
        # print(line)
        temp = list()
        for i in line:
            if i != ' ' and i != '\n':
                # print(i)
                temp.append(int(i))
        maze.append(temp)
        
    
    print(maze)
    maze = np.array(maze)

    height, width = maze.shape

    plt.figure(figsize=(10, 10))

    plt.imshow(maze, cmap="binary")

    plt.xticks(np.arange(width), np.arange(width))

    
    plt.yticks(np.arange(height), np.arange(height))
    
    for i in range(len(maze) - 1):
        plt.axhline(y=i+0.5,c='r',ls='--')
        plt.axvline(x=i+0.5,c='r',ls='--')


    # plt.show()
    # plt.close()

    V_map = get_V_map(maze)

    # print(V_map)
    pos_list_full = list()
    policy_list_full = list()
    value_list = list()
    pos_list_times = list()
    policy_list_times = list()
    value_list_times = list()
    V_map_list_times = list()
    epsilon_list = list()


    end_pos_x = len(maze) - 2
    end_pos_y = len(maze) - 2
    
    times = 10
    epoches = 2000
    epsilon_start = 0.9
    epsilon_end = 0.01
    decay_index = 100.0
    count = 0.0

    for i in range(epoches):
        print("========Epoch{:d}========".format(i+1))

        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(- count/decay_index)
        epsilon_list.append(epsilon)
        # epsilon = 0.5
        V_map, pos_list, policy_list, value = epsilon_greedy(1, 1, end_pos_x, end_pos_x, maze, V_map, 0.95, 0.2, epsilon)
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

    # plt.plot(np.arange(epoches), epsilon_list)

    value_sum = 0
    # value_list_times =list()
    value_list_temp = list()
    mean_list = list()
    err_list = list()
    for i in range(len(value_list)):
        value_list_temp.append(value_list[i])
        if i % 10 == 9:
            # value_list_times.append(value_sum)
            mean_list.append(np.mean(value_list_temp))
            err_list.append(np.std(value_list_temp))
            value_list_temp = list()


    mean_list = np.array(mean_list)
    err_list = np.array(err_list)
    # print(len(mean_list))

    plt.figure(figsize=(10, 10))
    plt.plot(np.arange(len(mean_list)) * 10, mean_list)
    plt.fill_between(np.arange(len(mean_list)) * 10, mean_list - err_list, mean_list + err_list, alpha=0.2)
    plt.show()    

    print(pos_list_full[epoches - 1])
