import numpy as np
import matplotlib.pyplot as plt
import random


def plot_maze(maze_map, save_file=None):
    height, width = maze_map.shape
    plt.imshow(maze_map, cmap="binary")
    plt.xticks(rotation=90)
    plt.xticks(np.arange(width), np.arange(width))
    plt.yticks(np.arange(height), np.arange(height))

    plt.gca().set_aspect('equal')
    if save_file is not None:
        plt.savefig(save_file)
    else:
        plt.show()


import random as rd
 
nearmaybe=[
    [-2,0],
    [2,0],
    [0,-2],
    [0,2]
]
 
def createMaze(row,col):
    maze=[[0 for i in range(col)] for i in range(row)]
    check=[]
    firstrow = random.randrange(1,row-2,2)
    firstcol = random.randrange(1,col-2,2)
    maze[firstrow][firstcol]=1
    check.append([firstrow,firstcol])
    while len(check):
        c = random.choice(check)
        nears = list()
        conditions = list()
        for maybe in nearmaybe:
            conditions.append([c[0]+maybe[0],c[1]+maybe[1]])
        for condition in conditions:
            if condition[0] >= 1 and condition[0] <= row-2 and condition[1] >= 1 and condition[1] <= col-2:
                nears.append([condition[0],condition[1]])
        for n in nears.copy():
            if maze[n[0]][n[1]]:
                nears.remove(n)
        for block in nears:
            if block[0] == c[0]:
                if block[1]<c[1]:
                    maze[block[0]][c[1]-1] = 1
                    maze[block[0]][block[1]] = 1
                    check.append([block[0],block[1]])
                else:
                    maze[block[0]][block[1]-1] = 1
                    maze[block[0]][block[1]] = 1
                    check.append([block[0],block[1]])
            else:
                if block[0]<c[0]:
                    maze[c[0]-1][block[1]] = 1
                    maze[block[0]][block[1]] = 1
                    check.append([block[0],block[1]])
                else:
                    maze[block[0]-1][block[1]] = 1
                    maze[block[0]][block[1]] = 1
                    check.append([block[0],block[1]])
        if not len(nears):
            check.remove(c)

    for i in range(len(maze)):
        for j in range(len(maze[0])):
            # print(1 - maze[i][j])
            maze[i][j] = 1 - maze[i][j]
 
    return maze



if __name__ == "__main__":
    maze=np.array(createMaze(37,37)) # 35 * 35
    name = "3"
    plot_maze(maze, name)
    np.savetxt(f"{name}.txt", maze, fmt="%d")
    print(maze)
