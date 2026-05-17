from march.pyt.mc import mc

import torch


"""
Edge and vertex convention:

                    v4_____________________v5_____________________v10
                    /|                    /|                     /|
                   / |                   / |                    / |
                  /  |                  /  |                   /  |
                 /___|_________________/___|__________________/   |
              v6|    |                 |v7 |                  |v11|
                |    |                 |   |                  |   |
                |    |                 |   |                  |   |
                |    |                 |   |                  |   |
                |    |_________________|___|__________________|___|
                |   / v0               |   / v1               |   / v8
                |  /                   |  /                   |  / 
                | /                    | /                    | /  
                |/_____________________|/_____________________|/
                v2                     v3                     v9
"""


if __name__ == "__main__":
    
    grids = torch.tensor([
        [0, 0, 0], #v0
        [1, 0, 0], #v1
        [0, 1, 0], #v2
        [1, 1, 0], #v3

        [0, 0, 1], #v4
        [1, 0, 1], #v5
        [0, 1, 1], #v6
        [1, 1, 1], #v7

        [2, 0, 0], #v8
        [2, 1, 0], #v9
        [2, 0, 1], #v10
        [2, 1, 1]  #v11
    ], dtype=torch.float32)

    cubes = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 8, 3, 9, 5, 10, 7, 11]
    ], dtype=torch.long)

    values = torch.tensor(
        [
            1,        #v0
            -0.5,     #v1
            1,        #v2
            -0.5,     #v3
            1,        #v4
            1,        #v5
            1,        #v6
            1,        #v7
            1,        #v8
            1,        #v9
            1,        #v10
            1         #v11
        ], dtype=torch.float32
    )
    
    iso = 0.0
    
    verts, faces = mc(grids, cubes, values, iso)
    
    print("# verts:", verts.shape[0])
    print("# faces:", faces.shape[0])
    
    print("Verts:")
    print(verts)
    
    print("Faces:")
    print(faces)