from march.pyt.data import FACE_TABLE

import os

if __name__ == "__main__":
    
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "lookup_table.txt")
    
    merged = []
    for i, face in enumerate(FACE_TABLE):
        for edge in face:
            merged.append(edge)
    
    with open(filename, "w") as f:
        f.write(f"Length of merged table: {len(merged)}\n\n")
    
        for i, edge in enumerate(FACE_TABLE):
            if len(edge) == 0:
                continue
            s = ', '.join(str(e) for e in edge) + ','
            f.write(f"{s}\n")