import cv2

def read_files(filename):
    obj_list = []
    with open(filename, 'r')as f:
        next(f)
        for line in f:
            parts = [p.strip() for p in line.split(',')]

            if len(parts) != 5:
                print(f"Warning: skip the line {line.strip()}")
                continue

            try:
                obj = {
                    'class':parts[0],
                    'x1':int(parts[1]),
                    'y1':int(parts[2]),
                    'x2':int(parts[3]),
                    'y2':int(parts[4])

                }
                obj_list.append(obj)
            except ValueError:
                print(f"Warning:fail to convert {line.strip()}")
    return obj_list

def connect_object(u, v, filename):
    obj_list = read_files(filename=filename)
    target_list = []
    for obj in obj_list:
        if obj['x1'] <= u and obj['x2'] >= u and obj['y1'] <= v and obj['y2'] >= v:
            target_list.append(obj['class'])

    return target_list