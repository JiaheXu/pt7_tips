
import os
import numpy as np
dir_path = './'
res = []
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        res.append(path)


for idx, file in enumerate(res):
    if(res[-4:-1]=='.py'):
        continue
    data = np.load(file, allow_pickle=True)
    new_data = []
    for point in data:
        point["left_ee"] = point["left_ee"] + 0.63
        point["right_ee"] = point["right_ee"] - 0.63
        new_data.append(point)
    np.save(str(idx),data)