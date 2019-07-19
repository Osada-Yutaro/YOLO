def random_shift(inp, oup):
    from ..architecture import constants as const
    import random
    import numpy as np
    import cv2

    width = 448
    height = 448
    S = const.S
    B = const.B
    C = const.C
    x_min = int(width*random.random()/10)
    x_max = int(width*(random.random()/10 + 0.9))
    y_min = int(height*random.random()/10)
    y_max = int(height*(random.random()/10 + 0.9))

    W_new = x_max - x_min
    H_new = y_max - y_min

    inp_new = cv2.resize(inp[x_min:x_max, y_min:y_max, :], dsize=(448, 448))
    oup_new = np.zeros([7, 7, 30], dtype='float32')

    for sx in range(S):
        for sy in range(S):
            for b in range(B):
                boundingbox = oup[sx, sy, C + 5*b: C + 5*b + 5]
                x_mid = (sx + boundingbox[0])*width/S
                y_mid = (sy + boundingbox[1])*height/S

                left = x_mid - boundingbox[2]*width/2
                right = x_mid + boundingbox[2]*width/2
                up = y_mid - boundingbox[3]*height/2
                down = y_mid + boundingbox[3]*height/2

                if (boundingbox[4] == 1.) and (x_min < right < x_max or x_min < left < x_max) and (y_min < down < y_max or y_min < up < y_max):
                    l_new = max(left, x_min)
                    r_new = min(right, x_max)
                    u_new = max(up, y_min)
                    d_new = min(down, y_max)

                    x_mid_new = ((l_new + r_new)/2 - x_min)*S/W_new
                    y_mid_new = ((d_new + u_new)/2 - y_min)*S/H_new
                    w_new = (r_new - l_new)/W_new
                    h_new = (d_new - u_new)/H_new

                    sx_new = int(x_mid_new)
                    sy_new = int(y_mid_new)

                    oup_new[sx_new, sy_new, C + 5*b + 0] = x_mid_new - sx_new
                    oup_new[sx_new, sy_new, C + 5*b + 1] = y_mid_new - sy_new
                    oup_new[sx_new, sy_new, C + 5*b + 2] = w_new
                    oup_new[sx_new, sy_new, C + 5*b + 3] = h_new
                    oup_new[sx_new, sy_new, C + 5*b + 4] = 1.

                    oup_new[sx_new, sy_new, 0:C] = np.maximum(oup[sx, sy, 0:C], oup_new[sx_new, sy_new, 0:C])

    return inp_new, oup_new

def random_reverse(inp, oup):
    import random
    import numpy as np
    import cv2
    C = 20
    if random.randint(0, 255)%2 == 0:
        eye = np.eye(30, dtype='float32')[C] + np.eye(30, dtype='float32')[C + 5]
        return cv2.flip(inp, 1), np.apply_along_axis(lambda x: (1 - eye)*x - eye*x + eye, 2, np.flip(oup, 0))
    return inp, oup

def convert_X(data_dir, filename):
    import os
    import numpy as np
    import cv2
    return cv2.resize(cv2.imread(os.path.join(data_dir, 'JPEGImages', filename + '.jpg')), dsize=(448, 448)).astype('float32')

def convert_Y(data_dir, filename):
    import os
    import numpy as np
    return np.load(os.path.join(data_dir, 'Segmentation', filename) + '.npy')
