# utilities functions
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
SIGMA = 1e-15

def similitudeMoments(img):
    return [n_moment(img,0,2),n_moment(img,0,3),n_moment(img,1,1),
        n_moment(img,1,2),n_moment(img,2,0),n_moment(img,2,1),n_moment(img,3,0)]

def n_moment(img,i,j):
    r,c = img.shape
    img = img.astype(np.float)
    y = np.arange(r)[:,None]
    x = np.arange(c)[None,:]
    m00 = np.sum(img)
    m10 = np.sum(img*x)
    m01 = np.sum(img*y)
    x_bar = m10/(m00+SIGMA)
    y_bar = m01/(m00+SIGMA)
    uij = np.sum(img * (x-x_bar)**i * (y-y_bar)**j)
    
    return uij / (m00 ** ((i+j)/2 +1)+SIGMA)


def img_to_binary(img,window_size = 30, threshold = 50):
    # do comparison for small patch
    gray_scale = np.array(img.convert("L")).astype(np.float)
    height, width = gray_scale.shape
    local_max = np.zeros_like(gray_scale)
    for r in range(0,height,window_size):
        for c in range(0,width,window_size):
            rr = r + window_size if r+window_size < height else height
            cc = c + window_size if c+window_size < width else width
            # print(np.max(self.gray_scale[r:rr,c:cc]))
            local_max[r:rr,c:cc] = np.max(gray_scale[r:rr,c:cc])
    return local_max - gray_scale > threshold


class TempMatcher:
    # this class only take binary input for performance reason
    def __init__(self, T):
        #initialize with template
        self.T = T.astype(np.bool)

    def set_template(self,T):
        self.T = T.astype(np.bool)

    def set_image(self,image):
        self.img = image
        self.w,self.h = image.size

    def match(self,scale=1):
        th,tw = self.T.shape
        direction = 0.1
        direction_decided = False
        s = scale
        while True:
            new_w = int(self.w * s)
            new_h = int(self.h * s)
            img = self.img.resize((new_w,new_h))
            binary = img_to_binary(img)
            dis_map = np.zeros((new_h-th+1,new_w-tw+1))
            for h in range(new_h-th+1):
                for w in range(new_w-tw+1):
                    patch = binary[h:h+th,w:w+tw]
                    dis_map[h,w] = self._distance(patch)
            # dis_map = nonmin_supress(dis_map)
            sorted_index = np.argsort(dis_map,axis=None)
            sorted_index = np.unravel_index(sorted_index,shape = dis_map.shape)
            chosen = [(sorted_index[1][0],sorted_index[0][0])]
            min_value = dis_map[sorted_index][0]
            for index in range(dis_map.size):
                value = dis_map[sorted_index][index]
                y = sorted_index[0][index]
                x = sorted_index[1][index]
                if value-min_value>0.01:
                    break
                if all(abs(y-yy) > 50 or abs(x-xx)>50 for (xx,yy) in chosen):
                    chosen.append((x,y))
            
            print(chosen)
            # plt.imshow(binary)
            # plt.show()
            print(min_value)
            print(s)

            # search until min is found
            try:
                if last < min_value:
                    # distance become larger
                    if not direction_decided:
                        s = scale
                        direction_decided = True
                        direction = -0.1
                    else:
                        return last
                        # and also coords
                else:
                    # distance become smaller
                    direction_decided = True
                    last = min_value
            except NameError:
                last = min_value

            s += direction

    def _distance(self, P):
        # call with patch
        T = self.T
        assert P.shape == T.shape
        r = T ^ P
        return np.sum(r)/r.size 

def nonmin_supress(img, kernel = 20):
    max_value = 1
    k = (kernel-1) //2
    rr,cc = img.shape
    for r in range(rr):
        for c in range(cc):
            rmin = max(r-k,0)
            rmax = min(r+k+1,rr-1)
            cmin = max(c-k,0)
            cmax = min(c+k+1,cc-1)
            less_than_other = True
            for i in range(rmin,rmax,1):
                for j in range(cmin,cmax,1):
                    if r != i and c != j and img[r,c] >= img[i,j]:
                        less_than_other = False
            if not less_than_other:
                img[r,c] = max_value
    return img

