from PIL import Image
import numpy as np
import math
from matplotlib import pyplot as plt
import json
from utils import *


class Segmenter:
    def __init__(self,filename):
        image_object = Image.open(filename)
        width, height = image_object.size
        if width > 1300:
            height = int(height*1200/width)
            width = 1200
            image_object=image_object.resize((width,height))
        self.height = height
        self.width = width

        self.img = image_object

        # background subtraction
        self.binary = img_to_binary(self.img,window_size=10,threshold=20)
        Image.fromarray(self.binary).show()
        # plt.imshow(self.binary)
        # plt.show()

        ## high and low template
        # temp_img = Image.open("Resources/The-Summit.jpg")
        # temp_bin = img_to_binary(temp_img)
        # self.high_t = temp_bin[291:371,103:135]
        # self.low_t = temp_bin[680:718,103:135]
        
        ## high and low moments
        # Image.fromarray(self.binary).show()
        # with open("high.json","w") as file:
        #     json.dump(similitudeMoments(high),file)
        # with open("low.json","w") as file:
        #     json.dump(similitudeMoments(low),file)
        
    def segment(self):
        hh = self.height
        ww = self.width
        window_w = 50
        window_h = 50
        w1 = ww//2 -window_w//2
        w2 = w1 + window_w
        h1 = 0
        h2 = h1 + window_h
        line_pos = []
        while h2 <= hh:
            patch = self.binary[h1:h2,w1:w2]
            # detect lines <= 3 deg, take at least 60% of pixels, min line distance bewteen them is 6
            lines = self._line_detect(patch,theta_max_in_deg=5,pixel_ratio=0.8,min_k_distance=10,min_b_distance=6)
            if len(lines) == 5:
                lines = np.array(lines)
                ks = lines[:,0]
                bs = lines[:,1]
                # the patch can not be all 1
                if np.sum(patch)>0.5 * patch.size:
                    h1 += 1
                    h2 += 1
                    continue
                # check the lines are parallel and distance is same.
                b_sorted = np.sort(bs)
                b_step = []
                for i in range(4):
                    b_step.append(b_sorted[i+1]-b_sorted[i])
                if max(b_step)-min(b_step)>5:
                    h1 += 1
                    h2 += 1
                    continue
                # print(ks)
                ### visualize the lines ###
                height, width = patch.shape
                test = np.copy(patch)
                for k,b in lines:
                    for x in range(width):
                        y = int(k*x+b)
                        if y >=0 and y<height:
                            test[y,x] = True
                plt.imshow(test)
                plt.show()
                ######

                middle_line_y = int(np.mean(bs))
                sh1 = h1+middle_line_y-window_h//2
                sh2 = h2+middle_line_y-window_h//2
                line_pos.append((sh1+sh2)//2)
                h1+=window_h
                h2+=window_h
            elif len(lines) == 0:
                h1+=window_h
                h2+=window_h
            else:
                h1 += 1
                h2 += 1
        print(line_pos)

        
        # #template match
        # matcher = TempMatcher(self.high_t)
        # matcher.set_image(self.img)
        # matcher.match(1.2)
        # # Moment matching
        # # moment window size
        # high_h = 116
        # high_w = 46
        # low_w = 46
        # low_h = 55
        # # load moment window
        # with open("high.json","r") as file:
        #     high_m = np.array(json.load(file),np.float)
        # with open("low.json","r") as file:
        #     low_m = np.array(json.load(file),np.float)
        # high_map = np.zeros((int((hh-high_h+1)/10)+3,int((ww-high_w+1)/10)+3))
        # # match
        # for h in range(0,hh,10):
        #     for w in range(0,ww,10):
        #         if h + high_h <= self.height and w+ high_w <= self.width:
        #             patch = self.binary[h:h+high_h,w:w+high_w]
        #             high_map[int(h/10),int(w/10)] = np.sqrt(np.sum((np.array(similitudeMoments(patch),np.float) - high_m)**2))
        #     print(h/hh)
        # plt.imshow(np.log(high_map/np.max(high_map)))
        # plt.show()



    def _line_detect(self,patch, theta_max_in_deg=10, num_bin=10, pixel_ratio = 0.5, min_b_distance = 6, min_k_distance = 0.1, check_continuity=False):
        height, width = patch.shape
        acc = dict()
        a_max = math.ceil(math.sqrt(theta_max_in_deg))
        k_list = [0]
        for i in range(1,num_bin+1):
            step = a_max / num_bin
            k_list.append((step * i)**2)
            k_list.append(-(step * i)**2)
        k_list = np.array(k_list)
        k_list *= (math.pi/180)
        k_list = np.tan(k_list)
        k_list = np.round(k_list,4)
        # print(k_list)

        for r in range(height):
            for c in range(width):
                if patch[r,c]:
                    for k in k_list:
                        b = int(r - k*c)
                        if (k,b) in acc:
                            acc[(k,b)] += 1
                        else:
                            acc[(k,b)] = 1

        lines = []
        for key, v in sorted(acc.items(), key=lambda i: -i[1]):
            k,b = key
            #check for length and neibhborhood
            if v > pixel_ratio * width and all(abs(k-ka)>min_k_distance or abs(b-ba)>min_b_distance for ka,ba in lines):
                # check for continuity (in order to remove title lines, cause problem when background substraction is not good)
                if check_continuity:
                    max_continued = 0
                    continued = 0
                    for x in range(width):
                        y = int(k*x+b)
                        if y >=0 and y<height and patch[y,x]:
                            continued += 1
                        else:
                            if continued > max_continued:
                                max_continued = continued
                            continued = 0
                    if max_continued > 0.5 * v:
                        lines.append(key)
                else:
                    lines.append(key)


        return lines
    

    
segmenter = Segmenter("Resources/7.jpeg")
# return the row index where there is sheet line
segmenter.segment()
