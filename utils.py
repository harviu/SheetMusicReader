# utilities functions
from matplotlib import pyplot as plt
import numpy as np
import cv2
import math
from PIL import Image

SIGMA = 1e-15

def merge_boxes(boxes, threshold):
    filtered_boxes = []
    while len(boxes) > 0:
        r = boxes.pop(0)
        boxes.sort(key=lambda box: box.distance(r))
        merged = True
        while (merged):
            merged = False
            i = 0
            for _ in range(len(boxes)):
                if r.overlap(boxes[i]) > threshold or boxes[i].overlap(r) > threshold:
                    r = r.merge(boxes.pop(i))
                    merged = True
                elif boxes[i].distance(r) > r.w / 2 + boxes[i].w / 2:
                    break
                else:
                    i += 1
        filtered_boxes.append(r)
    return filtered_boxes


def collect_info(output, target_img, box_color, box_thickness, label, text, text_color, info_boxes, duration = 0, isNote = False, staff = None):
    for box in info_boxes:
        box.draw(target_img, box_color, box_thickness)
        font = cv2.FONT_HERSHEY_DUPLEX
        textsize = cv2.getTextSize(text, font, fontScale=0.7, thickness=1)[0]
        x = int(box.getCorner()[0] - (textsize[0] // 2))
        y = int(box.getCorner()[1] + box.getHeight() + 20)
        cv2.putText(target_img, text, (x, y), font, fontScale=0.7, color=text_color, thickness=1)
        if isNote:
            pitch = staff.getPitch(round(box.getCenter()[1]))
            output.append(Primitive(label, duration, box, pitch))
        else:
            output.append(Primitive(label, duration, box))

def locate_templates(img, templates, start, stop, threshold):
    locations, scale = match(img, templates, start, stop, threshold)
    img_locations = []
    for i in range(len(templates)):
        w, h = templates[i].shape[::-1]
        w *= scale
        h *= scale
        # for operation preference, the * is the last compared with iteration [i] and [::-1]
        # * expension is for the purpose of zip 
        img_locations.append([BoundingBox(pt[0], pt[1], w, h) for pt in zip(*locations[i][::-1])])
    return img_locations

    
def match(img, templates, start_percent, stop_percent, threshold):
    # print(img.shape)
    # print(templates)
    img_width, img_height = img.shape[::-1]
    best_location_count = -1
    best_locations = []
    best_scale = 1

    # plt.axis([0, 2, 0, 1])
    # plt.show(block=False)

    x = []
    y = []
    for scale in [i/100.0 for i in range(start_percent, stop_percent + 1, 3)]:
        locations = []
        location_count = 0

        for template in templates:
            if (scale*template.shape[0] > img.shape[0] or scale*template.shape[1] > img.shape[1]):
                continue

            template = cv2.resize(template, None,
                fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            '''
                If input image is of size (WxH) and template image is of size (wxh), output image will 
                have a size of (W-w+1, H-h+1). Once you got the result, 
                you can use cv2.minMaxLoc() function to find where is the maximum/minimum value. 
            '''
            # print(type(result))
            result = np.where(result >= threshold)
            # If only condition is given, return the tuple condition.nonzero(), the indices where condition is True.
            location_count += len(result[0])
            locations += [result]

        # print("scale: {0}, hits: {1}".format(scale, location_count))
        x.append(location_count)
        y.append(scale)
        # plt.plot(y, x)
        # plt.pause(0.00001)
        if (location_count > best_location_count):
            best_location_count = location_count
            best_locations = locations
            best_scale = scale
            # plt.axis([0, 2, 0, best_location_count])
        elif (location_count < best_location_count):
            pass
    # plt.close()
    # best locations is the list of tuples containing the locations of match pt above the threshold
    return best_locations, best_scale


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

class Primitive(object):
    def __init__(self, primitive, duration, box, pitch=-1):
        self.pitch = pitch
        self.duration = duration
        self.primitive = primitive
        self.box = box

    def setPitch(self, pitch):
        self.pitch = pitch

    def setDuration(self, duration):
        self.duration = duration
    
    def getPrimitive(self):
        return self.primitive

    def getPitch(self):
        return self.pitch

    def getDuration(self):
        return self.duration

    def getBox(self):
        return self.box

class Staff(object):
    def __init__(self, staff_matrix, line_spacing, staff_img, staff_box=None, line_width=2, clef="treble", time_signature="44", instrument=-1):
        self.clef = clef
        self.time_signature = time_signature
        self.instrument = instrument
        # initialize the starting pos of each line, reference to half distance between two staves
        self.line_one = staff_matrix[0]
        self.line_two = staff_matrix[1]
        self.line_three = staff_matrix[2]
        self.line_four = staff_matrix[3]
        self.line_five = staff_matrix[4]
        self.staff_box = staff_box
        self.img = staff_img
        self.bars = []
        self.line_width = line_width
        self.line_spacing = line_spacing

    def setClef(self, clef):
        self.clef = clef

    def setTimeSignature(self, time):
        self.time_signature = time

    def setInstrument(self, instrument):
        self.instrument = instrument

    def addBar(self, bar):
        self.bars.append(bar)

    def getClef(self):
        return self.clef

    def getTimeSignature(self):
        return self.time_signature

    def getBox(self):
        return self.staff_box

    def getImage(self):
        return self.img

    def getLineWidth(self):
        return self.line_width

    def getLineSpacing(self):
        return self.line_spacing

    def getBars(self):
        return self.bars

    def getPitch(self, note_center_y):
        clef_info = {
            "treble": [("F5", "E5", "D5", "C5", "B4", "A4", "G4", "F4", "E4"), (5,3), (4,2)],
            "bass": [("A3", "G3", "F3", "E3", "D3", "C3", "B2", "A2", "G2"), (3,5), (2,4)]
        }
        note_names = ["C", "D", "E", "F", "G", "A", "B"]

        #print("[getPitch] Using {} clef".format(self.clef))

        # Check within staff first
        if (note_center_y in self.line_one):
            return clef_info[self.clef][0][0]
        elif (note_center_y in list(range(self.line_one[-1] + 1, self.line_two[0]))):
            return clef_info[self.clef][0][1]
        elif (note_center_y in self.line_two):
            return clef_info[self.clef][0][2]
        elif (note_center_y in list(range(self.line_two[-1] + 1, self.line_three[0]))):
            return clef_info[self.clef][0][3]
        elif (note_center_y in self.line_three):
            return clef_info[self.clef][0][4]
        elif (note_center_y in list(range(self.line_three[-1] + 1, self.line_four[0]))):
            return clef_info[self.clef][0][5]
        elif (note_center_y in self.line_four):
            return clef_info[self.clef][0][6]
        elif (note_center_y in list(range(self.line_four[-1] + 1, self.line_five[0]))):
            return clef_info[self.clef][0][7]
        elif (note_center_y in self.line_five):
            return clef_info[self.clef][0][8]
        else:
            # print("[getPitch] Note was not within staff")
            if (note_center_y < self.line_one[0]):
                # print("[getPitch] Note above staff ")
                # Check above staff
                line_below = self.line_one
                current_line = [int(pixel - self.line_spacing) for pixel in self.line_one] # Go to next line above
                octave = clef_info[self.clef][1][0]  # The octave number at line one
                note_index = clef_info[self.clef][1][1]  # Line one's pitch has this index in note_names

                while (current_line[0] > 0):
                    if (note_center_y in current_line):
                        # Grab note two places above
                        octave = octave + 1 if (note_index + 2 >= 7) else octave
                        note_index = (note_index + 2) % 7
                        return note_names[note_index] + str(octave)
                    elif (note_center_y in range(current_line[-1] + 1, line_below[0])):
                        # Grab note one place above
                        octave = octave + 1 if (note_index + 1 >= 7) else octave
                        note_index = (note_index + 1) % 7
                        return note_names[note_index] + str(octave)
                    else:
                        # Check next line above
                        octave = octave + 1 if (note_index + 2 >= 7) else octave
                        note_index = (note_index + 2) % 7
                        line_below = current_line.copy()
                        current_line = [int(pixel - self.line_spacing) for pixel in current_line]

                assert False, "[ERROR] Note was above staff, but not found"
            elif (note_center_y > self.line_five[-1]):
                # print("[getPitch] Note below staff ")
                # Check below staff
                line_above = self.line_five
                current_line = [int(pixel + self.line_spacing) for pixel in self.line_five]  # Go to next line above
                octave = clef_info[self.clef][2][0]  # The octave number at line five
                note_index = clef_info[self.clef][2][1]  # Line five's pitch has this index in note_names

                while (current_line[-1] < self.img.shape[0]):
                    if (note_center_y in current_line):
                        # Grab note two places above
                        octave = octave - 1 if (note_index - 2 <= 7) else octave
                        note_index = (note_index - 2) % 7
                        return note_names[note_index] + str(octave)
                    elif (note_center_y in range(line_above[-1] + 1, current_line[0])):
                        # Grab note one place above
                        octave = octave - 1 if (note_index - 1 >= 7) else octave
                        note_index = (note_index - 1) % 7
                        return note_names[note_index] + str(octave)
                    else:
                        # Check next line above
                        octave = octave - 1 if (note_index - 2 <= 7) else octave
                        note_index = (note_index - 2) % 7
                        line_above = current_line.copy()
                        current_line = [int(pixel + self.line_spacing) for pixel in current_line]
                assert False, "[ERROR] Note was below staff, but not found"
            else:
                # Should not get here
                assert False, "[ERROR] Note was neither, within, above or below staff"

class Bar(object):
    def __init__(self, key_signature="c"):
        self.primitives = []

    def addPrimitive(self, primitive):
        self.primitives.append(primitive)

    def getPrimitives(self):
        return self.primitives
    

class BoundingBox(object):
    def __init__(self, x, y, w, h):
        self.x = x;
        self.y = y;
        self.w = w;
        self.h = h;
        self.middle = self.x + self.w/2, self.y + self.h/2
        self.area = self.w * self.h

    def overlap(self, other):
        overlap_x = max(0, min(self.x + self.w, other.x + other.w) - max(self.x, other.x));
        overlap_y = max(0, min(self.y + self.h, other.y + other.h) - max(self.y, other.y));
        overlap_area = overlap_x * overlap_y
        return overlap_area / self.area

    def distance(self, other):
        dx = self.middle[0] - other.middle[0]
        dy = self.middle[1] - other.middle[1]
        return math.sqrt(dx*dx + dy*dy)

    def merge(self, other):
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        w = max(self.x + self.w, other.x + other.w) - x
        h = max(self.y + self.h, other.y + other.h) - y
        return BoundingBox(x, y, w, h)

    def draw(self, img, color, thickness):
        pos = ((int)(self.x), (int)(self.y))
        size = ((int)(self.x + self.w), (int)(self.y + self.h))
        cv2.rectangle(img, pos, size, color, thickness)

    def getCorner(self):
        return self.x, self.y

    def getWidth(self):
        return self.w

    def getHeight(self):
        return self.h

    def getCenter(self):
        return self.middle
