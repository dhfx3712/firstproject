import cv2
import numpy as np


class Annotator:

    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None, example='abc'):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        # use cv2
        self.im = im
        self.lw = line_width  # line width



    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        print (f'p1 :{p1}  ,p2: {p2}')
        # cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)


        if label:
            tf = max(self.lw - 1, 1)  # font thickness
            print (f'tf : {tf}')
            w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)
            # binary, contours, hierarchy = cv2.findContours(self.im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(self.im, contours, -1, color, 4)

            # cv2.imshow('image',self.im)
            # cv2.waitKey(0)





    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)



img = cv2.imread('./1111.jpg')

annotator = Annotator(img, line_width=4)

annotator.box_label([50, 150, 50 + 130, 150 + 150],'www', color=(0, 255, 0))  # borders




# a = np.array([[[10,10], [100,10], [100,100], [10,100]]], dtype = np.int32)
# b = np.array([[[100,100], [200,230], [150,200], [100,220]]], dtype = np.int32)
# print(a.shape)
# im = np.zeros([240, 320], dtype = np.uint8)
# cv2.polylines(im, a, 1, 255)
# cv2.fillPoly(im, b, 255)
# cv2.imshow('im', im)
# cv2.waitKey(0)





def is_in_poly(p, poly):
    """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
    """
    px, py = p
    is_in = False
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1 = corner
        x2, y2 = poly[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in



# point = [(50+50+180)/2,(150+150+300)/2]
# poly = np.array([[[50, 150]], [[280, 150]],[[50, 450]], [[280, 450]]])
poly = np.array([[[50, 150]], [[50, 450]], [[280, 450]],[[280, 150]],]) #点的顺序决定画图的形状,数据样式
# poly = np.array([[50, 150], [50, 450], [280, 450],[280, 150]]) #点的顺序决定画图的形状,数据样式 ,drawContours的[ploy]
# flag = is_in_poly(point,poly)
# print (flag)


# ploy = cv2.polylines(img, [[50, 150, 50 + 130, 150 + 150]], True, color=(98, 9, 11))


cnt = cv2.findContours(img[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(len(cnt),len(cnt[0]),cnt[0][0],np.shape(cnt[0][0]))

# flag = cv2.pointPolygonTest(cnt[0][0], (100, 100), False)

flag = cv2.pointPolygonTest(poly, (100, 100), False)
print (flag)

img1=cv2.drawContours(img,poly,-1,(0,255,0),5)  # img为三通道才能显示轮廓


#https://blog.csdn.net/weixin_44966641/article/details/119039522
img2 = cv2.fillPoly(img, [poly], color=(98, 9, 11))

img3 = cv2.polylines(img, [poly], True, (0, 200, 100), 3)

cv2.imshow('drawimg',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()



''''
判断某点是否在某个多边形（区域）内
https://blog.csdn.net/qq_45612211/article/details/125526328

'''



