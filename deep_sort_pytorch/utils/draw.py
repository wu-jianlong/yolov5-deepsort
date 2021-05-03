import numpy as np
import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, class_name=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        cla_name = int(class_name[i]) if class_name is not None else 0
        #自己的类
        names=['car', 'truck', 'chemical_vehicle', 'bus']
        cla_name=names[cla_name]
        color = compute_color_for_labels(id)
        label = '{}-{}'.format(id,cla_name)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 - t_size[1] - 4), color, -1)

        cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0], 2)
    return img

def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h  #中心框的坐标和宽高

def draw_trajectory(object_dic={},frame=None):  #画轨迹
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
    # 绘制轨迹
    for s in object_dic:
        i = int(s)
        # 这里可以将目标的坐标存起来后面可以继续做目标速度，行驶方向的判断
        # xlist, ylist, wlist, hlist = [], [], [], []

        # 限制轨迹最大长度
        if len(object_dic["%d" % i]["trace"]) > 20:
            for k in range(len(object_dic["%d" % i]["trace"]) - 20):
                del object_dic["%d" % i]["trace"][k]

        # # # 绘制轨迹
        if len(object_dic["%d" % i]["trace"]) > 2:
            for j in range(1, len(object_dic["%d" % i]["trace"]) - 1):
                pot1_x = object_dic["%d" % i]["trace"][j][0]
                pot1_y = object_dic["%d" % i]["trace"][j][1]
                pot2_x = object_dic["%d" % i]["trace"][j + 1][0]
                pot2_y = object_dic["%d" % i]["trace"][j + 1][1]
                # if pot2_x == pot1_x and pot1_y == pot2_y:
                #  del object_dic["%d" % i]
                clr = i % 9  # 轨迹颜色随机
                cv2.line(frame, (pot1_x, pot1_y), (pot2_x, pot2_y), track_colors[clr], 2)

    # 对已经消失的目标予以排除
    for s in object_dic:
        if object_dic["%d" % int(s)]["traced_frames"] > 0:
            object_dic["%d" % int(s)]["traced_frames"] -= 1
    for n in list(object_dic):
        if object_dic["%d" % int(n)]["traced_frames"] == 0:
            del object_dic["%d" % int(n)]

def draw_roi():
    pass
