import sys
sys.path.insert(0, './yolov5')  #新添加的目录会优先于其他目录被import检查

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.draw import draw_boxes, bbox_rel, draw_trajectory
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn


def detect(opt, save_img=False):
    out, source, weights, view_img, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)[
        'model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # print('names:',names)

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    save_path = str(Path(out))

    # 设置一个用来存放对象的字典
    object_dic = {}
    count_obj=[]

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, {}, im0s

            # s += '%gx%g ' % img.shape[2:]  # 打印图片的尺寸
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s[names[int(c)]]=int(n)

                bbox_xywh = []   #存放检测框的坐标信息
                confs = []    #存放检测对象的置信度
                classes = []   #存放检测对象的类别名
                # 上述三者的对应关系为列表的索引index

                # Adapt detections to deep sort input format
                # 这里det中存放的数据为【x_c,y_c,w,h,confince,class】
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    classes.append([cls.item()])
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                # print('bbox_xywh',bbox_xywh)

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                class_name=torch.Tensor(classes)

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0, class_name)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]   # 跟踪框的坐标值x1,y1,x2,y2
                    identities = outputs[:, -2]   # 跟踪目标的id
                    # print('bbox_xyxy    ',bbox_xyxy)
                    # print('id    ____',identities)
                    class_name_index=outputs[:,-1]   # 跟踪到的目标的类的索引值（换成类名的话需要names[class_name_index]）
                    draw_boxes(im0, bbox_xyxy, identities, class_name_index)

                    for i, box in enumerate(bbox_xyxy):
                        center = [int((bbox_xyxy[i][0] + bbox_xyxy[i][2]) / 2), int((bbox_xyxy[i][1] + bbox_xyxy[i][3]) / 2),
                                  int(bbox_xyxy[i][2] - bbox_xyxy[i][0]),
                                  int(bbox_xyxy[i][3] - bbox_xyxy[i][1])]

                        #用于记录所有出现的车的数量
                        if names[class_name_index[i]]=='car':
                            if not "%d" % identities[i] in count_obj:
                                count_obj.append("%d" % identities[i])

                        if not "%d" % identities[i] in object_dic:
                            # 创建当前id的字典：key(ID):val{轨迹，丢帧计数器}   当丢帧数超过10帧就删除该对象
                            object_dic["%d" % identities[i]] = {"trace": [], 'traced_frames': 10}
                            object_dic["%d" % identities[i]]["trace"].append(center)
                            object_dic["%d" % identities[i]]["traced_frames"] += 1
                            # 如果有，直接写入
                        else:
                            object_dic["%d" % identities[i]]["trace"].append(center)
                            object_dic["%d" % identities[i]]["traced_frames"] += 1
            else:
                deepsort.increment_ages()

            #绘制跟踪对象的轨迹
            draw_trajectory(object_dic, im0)

            cv2.putText(im0,'{}    {}'.format('car',len(count_obj)),(5,35),cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,0,0), 2)
            y=70
            for key, value in s.items():
                if value != 0:
                    # 为危化品车设置单独的警示颜色
                    if key == 'chemical_vehicle':
                        color=(0,0,255)
                    else:
                        color=(0,255,0)
                    cv2.putText(im0, "{} being tracked: {}".format(key, value), (5, y),cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2)
                    y += 35

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)


    if  save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/x_940.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/input/01.mp4', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    #parser.add_argument('--classes', nargs='+', type=int, default=[0] ,help='filter by class')
    parser.add_argument('--classes', nargs='+', type=int,help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)
