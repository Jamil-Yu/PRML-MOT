from loguru import logger
import sys
import cv2
import torch
from my_predictor import Predictor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from my_exp import Exp

from visualize import plot_tracking
from tracker.byte_tracker import BYTETracker
from detectors.yolox.tracking_utils.timer import Timer

import argparse
import os
import time
import importlib
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "--detector",default="yolox", help="choose your detector, eg. yolox-s, yolox-m, yolox-l, yolox-x"
    )

    parser.add_argument(
        "--path", default="/home/workspace/ByteTrack/videos/palace.mp4", help="path to images or video"
    )
    
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    parser.add_argument(
        "--vocabulary",
        default='lvis',
        type=str,
        help="vocabulary of the dataset. Now support lvis and coco and custom.",
    )

    parser.add_argument(
        "--thing_classes",
        default=None,
        type=str,
        help="predict only these thing classes. Only valid when vocabulary is custom. Using ',' to split classes.",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.1, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser



def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path)
    # cap = cv2.VideoCapture(1)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, args.path.split("/")[-1])
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 1 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            
            outputs, img_info = predictor.inference(frame, timer)
            tracker.isyolox=(predictor.exp._model == "yolox")
            # print(outputs[0])
            if outputs[0] is not None:
                
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                online_clss=[]
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        online_clss.append(t.cls)
                timer.toc()
                # save results
        
                results.append((frame_id + 1, online_tlwhs, online_ids, online_scores,online_clss))

                if predictor.exp._model == "ovd":
                    thing_classes=predictor.model.model.metadata.thing_classes
                else:
                    thing_classes=None
                
                online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, online_clss,frame_id=frame_id + 1,
                                          fps=1. / timer.average_time,model_type=predictor.exp._model,thing_classes=thing_classes)
                
                
                
                cv2.waitKey(1)
                
            else:
                timer.toc()
                online_im = img_info['raw_img']

            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                print("exit1")
                break
        else:
            print("exit2")
            break
        frame_id += 1
        




def main(exp, args):
    torch.cuda.set_device('cuda:0')

    file_name = exp.output_dir
    os.makedirs(file_name, exist_ok=True)

    exp.vocabulary = args.vocabulary
    

    if args.vocabulary=='custom':
        if args.thing_classes is None:
            raise ValueError("Custom vocabulary must be specified by --thing_classes")
        else:
            exp.thing_classes.clear()
            thing_classes = args.thing_classes.split(',')
            for thing_class in thing_classes:
                exp.thing_classes.append(thing_class)

    if args.save_result:
        vis_folder = os.path.join(file_name, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    
    args.device = "cuda"

    logger.info("Args: {}".format(args))


    
    exp.get_model_from_args(args)

    predictor = Predictor(None, exp)
    current_time = time.localtime()
    imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = Exp()

    main(exp, args)

