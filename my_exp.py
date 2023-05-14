# encoding: utf-8
import os
import numpy as np
import cv2
import torch
import torchvision
# from detectors.yolox.data.data_augment import preproc
# from detectors.yolox.utils import postprocess
import clip
from detectors.ultralytics import YOLO
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor

from detectors.object_centric_ovd.ovd.config import add_ovd_config
from detectors.object_centric_ovd.ovd.modeling.utils import reset_cls_test

# import some common detectron2 utilities
from detectron2.config import get_cfg

model_sets=[
    'yolov8',
    'yolox',
    'ovd',
]#detector models to choose from

def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

class BaseDetector(object):
    def __init__(self, model_name, model_path, device):
        self.model_name=model_name
        self.model_path=model_path
        self.device=device
    def forward(self,image):
        pass

class YOLOXDetector(BaseDetector):
    
    def __init__(self, model_name, model_path, device):
        super().__init__(model_name, model_path, device)
        self.depth = 1.33
        self.width = 1.25
        self.num_classes = 1
        from detectors.yolox.models import YOLOPAFPN, YOLOX, YOLOXHead
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels)
            self.model = YOLOX(backbone, head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        ckpt = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"],strict=False)
        self.test_size=(896, 1600)
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.num_classes=80
        self.confthre=0.001
        self.nmsthre=0.65
    @torch.no_grad()
    def forward(self,img):
        #img: h,w,c, numpy
        self.model.eval()
        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        # if self.device == "cuda":
        #    img = img.cuda()
        # #    img = img.half()
        outputs = self.model(img)
        outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )
        outputs = [outputs[0].cpu().numpy()]
        return outputs
    
class YOLOV8Detector(BaseDetector):

    def __init__(self, model_name, model_path, device):
        super().__init__(model_name, model_path, device)
        self.model = YOLO(self.model_path)
    @torch.no_grad()
    def forward(self, img):
        boxes=self.model(img)[0].boxes
        xyxy=boxes.xyxy
        conf=boxes.conf.unsqueeze(-1)
        cls=boxes.cls.unsqueeze(-1)
        outputs_wicls=torch.cat([xyxy,conf,cls],1)
        outputs=[outputs_wicls.cpu().numpy()]
        return outputs
    
class OVDetector(BaseDetector):
    def __init__(self, model_name,vocabulary,thing_classes, model_path, device,config_file,zero_shot_weight_path,clip_path):
        super().__init__(model_name, model_path, device)
        self.vocabulary=vocabulary
        self.thing_classes=thing_classes
        self.BUILDIN_CLASSIFIER = {
            'lvis': 'model_zoo/lvis_v1_clip_a+photo+cname.npy',
            'coco': 'model_zoo/coco_clip_a+photo+cname.npy',
        }

        self.BUILDIN_METADATA_PATH = {
            'lvis': 'lvis_v1_val',
            'coco': 'coco_2017_val',
        }   
        cfg = get_cfg()
        add_ovd_config(cfg)
        cfg.merge_from_file(config_file)#configs/lvis/LVIS_OVD_RKD_PIS_WeightTransfer_8x.yaml")
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = zero_shot_weight_path
        cfg.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS = False
        cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        self.model = DefaultPredictor(cfg)
        if self.vocabulary=='custom':
                    
            if MetadataCatalog.__contains__("new_classes"):
                MetadataCatalog.remove("new_classes")
            metadata = MetadataCatalog.get("new_classes")
            
            metadata.thing_classes = self.thing_classes
            classifier = self.dump_clip_embeddings(metadata.thing_classes,clip_path=clip_path)
            num_classes = len(metadata.thing_classes)
                    
            reset_cls_test(self.model.model, classifier, num_classes)
            self.model.metadata=metadata
            
                    
        elif self.vocabulary=='lvis' or self.vocabulary=='coco':
            metadata = MetadataCatalog.get(self.BUILDIN_METADATA_PATH[self.vocabulary])
            classifier = self.BUILDIN_CLASSIFIER[self.vocabulary]
            num_classes = len(metadata.thing_classes)
            self.model.metadata=metadata
            reset_cls_test(self.model.model, classifier, num_classes)
        else:
            assert False, "vocabulary not in {}".format(['custom','lvis','coco'])
    @torch.no_grad()
    def forward(self, img):
        instances=self.model(img)['instances']
        boxes=instances.pred_boxes.tensor
        conf=instances.scores.unsqueeze(-1)
        cls=instances.pred_classes.unsqueeze(-1)
        outputs_wicls=torch.cat([boxes,conf,cls],1)
        outputs=[outputs_wicls.cpu().numpy()]
        return outputs
    
    def dump_clip_embeddings(self,vocabulary, prompt='a photo of',clip_path=None):
        sentences = ['a photo of a {}'.format(x) for x in vocabulary]
        
        print('Loading CLIP')
        device = "cuda"
        print(clip)
        model, preprocess = clip.load(clip_path, device=device)
        text = clip.tokenize(sentences).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text).float().T

        return text_features






class Exp():
    def __init__(self):
        self.output_dir='output'
        self.optional_detector=None
        self._model="yolox"
        self.vocabulary='lvis'
        self.thing_classes=[]
        self.test_size=(896, 1600)
    
    def get_model_from_args(self,args):
        if args.detector in model_sets:
            self._model=args.detector
            if args.detector=='yolov8':
                self.optional_detector=YOLOV8Detector(model_name='yolov8',model_path='model_zoo/yolov8m.pt',device='cuda')
            elif args.detector=='ovd':
                self.optional_detector=OVDetector(model_name='ovd',vocabulary=self.vocabulary,thing_classes=self.thing_classes,
                                                  model_path='model_zoo/lvis_ovd_rkd_pis_weighttransfer_8x.pth',
                                                  device='cuda',
                                                  config_file='model_zoo/LVIS_OVD_RKD_PIS_WeightTransfer_8x.yaml',
                                                  zero_shot_weight_path='model_zoo/lvis_v1_clip_a+photo+cname.npy',
                                                  clip_path='model_zoo/ViT-B-32.pt')
            elif args.detector=='yolox':
                self.optional_detector=YOLOXDetector(model_name='yolox',model_path='model_zoo/bytetrack_x_mot20.tar',device='cuda')
        else:
            assert False, "detector model not in {}".format(model_sets)



