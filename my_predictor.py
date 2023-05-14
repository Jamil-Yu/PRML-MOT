import cv2
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Predictor(object):
    def __init__(
        self,
        model=None,
        exp=None,
    ):
        self.exp=exp
        if model is not None:
            self.model = model
        else:
            self.model=exp.optional_detector

    def inference(self, img, timer):
        img_info = {"id": 0}
        img_info["file_name"] = None
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        
        
        
        with torch.no_grad():
            timer.tic()
            
            outputs=self.model.forward(img)
        return outputs, img_info#返回的是detector模型的输出和图像信息
# import argparse   
# def make_parser():
#     parser = argparse.ArgumentParser("Test")
#     parser.add_argument(
#         "--detector",
#         default="ovd",
#     )
#     parser.add_argument(
#         "--path",
#         default=None,
#     )
#     parser.add_argument(
#         "--vocabulary",
#         default='custom',
#         type=str,
#         help="vocabulary of the dataset. Now support lvis and coco and custom.",
#     )

#     parser.add_argument(
#         "--thing_classes",
#         default="bicycle",
#         type=str,
#         help="predict only these thing classes. Only valid when vocabulary is custom. Using ',' to split classes.",
#     )
#     return parser
# def test_predictor():
#     args = make_parser().parse_args()
#     exp = get_exp("/home/workspace/ByteTrack/exps/my_exp.py", None)
#     exp._model="ovd"
#     exp.vocabulary = args.vocabulary
#     exp.thing_classes.clear()
#     thing_classes = args.thing_classes.split(',')
#     for thing_class in thing_classes:
#         exp.thing_classes.append(thing_class)
#     exp.get_model_from_args(args)
#     image=cv2.imread("/home/workspace/ByteTrack/store.jpg")
#     predictor = Predictor(None, exp, None, None, "gpu", False)
#     #print(predictor.model.metadata)
#     #print(predictor.inference(image, Timer())[0]['instances'].pred_boxes)

# if __name__ == "__main__":
#     test_predictor()