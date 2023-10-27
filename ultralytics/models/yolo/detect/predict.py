# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        is_list = isinstance(orig_imgs, list)  # input images are a list, not a torch.Tensor
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if is_list else orig_imgs
            if is_list:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape, ratio_pad=[[self.imgsz[0]/orig_img.shape[0], self.imgsz[1]/orig_img.shape[1]], [16, 16]])
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
    
    def postprocess_xml(self, preds, img, orig_imgs, det_path, img_path):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        # results = []
        is_list = isinstance(orig_imgs, list)  # input images are a list, not a torch.Tensor
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if is_list else orig_imgs
            if is_list:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape, ratio_pad=[[self.imgsz[0]/orig_img.shape[0], self.imgsz[1]/orig_img.shape[1]], [16, 16]])
            # img_path = self.batch[0][i]
            # results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
                save_path = xml_construct(pred, det_path[i], img_path[i])
        return save_path

from lxml.etree import Element, tostring
from lxml.etree import SubElement as subElement
from datetime import datetime

def xml_construct(preds, save_path, filename):
    class_list = ['A220', 'A320/321', 'A330', 'ARJ21', 'Boeing737', 'Boeing787', 'other']

    root = Element('annotation')  # 根节点
 
    source = subElement(root, 'source')
    id = subElement(source, 'id')
    id.text = filename.split('/')[-1][:-4]
    filename_node = subElement(source, 'filename')
    filename_node.text = filename.split('/')[-1]
    origin = subElement(source, 'origin')
    origin.text = 'GF3'

    research = subElement(root, 'research')
    version = subElement(research, 'version')
    version.text = '1.0'
    provider = subElement(research, 'provider')
    provider.text = 'XiDian University'
    author = subElement(research, 'author')
    author.text = 'IPIC'
    pluginname = subElement(research, 'pluginname')
    pluginname.text = 'object detection'
    pluginclass = subElement(research, 'pluginclass')
    pluginclass.text = 'SAR images object detection'
    testperson = subElement(research, 'testperson')
    testperson.text = 'Zhang Yusi'
    time_node = subElement(research, 'time')
    time_node.text = str(datetime.now())

    objects = subElement(root, 'objects')
    for i in range(preds.shape[0]):
        obj = subElement(objects, 'object')

        coordinate = subElement(obj, 'coordinate')
        coordinate.text = 'pixel'
        type_node = subElement(obj, 'type')
        type_node.text = 'rectangle'
        description = subElement(obj, 'description')
        description.text = ''

        possibleresult = subElement(obj, 'possibleresult')
        name_cls = subElement(possibleresult, 'name')
        name_cls.text = class_list[int(preds[i][-1].cpu().numpy())]
        probability = subElement(possibleresult, 'probability')
        probability.text = str(preds[i][-2].cpu().numpy())

        points = subElement(obj, 'points')
        point1 = subElement(points, 'point')
        point1.text = str(preds[i][0].cpu().numpy()) + ',' + str(preds[i][1].cpu().numpy())
        point2 = subElement(points, 'point')
        point2.text = str(preds[i][2].cpu().numpy()) + ',' + str(preds[i][1].cpu().numpy())
        point3 = subElement(points, 'point')
        point3.text = str(preds[i][2].cpu().numpy()) + ',' + str(preds[i][3].cpu().numpy())
        point4 = subElement(points, 'point')
        point4.text = str(preds[i][0].cpu().numpy()) + ',' + str(preds[i][3].cpu().numpy())
        point5 = subElement(points, 'point')
        point5.text = str(preds[i][0].cpu().numpy()) + ',' + str(preds[i][1].cpu().numpy())

 
    xml = tostring(root, pretty_print=True) #将上面设定的一串节点信息导出
    with open(save_path, 'wb') as f: #将节点信息写入到文件路径save_path中
        f.write(xml)
    return save_path
 
#----------调用上面所写的函数进行试验,创建名为test.xml的xml文件----------
# xml_construct('test.xml','test','test','test',width=1600,height=1200,)
