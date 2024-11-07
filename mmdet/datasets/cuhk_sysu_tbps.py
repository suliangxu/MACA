from .builder import DATASETS
from .custom import CustomDataset
import os.path as osp
from .pipelines import Compose
import scipy.io as scio
import pickle
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
import re
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import torch
import mmcv


@DATASETS.register_module()
class CuhkDataset(CustomDataset):
    CLASSES = ('person',)
    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline,
                 query_test_pipeline=None,
                 classes=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 gallery_file=None,
                 test_mode=False,
                 gallery=False,
                 query=False,
                 filter_empty_gt=True):

        self.ann_file = ann_file
        self.data_root = data_root
        self.dirPath = osp.join(self.data_root, "Image", "SSM")
        self.imagePath = osp.join(self.data_root, "annotation", "Images.mat")
        self.personPath = osp.join(self.data_root, "annotation", "Person.mat")
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.gallery_file = gallery_file
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.gallery = gallery
        self.query = query
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.ann_file is None or osp.isabs(self.ann_file)):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.gallery_file is None
                    or osp.isabs(self.gallery_file)):
                self.gallery_file = osp.join(self.data_root,
                                              self.gallery_file)

        self.imageInfo = scio.loadmat(self.imagePath)['Img'][0]
        self.personInfo = scio.loadmat(self.personPath)['Person'][0]
        self.integrate_pid()

        # load annotations (and proposals)
        if not self.gallery:
            self.data_infos = self.load_annotations(self.ann_file)

        if self.gallery:
            self.data_infos = self.load_gallery_images(self.gallery_file)

        if self.query:
            self.gallery_classes = self.load_gallery_classes(self.gallery_file)

        # filter images too small
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)
        if query_test_pipeline is None:
            self.query_test_pipeline = None
        else:
            self.query_test_pipeline = Compose(query_test_pipeline)

        # query mode
        self.query_mode = False

    def BoxIndex(self, bbox, anno):
        for i in range(len(anno)):
            if (bbox == anno[i]).sum() == 4:
                return i

    def integrate_pid(self):
        pid_dict = {}
        bbox_dict = {}
        for i in self.imageInfo:
            imname = i[0][0]
            length = i[1][0][0]
            pid_dict[imname] = [-1] * length
            bbox_dict[imname] = []
            for j in range(length):
                bbox = i[2]['idlocate'][0][j][0].astype(np.int32)
                bbox[2:] += bbox[:2]
                bbox_dict[imname].append(bbox)

        for i in self.personInfo:
            pid = int(i[0][0][1:])
            appear_times = i[1][0][0]
            for j in range(appear_times):
                imname, bbox = i[2][0]['imname'][j][0], i[2][0]['idlocate'][j][0].astype(np.int32)
                bbox[2:] += bbox[:2]
                index = self.BoxIndex(bbox, bbox_dict[imname])
                pid_dict[imname][index] = pid

        self.pid_dict = pid_dict
        self.bbox_dict = bbox_dict

    def load_ori_anno(self):
        f = open(self.ann_file, 'rb')
        anno = pickle.load(f)
        f.close()

        return anno

    def change_to_coco_style(self, annota):
        annoData = []
        pid = [0]  # pid from 1 to 5532

        for i in range(len(annota)):
            content = annota[i]

            if content['id'] not in pid:
                pid.append(content['id'])

            pid_list = self.pid_dict[content['pic_path']].copy()
            bbox_list = self.bbox_dict[content['pic_path']].copy()

            index = pid_list.index(content['id'])  # traget person pid
            pid_list[0], pid_list[index] = pid_list[index], pid_list[0]  # change the target person into the first item
            bbox_list[0], bbox_list[index] = bbox_list[index], bbox_list[0]

            for k in range(2):
                tmpdict = {}
                anno = {}

                tmpdict['filename'] = content['pic_path']
                size = Image.open(osp.join(self.dirPath, tmpdict['filename'])).size
                tmpdict['width'] = size[0]
                tmpdict['height'] = size[1]

                anno['bboxes'] = bbox_list
                anno['labels'] = [0] * len(pid_list)
                anno['pids'] = pid_list
                anno['descriptions'] = [content['description'][k]]
                anno['bert_token'] = [content['token'][k]]
                anno['bert_attention_mask'] = [content['mask'][k]]
                tmpdict['ann'] = anno

                annoData.append(tmpdict)

        self.pid_list = pid

        return annoData

    def reset_pid(self, pid):
        pids = []
        for i in pid:
            if i in self.pid_list:
                index = self.pid_list.index(i)
            else:
                index = 5555
            pids.append(index)

        return pids

    def load_annotations(self, ann_file):
        anno = self.load_ori_anno()

        annoData = self.change_to_coco_style(anno)

        for i in annoData:
            i['filename'] = osp.join(self.dirPath, i['filename'])
            anno = i['ann']
            anno['bboxes'] = np.array(anno['bboxes'], dtype=np.float32)
            anno['labels'] = np.array(anno['labels'])
            anno['pids'] = self.reset_pid(anno['pids'])
            anno['pids'] = np.array(anno['pids'])
            anno['bert_token'] = np.array(anno['bert_token'])
            anno['bert_attention_mask'] = np.array(anno['bert_attention_mask'])

        return annoData

    def load_gallery_images(self, gallery_file):
        data = scio.loadmat(gallery_file)['pool']
        annoData = []
        image_bbox_dict = {}

        for i in self.imageInfo:
            bbox_list = []
            for j in i[2][0]:
                bbox_ = j[0][0].astype(np.int32)
                bbox_[2:] += bbox_[:2]
                bbox_list.append(bbox_)
            image_bbox_dict[i[0][0]] = np.array(bbox_list)

        name2index = []
        for i in data:
            image_name = i[0][0]
            tmpdict = {}
            anno = {}
            tmpdict['filename'] = osp.join(self.dirPath, image_name)
            size = Image.open(tmpdict['filename']).size
            tmpdict['width'] = size[0]
            tmpdict['height'] = size[1]
            anno['bboxes'] = np.array(image_bbox_dict[image_name], dtype=np.float32)
            anno['labels'] = np.array([0] * len(anno['bboxes']))
            tmpdict['ann'] = anno
            annoData.append(tmpdict)
            name2index.append(image_name)

        self.name2index = name2index

        return annoData

    def load_gallery_classes(self, gallery_file):
        gallery_size = re.match(r'.*(TestG.*).mat', gallery_file).groups()[0]
        data = scio.loadmat(gallery_file)[gallery_size][0]

        query_gallery_list = {}
        for i in data:
            imname = osp.join(self.dirPath, i[0]['imname'][0, 0][0])
            idname = i[0]['idname'][0, 0][0]
            pid = int(re.match(r'p([\d]+)', idname).groups()[0])
            gallery_list = i[1][0]

            tmpdict = {}
            tmpdict['imname'] = imname
            tmpdict['gallery_list'] = gallery_list

            query_gallery_list[pid] = tmpdict

        return query_gallery_list

    def prepare_train_img(self, idx):
        """Get training data and annotations after.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)

        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)

        self.pre_pipeline(results)
        return self.pipeline(results)

    def calculate_similarity(self, image_feature_local, text_feature_local):
        image_feature_local = image_feature_local.view(image_feature_local.size(0), -1)
        image_feature_local = image_feature_local / image_feature_local.norm(dim=1, keepdim=True)

        text_feature_local = text_feature_local.view(text_feature_local.size(0), -1)
        text_feature_local = text_feature_local / text_feature_local.norm(dim=1, keepdim=True)

        similarity = torch.mm(image_feature_local, text_feature_local.t())

        return similarity

    def compute_sim(self, gallery_output, query_output):
        sim_coarse = self.calculate_similarity(gallery_output['gallery_feat_coarse'],
                                               query_output['query_feat_coarse'])

        sim_fine = self.calculate_similarity(gallery_output['gallery_feat_fine'],
                                              query_output['query_feat_fine']).reshape(-1, 1)

        sim = sim_coarse * 0.75 + sim_fine * 0.25

        return sim

    def evaluate_PS(self,
                    gallery_outputs,
                    query_outputs,
                    gallery_name2index,
                    prog_bar,
                    output_res=True):
        gallery_classes = self.gallery_classes
        if self.pid_list[0] == 0:
            self.pid_list.pop(0)


        aps = []
        accs = []
        topk = [1, 5, 10]

        for index in range(len(gallery_classes)):
            pid = self.pid_list[index]
            gallery_image_list = gallery_classes[pid]['gallery_list']

            count_gt, count_tp = 0, 0
            name2sim = {}
            name2gt = {}
            name2detbbox = {}
            sims = []
            y_true, y_score = [], []
            imgs, rois = [], []

            search_detail = []

            for img in gallery_image_list:
                imgname = img['imname'][0]
                imgindex = gallery_name2index.index(imgname)
                gt = img[1][0].astype(np.int32)
                count_gt += gt.size > 0

                if len(gallery_outputs[imgindex]['gallery_bbox']) == 0:  # this image do no detect anything
                    continue

                sim = self.compute_sim(gallery_outputs[imgindex], query_outputs[index])

                sim = np.array(sim).reshape(-1)

                name2detbbox[imgname] = gallery_outputs[imgindex]['gallery_bbox']
                name2sim[imgname] = sim
                name2gt[imgname] = gt
                sims.extend(list(sim))

                for i in gallery_outputs[imgindex]['gallery_bbox']:
                    search_detail.append([imgname, i])

            for gallery_imname, sim in name2sim.items():
                gt = name2gt[gallery_imname]
                det = name2detbbox[gallery_imname]
                # assign label for each det
                label = np.zeros(len(sim), dtype=np.int32)
                if gt.size > 0:  # this image contains the query person
                    w, h = gt[2], gt[3]
                    gt[2:] += gt[:2]
                    iou_thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                    inds = np.argsort(sim)[::-1]
                    sim = sim[inds]
                    det = det[inds]

                    # only set the first matched det as true positive
                    for j, roi in enumerate(det[:, :4]):
                        if self._compute_iou(roi, gt) >= iou_thresh:
                            label[j] = 1
                            count_tp += 1
                            break
                y_true.extend(list(label))  # label: 1 for gt and 0 for no gt
                y_score.extend(list(sim))  # similarity
                imgs.extend([gallery_imname] * len(sim))  # image name
                rois.extend(list(det))  # bbox, [5]

            # 3. Compute AP for this query (need to scale by recall rate)
            y_score = np.asarray(y_score)
            y_true = np.asarray(y_true)
            assert count_tp <= count_gt
            recall_rate = count_tp * 1.0 / count_gt
            ap = 0 if count_tp == 0 else average_precision_score(y_true, y_score) * recall_rate
            aps.append(ap)
            inds = np.argsort(y_score)[::-1]

            y_score = y_score[inds]
            y_true = y_true[inds]

            accs.append([min(1, sum(y_true[:k])) for k in topk])
            prog_bar.update()

            # imgs = np.array(imgs)[inds]
            # rois = np.array(rois)[inds]
            # for k in range(10):
            #     self.draw_new(y_score[k], imgs[k], rois[k], index)

        aps = np.mean(aps)
        accs = np.mean(accs, axis=0)

        if output_res:
            print("search ranking:")
            print("  mAP = {:.2%}".format(aps))
            for i, k in enumerate(topk):
                print("  top-{:2d} = {:.2%}".format(k, accs[i]))

        return aps, accs

    def evaluate_PS_double_query(self,
                                 gallery_outputs,
                                 query_outputs,
                                 gallery_name2index):
        prog_bar = mmcv.ProgressBar(len(query_outputs))
        aps1, accs1 = self.evaluate_PS(gallery_outputs, query_outputs[::2], gallery_name2index,
                                       prog_bar, output_res=False)
        aps2, accs2 = self.evaluate_PS(gallery_outputs, query_outputs[1::2], gallery_name2index,
                                       prog_bar, output_res=False)

        aps = (aps1 + aps2) / 2.
        accs = (accs1 + accs2) / 2.

        print("\nsearch ranking:")
        print("  mAP = {:.2%}".format(aps))
        for i, k in enumerate([1, 5, 10]):
            print("  top-{:2d} = {:.2%}".format(k, accs[i]))

        return aps, accs

    def _compute_iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter * 1.0 / union

    def draw_new(self, sim, img, roi, index):
        text = self.data_infos[index * 2]['ann']['descriptions'][0]
        img_path = osp.join(self.dirPath, img)

        img = Image.open(img_path)
        draw = PIL.ImageDraw.Draw(img)

        color_for_draw = tuple(np.random.randint(0, 255, size=[3]))
        a, b, c, d, _ = roi
        pos = (a, b - 30)

        draw.rectangle([a, b, c, d], outline=color_for_draw, width=2)
        font = ImageFont.truetype("NotoSansCJK-Bold.ttc", 15, encoding='utf-8')

        draw.text(pos, str(sim), color_for_draw, font=font)

        plt.figure(figsize=(15, 12))
        # print(text)
        plt.title(text, fontsize=15, wrap=True)
        plt.imshow(img)
        plt.show()
