from .builder import DATASETS
from .custom import CustomDataset
import os.path as osp
from .pipelines import Compose
import scipy.io as scio
import pickle
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import torch

import mmcv

@DATASETS.register_module()
class PRWDataset(CustomDataset):
    CLASSES = ('person',)
    def __init__(self,
                 data_root,
                 ann_file,
                 img_list_file,
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
        self.dirPath = osp.join(self.data_root, "frames")
        self.img_list_file = img_list_file
        self.personPath = osp.join(self.data_root, "annotations")
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
            if not (self.gallery_file is None or osp.isabs(self.gallery_file)):
                self.gallery_file = osp.join(self.data_root, self.gallery_file)
            if not (self.img_list_file is None or osp.isabs(self.img_list_file)):
                self.img_list_file = osp.join(self.data_root, self.img_list_file)

        self.imageInfo = self.read_mat(self.img_list_file)
        self.integrate_pid()

        # load annotations (and proposals)
        if not self.gallery:
            self.data_infos = self.load_annotations()

        if self.gallery:
            self.data_infos = self.load_gallery_images()

        if self.query:
            self.gallery_classes = self.get_query_info()

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

    def read_mat(self, path):
        info = scio.loadmat(path)
        for k in info.keys():
            if k not in ['__header__', '__version__', '__globals__']:
                return info[k]

    def BoxIndex(self, bbox, anno):
        for i in range(len(anno)):
            if (bbox == anno[i]).sum() == 4:
                return i

    def integrate_pid(self):
        pid_dict = {}
        bbox_dict = {}

        for i in self.imageInfo:
            imname = i[0][0] + '.jpg'
            person_file = osp.join(self.personPath, imname + '.mat')
            persoInfo = self.read_mat(person_file)

            length = len(persoInfo)
            pid_dict[imname] = []
            bbox_dict[imname] = []
            for j in range(length):
                pid = int(persoInfo[j][0])
                pid_dict[imname].append(pid)

                bbox = persoInfo[j][1:].astype(np.float32)
                bbox[2:] += bbox[:2]
                bbox_dict[imname].append(bbox)

        self.pid_dict = pid_dict
        self.bbox_dict = bbox_dict

    def get_query_info(self):
        query_pid_list = []
        for info in self.data_infos:
            ind = info['ann']['pids'][0]
            query_pid_list.append(self.pid_list[ind])
        self.query_pid_list = query_pid_list

        pid2img = {}
        pid2gt = {}
        for img, pids in self.pid_dict.items():
            for i, pid in enumerate(pids):
                if pid == -2:
                    continue
                else:
                    gt = self.bbox_dict[img][i]
                    if pid not in pid2img.keys():
                        pid2img[pid] = [img]
                        pid2gt[pid] = [gt]
                    else:
                        pid2img[pid].append(img)
                        pid2gt[pid].append(gt)

        self.pid2img = pid2img
        self.pid2gt = pid2gt

    def load_ori_anno(self):
        f = open(self.ann_file, 'rb')
        anno = pickle.load(f)
        f.close()

        return anno

    def change_to_coco_style(self, annota):
        annoData = []
        pid = [0]

        for i in range(len(annota)):
            content = annota[i]

            if content['id'] not in pid:
                pid.append(content['id'])

            pid_list = self.pid_dict[content['pic_path']].copy()
            bbox_list = self.bbox_dict[content['pic_path']].copy()

            index = pid_list.index(content['id'])
            pid_list[0], pid_list[index] = pid_list[index], pid_list[0]
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

    def change_to_coco_style_query(self, annota):
        annoData = []
        pid = [0]

        for i in range(len(annota)):
            content = annota[i]
            tmpdict = {}
            anno = {}
            tmpdict['filename'] = content['pic_path']
            size = Image.open(osp.join(self.dirPath, tmpdict['filename'])).size
            tmpdict['width'] = size[0]
            tmpdict['height'] = size[1]

            if content['id'] not in pid:
                pid.append(content['id'])

            pid_list = self.pid_dict[content['pic_path']].copy()
            bbox_list = self.bbox_dict[content['pic_path']].copy()

            index = pid_list.index(content['id'])
            pid_list[0], pid_list[index] = pid_list[index], pid_list[0]
            bbox_list[0], bbox_list[index] = bbox_list[index], bbox_list[0]

            anno['bboxes'] = bbox_list
            anno['labels'] = [0] * len(pid_list)
            anno['pids'] = pid_list
            anno['descriptions'] = content['description']
            anno['bert_token'] = content['token']
            anno['bert_attention_mask'] = content['mask']
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

    def load_annotations(self):
        anno = self.load_ori_anno()

        if self.query:
            annoData = self.change_to_coco_style(anno)
        else:
            annoData = self.change_to_coco_style_query(anno)

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

    def load_gallery_images(self):
        annoData = []
        name2index = []

        for i in self.imageInfo:
            imname = i[0][0] + '.jpg'
            person_file = osp.join(self.personPath, imname + '.mat')
            persoInfo = self.read_mat(person_file)
            bbox = persoInfo[:, 1:].astype(np.float32)
            bbox[:, 2:] += bbox[:, :2]

            tmpdict = {}
            anno = {}
            tmpdict['filename'] = osp.join(self.dirPath, imname)
            size = Image.open(tmpdict['filename']).size
            tmpdict['width'] = size[0]
            tmpdict['height'] = size[1]
            anno['bboxes'] = np.array(bbox, dtype=np.float32)
            anno['labels'] = np.array([0] * len(anno['bboxes']))
            tmpdict['ann'] = anno
            annoData.append(tmpdict)
            name2index.append(imname)

        self.name2index = name2index

        return annoData

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

    def calculate_similarity(self, image_feature, text_feature):
        image_feature = image_feature.view(image_feature.size(0), -1)
        image_feature = image_feature / image_feature.norm(dim=1, keepdim=True)

        text_feature = text_feature.view(text_feature.size(0), -1)
        text_feature = text_feature / text_feature.norm(dim=1, keepdim=True)

        similarity = torch.mm(image_feature, text_feature.t())

        return similarity

    def compute_sim(self, gallery_output, query_output):
        sim_coarse = self.calculate_similarity(gallery_output['gallery_feat_coarse'],
                                               query_output['query_feat_coarse'])
        sim_fine = self.calculate_similarity(gallery_output['gallery_feat_fine'],
                                               query_output['query_feat_fine']).reshape(-1, 1)

        sim = 0.75 * sim_coarse + 0.25 * sim_fine

        return sim

    def evaluate_PS(self,
                    gallery_outputs,
                    query_outputs,
                    gallery_name2index,
                    prog_bar,
                    output_res=True):
        aps = []
        accs = []
        topk = [1, 5, 10]

        for index, pid in enumerate(self.query_pid_list):
            count_gt, count_tp = 0, 0
            name2sim = {}
            name2gt = {}
            name2detbbox = {}
            sims = []
            y_true, y_score = [], []
            imgs, rois = [], []

            gt_img = self.pid2img[pid]
            gt_bbox = self.pid2gt[pid]

            for imgindex, img in enumerate(gallery_name2index):
                if img in gt_img:
                    ind = gt_img.index(img)
                    gt = gt_bbox[ind]
                else:
                    gt = np.array([])
                count_gt += gt.size > 0

                if len(gallery_outputs[imgindex]['gallery_bbox']) == 0:  # this image do no detect anything
                    continue

                sim = self.compute_sim(gallery_outputs[imgindex], query_outputs[index])

                sim = np.array(sim).reshape(-1)

                name2detbbox[img] = gallery_outputs[imgindex]['gallery_bbox']
                name2sim[img] = sim
                name2gt[img] = gt
                sims.extend(list(sim))

            for gallery_imname, sim in name2sim.items():
                gt = name2gt[gallery_imname]
                det = name2detbbox[gallery_imname]
                # assign label for each det
                label = np.zeros(len(sim), dtype=np.int32)
                if gt.size > 0:
                    iou_thresh = 0.5
                    inds = np.argsort(sim)[::-1]
                    sim = sim[inds]
                    det = det[inds]

                    # only set the first matched det as true positive
                    for j, roi in enumerate(det[:, :4]):
                        if self._compute_iou(roi, gt) >= iou_thresh:
                            label[j] = 1
                            count_tp += 1
                            break
                y_true.extend(list(label))
                y_score.extend(list(sim))
                imgs.extend([gallery_imname] * len(sim))
                rois.extend(list(det))

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
        self.query_pid_list = self.query_pid_list[::2]
        prog_bar = mmcv.ProgressBar(len(query_outputs))
        aps1, accs1 = self.evaluate_PS(gallery_outputs, query_outputs[::2], gallery_name2index,
                                       prog_bar, output_res=False)
        aps2, accs2 = self.evaluate_PS(gallery_outputs, query_outputs[1::2], gallery_name2index,
                                       prog_bar, output_res=False)

        aps = (aps1 + aps2) / 2.
        accs = (accs1 + accs2) / 2.

        print("search ranking:")
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
