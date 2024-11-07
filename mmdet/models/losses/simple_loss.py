import torch
from torch import nn
import torch.nn.functional as F
from ..builder import LOSSES
import numpy as np


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(classifier, self).__init__()

        self.block = nn.Linear(input_dim, output_dim)
        self.block.apply(weights_init_classifier)

    def forward(self, x):
        x = self.block(x)
        return x


class Id_Loss(nn.Module):
    def __init__(self, class_num, feature_length):
        super(Id_Loss, self).__init__()

        self.W = classifier(feature_length, class_num)

    def forward(self, image_embedding, text_embedding, label):
        label = label.view(label.size(0))

        criterion = nn.CrossEntropyLoss(reduction='mean')

        score_i2t = self.W(image_embedding[:, 0, :])
        score_t2i = self.W(text_embedding[:, 0, :])

        Li2t = criterion(score_i2t, label)
        Lt2i = criterion(score_t2i, label)

        loss = (Li2t + Lt2i) / 2

        return loss


class CRLoss(nn.Module):
    def __init__(self, margin):
        super(CRLoss, self).__init__()

        self.margin = np.array([margin])

    def semi_hard_negative(self, loss, margin):
        negative_index = np.where(np.logical_and(loss < margin, loss > 0))[0]
        return np.random.choice(negative_index) if len(negative_index) > 0 else None

    def calculate_similarity(self, image_embedding, text_embedding):
        image_embedding = image_embedding.view(image_embedding.size(0), -1)
        text_embedding = text_embedding.view(text_embedding.size(0), -1)
        image_embedding_norm = image_embedding / (image_embedding.norm(dim=1, keepdim=True) + 1e-8)
        text_embedding_norm = text_embedding / (text_embedding.norm(dim=1, keepdim=True) + 1e-8)
        similarity = torch.mm(image_embedding_norm, text_embedding_norm.t())

        return similarity

    def get_triplets(self, similarity, labels, margin):
        device = similarity.device
        similarity = similarity.cpu().data.numpy()

        labels = labels.cpu().data.numpy()
        triplets = []

        for idx, label in enumerate(labels):  # same class calculate together
            negative = np.where(labels != label)[0]

            ap_sim = similarity[idx, idx]

            loss = similarity[idx, negative] - ap_sim + margin[idx]

            negetive_index = self.semi_hard_negative(loss, margin[idx])

            if negetive_index is not None:
                triplets.append([idx, idx, negative[negetive_index]])

        if len(triplets) == 0:
            triplets.append([idx, idx, negative[0]])

        triplets = torch.LongTensor(np.array(triplets))

        return_margin = torch.FloatTensor(np.array(margin[triplets[:, 0]])).to(device)

        return triplets, return_margin

    def calculate_loss(self, similarity, label, margin):
        image_triplets, img_margin = self.get_triplets(similarity, label, margin)
        text_triplets, txt_margin = self.get_triplets(similarity.t(), label, margin)

        image_anchor_loss = F.relu(img_margin
                                   - similarity[image_triplets[:, 0], image_triplets[:, 1]]
                                   + similarity[image_triplets[:, 0], image_triplets[:, 2]])

        similarity = similarity.t()
        text_anchor_loss = F.relu(txt_margin
                                  - similarity[text_triplets[:, 0], text_triplets[:, 1]]
                                  + similarity[text_triplets[:, 0], text_triplets[:, 2]])

        loss = torch.sum(image_anchor_loss) + torch.sum(text_anchor_loss)

        return loss

    def forward(self, img, txt, labels):
        similarity = self.calculate_similarity(img, txt)

        cr_loss = self.calculate_loss(similarity, labels, self.margin.repeat(len(labels)))

        return cr_loss


@LOSSES.register_module()
class SimpleLoss(nn.Module):
    def __init__(self, num_class, feat_coarse, feat_fine, margin=0.5, tmp=0.07, lambda_=0.75):
        super(SimpleLoss, self).__init__()

        self.id_loss_fun_coarse = Id_Loss(num_class, feat_coarse)
        self.id_loss_fun_fine = Id_Loss(num_class, feat_fine)

        self.cr_loss = CRLoss(margin)

        self.temp = nn.Parameter(torch.ones([]) * tmp)
        self.lambda_ = lambda_

    def compute_itc(self, visual_feats, textual_feats):
        visual_feats = visual_feats / visual_feats.norm(dim=1, keepdim=True)
        textual_feats = textual_feats / textual_feats.norm(dim=1, keepdim=True)

        sim_targets_t2i = torch.zeros((len(textual_feats), len(visual_feats)), device=visual_feats.device)
        fac = len(visual_feats) // len(textual_feats)
        for i in range(len(textual_feats)):
            pos = fac*i
            sim_targets_t2i[i][pos] = 1

        sim_i2t = visual_feats @ textual_feats.t() / self.temp
        sim_t2i = textual_feats @ visual_feats.t() / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets_t2i.t(), dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets_t2i, dim=1).mean()

        loss_itc = (loss_i2t + loss_t2i) / 2

        return loss_itc

    def compute_itc_att(self, visual_feats, att_feats, labels):
        visual_feats = visual_feats / visual_feats.norm(dim=1, keepdim=True)
        att_feats = att_feats / att_feats.norm(dim=1, keepdim=True)

        sim_i2t = visual_feats @ att_feats.t() / self.temp
        sim_t2i = att_feats @ visual_feats.t() / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * labels.t(), dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * labels, dim=1).mean()

        loss_itc = (loss_i2t + loss_t2i) / 2

        return loss_itc

    def forward(self,
                image_coarse,
                text_coarse,
                image_fine,
                text_fine,
                att_local,
                att_labels,
                pid
                ):
        p_ind = [len(image_coarse) // len(text_coarse) * i for i in range(len(text_coarse))]
        label = torch.cat(pid)

        id_loss_coarse = self.id_loss_fun_coarse(image_coarse[p_ind], text_coarse, label)
        id_loss_fine = self.id_loss_fun_fine(image_fine[:len(text_coarse)].unsqueeze(1), text_fine[:len(text_coarse)].unsqueeze(1), label)
        id_loss = self.lambda_ * id_loss_coarse + (1 - self.lambda_) * id_loss_fine

        cr_loss_coarse = self.cr_loss(image_coarse[p_ind], text_coarse, label)
        cr_loss_fine = self.cr_loss(image_fine.unsqueeze(1), text_fine.unsqueeze(1), label)
        cr_loss = self.lambda_ * cr_loss_coarse + (1 - self.lambda_) * cr_loss_fine

        loss_itc_coarse = self.compute_itc(image_coarse[:, 0, :], text_coarse[:, 0, :])
        loss_itc_fine = self.compute_itc_att(image_fine, att_local, att_labels)
        loss_itc = self.lambda_ * loss_itc_coarse + (1 - self.lambda_) * loss_itc_fine

        return id_loss, cr_loss, loss_itc
