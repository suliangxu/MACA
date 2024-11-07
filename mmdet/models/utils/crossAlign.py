import torch
import torch.nn as nn
import transformers as ppb

from mmcv.runner.base_module import BaseModule
from mmcv.utils import Registry

import re
import nltk
from .conv_bottle_neck import ResNet_text_50, Bottleneck, conv

CROSSALIGN = Registry('cross_align')

class Attribute_Align(nn.Module):
    def __init__(self,
                 visual_embeddings,
                 textual_embeddings,
                 fine_embeddings,
                 memory_size,
                 feat_map,
                 att_part=5):
        super(Attribute_Align, self).__init__()

        self.instance_branch = nn.Sequential(
            Bottleneck(visual_embeddings, 1024, width=512, conv3=True,
                       downsample=nn.Conv2d(visual_embeddings, 1024, kernel_size=1, bias=False)),
            Bottleneck(1024, fine_embeddings, width=1024, conv3=True,
                       downsample=nn.Conv2d(1024, fine_embeddings, kernel_size=1, bias=False))
        )
        self.attribute_branch = nn.Sequential(
            Bottleneck(textual_embeddings, 1024, width=512, conv3=False,
                       downsample=nn.Conv2d(textual_embeddings, 1024, kernel_size=1, bias=False)),
            Bottleneck(1024, fine_embeddings, width=1024, conv3=False,
                       downsample=nn.Conv2d(1024, fine_embeddings, kernel_size=1, bias=False))
        )
        self.conv_fine = conv(fine_embeddings, fine_embeddings)
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.text_fc = nn.Linear(fine_embeddings, fine_embeddings)

        self.memory_size = 0
        self.register_buffer('image_memory', torch.zeros(memory_size, visual_embeddings, feat_map[0], feat_map[1]))
        self.register_buffer('att_memory', torch.zeros(memory_size, att_part, 5, textual_embeddings))
        self.register_buffer('att_mask', torch.zeros(memory_size, att_part, 5, dtype=torch.int))
        self.memory_header = 0

    def update_memory(self, image_feat, att_feats, att_masks):
        for i in range(len(image_feat)):
            self.image_memory[self.memory_header] = image_feat[i].detach().clone()
            self.att_memory[self.memory_header] = att_feats[i].detach().clone()
            self.att_mask[self.memory_header] = att_masks[i].detach().clone()

            self.memory_header = (self.memory_header + 1) % self.image_memory.size(0)

        self.memory_size += len(image_feat)
        if self.memory_size > self.image_memory.size(0):
            self.memory_size = self.image_memory.size(0)

    def effect_memory(self, image_feature, att_feature, att_mask_now, memory=True):
        if not memory or self.memory_size == 0:
            img_feat = image_feature
            att_feat = att_feature
            att_masks = att_mask_now
        else:
            img_feat = torch.cat((image_feature, self.image_memory[:self.memory_size]), dim=0)
            att_feat = torch.cat((att_feature, self.att_memory[:self.memory_size]), dim=0)
            att_masks = torch.cat((att_mask_now, self.att_mask[:self.memory_size]), dim=0)

        att_part = att_masks.sum(dim=-1).bool()

        # classes
        eff_att_part = att_masks.max(dim=-1)[0]
        eff_att_part_ind = eff_att_part[att_part]
        labels = torch.zeros(len(eff_att_part_ind), len(img_feat), device=img_feat.device)
        att_per_text = []
        count = 0
        for i, part in enumerate(att_part):
            tmp = []
            for k in part:
                if k:
                    tmp.append(count)
                    count += 1
            labels[tmp, i] = 1
            att_per_text.append(tmp)

        if memory:
            self.update_memory(image_feature, att_feature, att_mask_now)

        return img_feat, att_feat, labels, att_masks

    def forward(self, image_feature, att_info, mode=0):
        if mode == 1:
            att_feature = att_info['att_feat_local']
            att_mask = att_info['att_feat_mask']

            image_feat, att_feat, labels, att_per_text = self.effect_memory(image_feature[:, 0],
                                                                             att_feature, att_mask, memory=True)

            image_feat = self.instance_branch(image_feat)
            image_feat = self.conv_fine(image_feat)
            image_feat = self.global_maxpool(image_feat).squeeze(-1).squeeze(-1)

            bs, part, token, emb = att_feat.shape
            att_feat = att_feat.reshape(bs * part, token, emb).permute(0, 2, 1)
            att_feat = self.attribute_branch(att_feat.unsqueeze(-1))
            att_feat = self.conv_fine(att_feat).permute(0, 2, 1).reshape(bs, part, token, -1)

            att_local = []
            text_feat = []
            for feats, masks in zip(att_feat, att_per_text.bool()):
                text_feat.append(feats[masks].max(dim=0)[0])
                for feat, mask in zip(feats, masks):
                    if mask.sum() != 0:
                        att_local.append(feat[mask].max(dim=0)[0])

            att_local = torch.vstack(att_local)
            text_feat = torch.vstack(text_feat)
            text_feat = self.text_fc(text_feat)

            out = {'image_fine': image_feat, 'text_fine': text_feat, 'att_local': att_local, 'att_labels': labels}

        elif mode == 2:
            image_feat = self.instance_branch(image_feature)
            image_feat = self.conv_fine(image_feat)
            image_feat = self.global_maxpool(image_feat).squeeze(-1).squeeze(-1)

            out = {'gallery_feat_fine': image_feat}

        elif mode == 3:
            att_feature = att_info['att_feat_local']
            att_mask = att_info['att_feat_mask']

            bs, part, token, emb = att_feature.shape
            att_feat = att_feature.reshape(bs * part, token, emb).permute(0, 2, 1)
            att_feat = self.attribute_branch(att_feat.unsqueeze(-1))
            att_feat = self.conv_fine(att_feat).permute(0, 2, 1).reshape(bs, part, token, -1)

            att_local = []
            text_feat = []
            for feats, masks in zip(att_feat, att_mask.bool()):
                text_feat.append(feats[masks].max(dim=0)[0])

                for feat, mask in zip(feats, masks):
                    if mask.sum() != 0:
                        att_local.append(feat[mask].max(dim=0)[0])

            att_local = torch.vstack(att_local)
            text_feat = torch.vstack(text_feat)
            text_feat = self.text_fc(text_feat)

            out = {'query_feat_fine': text_feat}

        return out


@CROSSALIGN.register_module()
class CrossModalityStructure(BaseModule):
    def __init__(self,
                 visual_embeddings,
                 textual_embeddings,
                 hidden_embeddings,
                 coarse_embeddings,
                 **kwargs
                 ):
        super().__init__()

        model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.text_embed = model_class.from_pretrained(pretrained_weights)
        self.text_embed.eval()
        for p in self.text_embed.parameters():
            p.requires_grad = False

        self.textual_CNN = ResNet_text_50(textual_embeddings, hidden_embeddings)

        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.proposal_branch = nn.Sequential(
            Bottleneck(visual_embeddings, 1024, width=512, conv3=True,
                       downsample=nn.Conv2d(visual_embeddings, 1024, kernel_size=1, bias=False)),
            Bottleneck(1024, coarse_embeddings, width=1024, conv3=True,
                       downsample=nn.Conv2d(1024, coarse_embeddings, kernel_size=1, bias=False))
        )

        self.text_change = nn.Linear(hidden_embeddings, coarse_embeddings)
        self.conv_coarse = conv(coarse_embeddings, coarse_embeddings)

        self.fine_grained_align = Attribute_Align(visual_embeddings, textual_embeddings, **kwargs)

    def get_text_feature(self, bert_token, bert_attention_mask, mode, img_metas, index=0, att_flag=True):
        if mode == 3:
            bert_token = [i for i in bert_token[0]]
            bert_attention_mask = [i for i in bert_attention_mask[0]]

        token = []
        attention_mask = []
        text_length = []
        text_att = []
        for i in range(len(bert_token)):
            token.append(bert_token[i][index])
            attention_mask.append(bert_attention_mask[i][index])
            text_att.append(img_metas[i]['descriptions'][index])
            text_length.append((bert_attention_mask[i][index] == 1).sum())
        token = torch.stack(token)
        attention_mask = torch.stack(attention_mask)
        with torch.no_grad():
            txt = self.text_embed(token, attention_mask=attention_mask)
            txt = txt[0]  # [2, 64, 768]   batchsize:2, token_len:64, emb_dim:768

        if not att_flag:
            att_out = {}
        else:
            att_out = {}
            att = self.get_attributes(text_att)
            att_ind, eff_parts, new_labels = self.get_ind(token, att)
            att['parts'] = eff_parts
            att['labels'] = new_labels
            att_feat = self.get_att_feat_from_sentence(txt, att_ind, att['parts'], att['labels'])

            att_out.update(att)
            att_out['att_ind'] = att_ind
            att_out.update(att_feat)

        return txt, text_length, att_out

    def get_att_feat_from_sentence(self, txt, att_ind, att_parts, att_classes):
        att_feat_expand = []
        att_mask_expand = []
        for feat, indx, part, classes in zip(txt, att_ind, att_parts, att_classes):
            feats_expand = torch.zeros(5, 5, 768, device=feat.device)
            feats_mask = torch.zeros(5, 5, dtype=torch.int, device=feat.device)
            for i, p in enumerate(part):
                ind = indx[i]
                num = ind[1] - ind[0]
                if num > 5:
                    num = 5
                feats_expand[p-1, :num] = feat[ind[0]: ind[0] + num]
                feats_mask[p-1, :num] = classes[p-1]

            att_feat_expand.append(feats_expand.unsqueeze(0))
            att_mask_expand.append(feats_mask.unsqueeze(0))

        att_feat_expand = torch.vstack(att_feat_expand)
        att_mask_expand = torch.vstack(att_mask_expand)

        out = {'att_feat_local': att_feat_expand, 'att_feat_mask': att_mask_expand}

        return out

    def txt_embedding(self, bert_token, bert_attention_mask, mode, img_metas):
        txt, text_length, att_info = self.get_text_feature(bert_token, bert_attention_mask, mode, img_metas)

        _, text_feature = self.textual_CNN(txt)
        text_feature = text_feature.squeeze(2).permute(0, 2, 1)
        text_feature = self.text_change(text_feature).unsqueeze(-1)
        text_feature = text_feature.permute(0, 2, 1, 3)

        text_coarse = self.global_maxpool(text_feature)  # 64,2048
        text_coarse = self.conv_coarse(text_coarse)  # 64,1024

        return text_coarse, att_info

    def image_embedding(self, image_feature):
        bs, pro, emb, h, w = image_feature.shape

        image_feature = image_feature.reshape(bs * pro, emb, h, w)
        image_feature = self.proposal_branch(image_feature).reshape(bs*pro, -1, h*w, 1)

        image_coarse = self.global_maxpool(image_feature)

        image_coarse = self.conv_coarse(image_coarse)

        return image_coarse

    def forward(self, img_metas, mode=0, roi_feats=None, bert_token=None, bert_attention_mask=None):
        image_feature, text_feature, att_info = None, None, None

        if mode == 1:
            image_coarse = self.image_embedding(roi_feats['crops'])
            text_coarse, att_info = self.txt_embedding(bert_token, bert_attention_mask, mode, img_metas)

            image_feature = roi_feats['crops']

            out = {
                "image_coarse": image_coarse.unsqueeze(1),
                "text_coarse": text_coarse.unsqueeze(1),
            }

        elif mode == 2:
            image_coarse = self.image_embedding(roi_feats['crops'].unsqueeze(0))
            image_feature = roi_feats['crops']

            out = {'gallery_feat_coarse': image_coarse}

        elif mode == 3:
            text_coarse, att_info = self.txt_embedding(bert_token, bert_attention_mask, mode, img_metas)

            out = {'query_feat_coarse': text_coarse}

        fine_out = self.fine_grained_align.forward(image_feature, att_info, mode)
        out.update(fine_out)

        return out

    def get_attributes(self, text):
        def clean_text(text):
            text = text.lower()
            text = re.sub(r'[^\x00-\x7F]+', ' ', text)
            text = re.sub(r"([.,!:?()])", r" \1 ", text)
            text = re.sub(r"\s{2,}", " ", text)
            # text = text.replace("-", " ")
            return text

        def get_nps_from_tree(tree, words_original, attachNP=False, skip_single_word=False):
            nps = []
            st = 0
            for subtree in tree:
                if isinstance(subtree, nltk.tree.Tree):
                    if subtree.label() == 'NP':
                        np = subtree.leaves()
                        ed = st + len(np)
                        if not skip_single_word or len(np) > 1:
                            nps.append({'st': st,
                                        'ed': ed,
                                        'text': ' '.join(words_original[st:ed])})
                            if attachNP:
                                nps[-1]['np'] = np
                    st += len(subtree.leaves())
                else:
                    st += 1
            return nps

        def get_nps_nltk_raw(doc):
            GRAMMAR = r"""
            NBAR:
              {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns（名词和形容词，并且以名词结尾）
            NP:
              {<NBAR>}
              {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
            """

            _PARSER = nltk.RegexpParser(GRAMMAR)
            doc = clean_text(doc)

            words_original = nltk.word_tokenize(doc)
            # words_original = doc.split(' ')
            parse_tree = _PARSER.parse(nltk.pos_tag(words_original))
            nps = get_nps_from_tree(parse_tree, words_original)
            return nps

        def glo_att(att):
            att_list = ['man', 'men', 'woman', 'women', 'girl', 'boy', 'person', 'female', 'male', 'lady', 'child', 'guy', 'gentleman', 'kid']
            for k in att_list:
                if k in att:
                    return True
            return False

        def att_parse(nps):
            res, part = [], []
            for k in nps:
                if glo_att(k['text']):
                    res.append(k['text'])
                    part.append(-1)
                    continue
                flag = 0
                for w, c in noun_to_class.items():
                    if w in k['text']:
                        if c not in part:
                            part.append(c)
                            flag = 1
                        break

                if flag == 0:
                    continue
                res.append(k['text'])

            if len(res) == 0:
                res.append(nps[0]['text'])
                part.append(1)

            return res, part

        def label_assign(nps):
            res = [0] * len(noun_to_label)
            for n in nps:
                if glo_att(n['text']):
                    continue
                for i, label in enumerate(noun_to_label):
                    for k, v in label.items():
                        if k in n['text']:
                            res[i] = v
            return res

        # [3, 9, 7, 6, 4]
        attribute_classes = {
            1: {1: ['hat', 'cap', 'hood'], 2: ['glasses', 'sunglasses', 'eyeglasses']},
            2: {3: ['shirt', 'blouse', 'polo', 'tshirt', 'shirts'], 4: ['jacket', 'coat', 'overcoat'],
                5: ['suit', 'tails', 'blazer', 'tuxedo'], 6: ['tank', 'vest', 'undershirt'], 7: ['sweater', 'cardigan'],
                8: ['bra'], 9: ['sweatshirt', 'pullover', 'jumper'], 10: ['jersey']},
            3: {11: ['pants', 'slacks', 'capris', 'trousers', 'khakis'], 12: ['shorts'], 13: ['skirt', 'miniskirt'],
                14: ['jeans'], 15: ['socks', 'stockings', 'pantyhose'], 16: ['leggings', 'tights']},
            4: {17: ['polka'], 18: ['apron'], 19: ['tunic'], 20: ['robe', 'dress', 'gown'], 21: ['overalls']},
            5: {22: ['bag', 'backpack', 'pack', 'bookbag'], 23: ['handbag', 'purse', 'satchel', 'briefcase'],
                24: ['suitcase']}
        }

        class_to_noun = {
            1: ['shirt', 'jacket', 'top', 'blouse', 'suit', 'coat', 'tank', 'sweater', 'polo', 'vest', 'tails',
                'robe', 'polka', 'blazer', 'cardigan', 'tshirt', 'apron', 'undershirt', 'shirts', 'sweatshirt',
                'tunic', 'pullover', 'bra', 'jumper', 'overcoat', 'jersey', 'tuxedo'],
            2: ['pants', 'shorts', 'dress', 'skirt', 'jeans', 'socks', 'leggings', 'slacks', 'capris', 'tights',
                'trousers', 'gown', 'stockings', 'jean', 'pant', 'pantyhose', 'khakis', 'overalls', 'miniskirt'],
            3: ['bag', 'backpack', 'handbag', 'suitcase', 'purse', 'pack', 'bags', 'satchel', 'briefcase', 'bookbag'],
        }

        noun_to_class = {}
        for k in class_to_noun.keys():
            for n in class_to_noun[k]:
                noun_to_class[n] = k

        noun_to_label = []
        for k, v in attribute_classes.items():
            tmp = {}
            for m in v.keys():
                for n in v[m]:
                    tmp[n] = m
            noun_to_label.append(tmp)

        noun_to_class = {}
        for i, lab in enumerate(noun_to_label):
            for k, v in lab.items():
                noun_to_class[k] = i + 1

        res_list = []
        part_list = []
        label_list = []
        for txt in text:
            nps = get_nps_nltk_raw(txt)
            res, part = att_parse(nps)
            labels = label_assign(nps)

            res_list.append(res)
            part_list.append(part)
            label_list.append(labels)

        result = {'atts': res_list, 'parts': part_list, 'labels': label_list}
        return result

    def get_ind(self, tokens, atts):
        index = []
        new_parts = []
        new_labels = []
        for att, parts, token, labels in zip(atts['atts'], atts['parts'], tokens, atts['labels']):
            tok = self.tokenizer(att)['input_ids']
            ind = []
            new_part = []
            for indx, part in zip(tok, parts):
                info = indx[1:-1]
                for k in range(len(token)):
                    if token[k] == info[0]:
                        kk = 1
                        while (kk < len(info)):
                            if token[k + kk] != info[kk]:
                                break
                            kk += 1
                        if kk == len(info):
                            ind.append([k, k + kk])
                            new_part.append(part)
                            break
            tmp = []
            ind_fi, new_part_fi = [], []
            for ind1, part1 in zip(ind, new_part):
                if part1 == -1:
                    tmp = ind1
                else:
                    ind_fi.append(ind1)
                    new_part_fi.append(part1)

            if labels == [0, 0, 0, 0, 0]:
                tmp_labels = [1, 0, 0, 0, 0]
            else:
                tmp_labels = labels

            if len(ind_fi) == 0:
                index.append([tmp])
                new_parts.append([1])
                tmp_labels = [1, 0, 0, 0, 0]
            else:
                index.append(ind_fi)
                new_parts.append(new_part_fi)

            new_labels.append(tmp_labels)

        return index, new_parts, new_labels
