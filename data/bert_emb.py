import pickle
import json
import transformers as ppb

def bert_pretrain(des, tokenizer, flag, max_len=64):
    res = []
    res.append(tokenizer.encode(des[0], add_special_tokens=True))
    len_txt = 1
    if flag:   # PRW-train only contain one description
        len_txt = 2
        res.append(tokenizer.encode(des[1], add_special_tokens=True))

    padded = []
    for i in range(len_txt):
        padded.append([1]*max_len)
        length = len(res[i])
        if length < max_len:
            padded[i][length:] = [0] * (max_len - length)
            res[i] += [0] * (max_len - length)
        else:
            res[i] = res[i][:max_len-1] + [res[i][-1]]

    return res, padded

def emb_one_set(ann_file, res_file, flag=True):
    with open(ann_file, 'r') as f:
        anno = f.read()

    anno = json.loads(anno)

    for i in range(len(anno)):
        tmp = []
        for k in anno[i]['desrciption']:
            if k == '':
                continue
            elif len(tmp) == 2:
                tmp[-1] = tmp[-1] + ' ' + k
            else:
                tmp.append(k)
        anno[i]['description'] = tmp

        token, mask = bert_pretrain(tmp, tokenizer, flag)
        anno[i]['token'] = token
        anno[i]['mask'] = mask

    f = open(res_file, 'wb')
    pickle.dump(anno, f, 0)
    f.close()

if __name__ == "__main__":
    max_emb_len = 64

    model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    print("Start embedding descriptions...")
    emb_one_set('./CUHK-SYSU-TBPS/CUHK-SYSU-TBPS_train', './CUHK-SYSU-TBPS/CUHK-SYSU-TBPS_train.pkl')
    emb_one_set('./CUHK-SYSU-TBPS/CUHK-SYSU-TBPS_test', './CUHK-SYSU-TBPS/CUHK-SYSU-TBPS_test.pkl')
    print("Successfully embedding CUHK-SYSU-TBPS dataset")

    emb_one_set('./PRW-TBPS/PRW-TBPS_train', './PRW-TBPS/PRW-TBPS_train.pkl', flag=False)
    emb_one_set('./PRW-TBPS/PRW-TBPS_test', './PRW-TBPS/PRW-TBPS_test.pkl')
    print("Successfully embedding PRW-TBPS dataset")
