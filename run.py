import os
import sys
import json
import csv
import time
import argparse
import numpy as np
from pprint import pprint
from collections import defaultdict as ddict
from ordered_set import OrderedSet

import torch
from torch.utils.data import DataLoader

from helper import *
from data_loader import TrainDataset, TestDataset
from model.models import CompGCN_TransE, CompGCN_DistMult, CompGCN_ConvE

class Runner(object):
    def load_data(self):
        """
        加载三元组数据，并加载边权重（若提供）。
        """
        ent_set, rel_set = OrderedSet(), OrderedSet()
        # 第一遍：收集所有实体和关系
        for split in ['train', 'test', 'valid']:
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        # 添加反向关系
        self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_ent      = len(self.ent2id)
        self.p.num_rel      = len(self.rel2id) // 2
        self.p.embed_dim    = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

        self.data = ddict(list)
        sr2o = ddict(set)

        # 加载权重（如果指定）
        if self.p.weight_file is not None:
            with open(self.p.weight_file, 'r') as f:
                self.train_weights = [float(line.strip()) for line in f]
        else:
            self.train_weights = None

        train_triples = []  # 存储训练三元组用于权重对齐
        for split in ['train', 'test', 'valid']:
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub, rel, obj))

                if split == 'train':
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel+self.p.num_rel)].add(sub)
                    train_triples.append((sub, rel, obj))

        self.data = dict(self.data)

        # 验证权重数量
        if self.train_weights is not None:
            if len(train_triples) != len(self.train_weights):
                raise ValueError("Number of weights ({}) does not match number of training triples ({})".format(
                    len(self.train_weights), len(train_triples)))
        else:
            self.train_weights = [1.0] * len(train_triples)

        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel+self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples  = ddict(list)

        for (sub, rel), obj in self.sr2o.items():
            self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj),        'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

        self.triples = dict(self.triples)

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return DataLoader(
                    dataset_class(self.triples[split], self.p),
                    batch_size      = batch_size,
                    shuffle         = shuffle,
                    num_workers     = max(0, self.p.num_workers),
                    collate_fn      = dataset_class.collate_fn
                )

        # 测试集不 shuffle，保证顺序与 triples 列表一致
        self.data_iter = {
            'train':        get_data_loader(TrainDataset, 'train',         self.p.batch_size, shuffle=True),
            'valid_head':   get_data_loader(TestDataset,  'valid_head',   self.p.batch_size, shuffle=False),
            'valid_tail':   get_data_loader(TestDataset,  'valid_tail',   self.p.batch_size, shuffle=False),
            'test_head':    get_data_loader(TestDataset,  'test_head',    self.p.batch_size, shuffle=False),
            'test_tail':    get_data_loader(TestDataset,  'test_tail',    self.p.batch_size, shuffle=False),
        }

        self.train_triples = train_triples
        self.edge_index, self.edge_type, self.edge_weight = self.construct_adj()

    def construct_adj(self):
        """
        构建邻接矩阵及对应的关系类型和边权重。
        """
        edge_index, edge_type, edge_weight = [], [], []

        # 原始方向
        for (sub, rel, obj), w in zip(self.train_triples, self.train_weights):
            edge_index.append((sub, obj))
            edge_type.append(rel)
            edge_weight.append(w)

        # 反向方向
        for (sub, rel, obj), w in zip(self.train_triples, self.train_weights):
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)
            edge_weight.append(w)

        edge_index  = torch.LongTensor(edge_index).to(self.device).t()
        edge_type   = torch.LongTensor(edge_type).to(self.device)
        edge_weight = torch.FloatTensor(edge_weight).to(self.device)

        return edge_index, edge_type, edge_weight

    def __init__(self, params):
        self.p          = params
        self.logger     = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

        self.logger.info(vars(self.p))
        pprint(vars(self.p))

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_data()

        # 保存实体和关系映射
        mapping = {
            'ent2id': self.ent2id,
            'id2ent': {str(k): v for k, v in self.id2ent.items()},
            'rel2id': self.rel2id,
            'id2rel': {str(k): v for k, v in self.id2rel.items()}
        }
        mapping_path = os.path.join('./checkpoints', self.p.name + '_mapping.json')
        os.makedirs('./checkpoints', exist_ok=True)
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        self.logger.info(f'Entity and relation mappings saved to {mapping_path}')

        self.model        = self.add_model(self.p.model, self.p.score_func, self.edge_weight)
        self.optimizer    = self.add_optimizer(self.model.parameters())

    def add_model(self, model, score_func, edge_weight):
        model_name = '{}_{}'.format(model, score_func).lower()

        if model_name == 'compgcn_transe':
            model = CompGCN_TransE(self.edge_index, self.edge_type, params=self.p, edge_weight=edge_weight)
        elif model_name == 'compgcn_distmult':
            model = CompGCN_DistMult(self.edge_index, self.edge_type, params=self.p, edge_weight=edge_weight)
        elif model_name == 'compgcn_conve':
            model = CompGCN_ConvE(self.edge_index, self.edge_type, params=self.p, edge_weight=edge_weight)
        else:
            raise NotImplementedError(f'Model {model_name} not implemented')

        model.to(self.device)
        return model

    def add_optimizer(self, parameters):
        return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)

    def read_batch(self, batch, split):
        if split == 'train':
            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label
        else:
            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label

    def save_model(self, save_path):
        state = {
            'state_dict'    : self.model.state_dict(),
            'best_val'      : self.best_val,
            'best_epoch'    : self.best_epoch,
            'optimizer'     : self.optimizer.state_dict(),
            'args'          : vars(self.p)
        }
        torch.save(state, save_path)

    def load_model(self, load_path):
        state = torch.load(load_path)
        self.model.load_state_dict(state['state_dict'])
        self.best_val   = state['best_val']
        self.best_val_mrr = self.best_val['mrr']
        self.optimizer.load_state_dict(state['optimizer'])

    def predict(self, split='valid', mode='tail_batch', return_ranks=False):
        """
        进行预测并返回结果。
        if return_ranks: 返回 (results, ranks_list)
        else: 返回 results
        """
        self.model.eval()
        all_ranks = [] if return_ranks else None

        with torch.no_grad():
            results = {}
            data_key = '{}_{}'.format(split, mode.split('_')[0])
            if data_key not in self.data_iter:
                raise KeyError(f"Data key {data_key} not found.")
            data_loader = self.data_iter[data_key]
            iterator = iter(data_loader)

            for step, batch in enumerate(iterator):
                sub, rel, obj, label = self.read_batch(batch, split)
                pred = self.model.forward(sub, rel)
                b_range = torch.arange(pred.size(0), device=self.device)
                target_pred = pred[b_range, obj]
                # 屏蔽所有正确实体（过滤设置）
                pred = torch.where(label.byte(), -torch.ones_like(pred) * 1e8, pred)
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
                ranks = ranks.float()

                if return_ranks:
                    all_ranks.extend(ranks.cpu().tolist())

                results['count'] = results.get('count', 0.0) + torch.numel(ranks)
                results['mr'] = results.get('mr', 0.0) + torch.sum(ranks).item()
                results['mrr'] = results.get('mrr', 0.0) + torch.sum(1.0 / ranks).item()
                for k in range(10):
                    results['hits@{}'.format(k+1)] = results.get('hits@{}'.format(k+1), 0.0) + torch.numel(ranks[ranks <= (k+1)])

                if step % 100 == 0:
                    self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))

            # 归一化
            for k in list(results.keys()):
                if k != 'count':
                    results[k] /= results['count']

        if return_ranks:
            return results, all_ranks
        else:
            return results

    def evaluate(self, split, epoch, return_ranks=False):
        """
        评估模型在验证集或测试集上的性能。
        if return_ranks: 返回 (results, left_ranks, right_ranks)
        else: 返回 results
        """
        if return_ranks:
            left_results, left_ranks = self.predict(split=split, mode='tail_batch', return_ranks=True)
            right_results, right_ranks = self.predict(split=split, mode='head_batch', return_ranks=True)
        else:
            left_results = self.predict(split=split, mode='tail_batch', return_ranks=False)
            right_results = self.predict(split=split, mode='head_batch', return_ranks=False)

        results = get_combined_results(left_results, right_results)
        self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(
            epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))

        if return_ranks:
            return results, left_ranks, right_ranks
        else:
            return results

    def run_epoch(self, epoch, val_mrr=0):
        self.model.train()
        losses = []
        train_iter = iter(self.data_iter['train'])

        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            sub, rel, obj, label = self.read_batch(batch, 'train')

            pred = self.model.forward(sub, rel)
            loss = self.model.loss(pred, label)

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if step % 100 == 0:
                self.logger.info('[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses), self.best_val_mrr, self.p.name))

        loss = np.mean(losses)
        self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
        return loss

    def fit(self):
        self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
        save_path = os.path.join('./checkpoints', self.p.name)

        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        kill_cnt = 0
        for epoch in range(self.p.max_epochs):
            train_loss = self.run_epoch(epoch, val_mrr)
            val_results = self.evaluate('valid', epoch, return_ranks=False)

            if val_results['mrr'] > self.best_val_mrr:
                self.best_val = val_results
                self.best_val_mrr = val_results['mrr']
                self.best_epoch = epoch
                self.save_model(save_path)
                kill_cnt = 0
            else:
                kill_cnt += 1
                if kill_cnt % 10 == 0 and self.p.gamma > 5:
                    self.p.gamma -= 5
                    self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
                if kill_cnt > 25:
                    self.logger.info("Early Stopping!!")
                    break

            self.logger.info('[Epoch {}]: Training Loss: {:.5}, Valid MRR: {:.5}\n\n'.format(epoch, train_loss, self.best_val_mrr))

        self.logger.info('Loading best model, Evaluating on Test data')
        self.load_model(save_path)
        test_results, left_ranks, right_ranks = self.evaluate('test', self.best_epoch, return_ranks=True)

        # 保存测试集聚合结果
        test_results_path = os.path.join('./checkpoints', self.p.name + '_test_results.json')
        with open(test_results_path, 'w') as f:
            serializable_results = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                                    for k, v in test_results.items()}
            json.dump(serializable_results, f, indent=2)
        self.logger.info(f'Test results saved to {test_results_path}')

        # 保存详细排名（尾预测）
        test_tail_triples = self.triples['test_tail']
        tail_csv_path = os.path.join('./checkpoints', self.p.name + '_test_tail_ranks.csv')
        with open(tail_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['head', 'relation', 'tail', 'rank'])
            for triple_dict, rank in zip(test_tail_triples, left_ranks):
                h, r, t = triple_dict['triple']
                h_name = self.id2ent[h]
                r_name = self.id2rel[r]
                t_name = self.id2ent[t]
                writer.writerow([h_name, r_name, t_name, rank])
        self.logger.info(f'Test tail ranks saved to {tail_csv_path}')

        # 保存详细排名（头预测）
        test_head_triples = self.triples['test_head']
        head_csv_path = os.path.join('./checkpoints', self.p.name + '_test_head_ranks.csv')
        with open(head_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['head', 'relation', 'tail', 'rank'])
            for triple_dict, rank in zip(test_head_triples, right_ranks):
                h, r_inv, t = triple_dict['triple']
                # 转换为原始方向
                if r_inv >= self.p.num_rel:
                    r = r_inv - self.p.num_rel
                    original_h = t
                    original_t = h
                else:
                    # 若未使用逆关系（理论上不会发生），保持原样
                    r = r_inv
                    original_h = h
                    original_t = t
                h_name = self.id2ent[original_h]
                r_name = self.id2rel[r]
                t_name = self.id2ent[original_t]
                writer.writerow([h_name, r_name, t_name, rank])
        self.logger.info(f'Test head ranks saved to {head_csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-name',        default='testrun',                 help='Set run name for saving/restoring models')
    parser.add_argument('-data',        dest='dataset',         default='FB15k-237',            help='Dataset to use, default: FB15k-237')
    parser.add_argument('-model',       dest='model',       default='compgcn',      help='Model Name')
    parser.add_argument('-score_func',  dest='score_func',   default='conve',       help='Score Function for Link prediction')
    parser.add_argument('-opn',         dest='opn',             default='corr',                 help='Composition Operation to be used in CompGCN')

    parser.add_argument('-batch',       dest='batch_size',      default=128,    type=int,       help='Batch size')
    parser.add_argument('-gamma',       type=float,             default=40.0,                   help='Margin')
    parser.add_argument('-gpu',         type=str,               default='0',                    help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-epoch',       dest='max_epochs',  type=int,       default=500,        help='Number of epochs')
    parser.add_argument('-l2',          type=float,             default=0.0,                    help='L2 Regularization for Optimizer')
    parser.add_argument('-lr',          type=float,             default=0.001,                  help='Starting Learning Rate')
    parser.add_argument('-lbl_smooth',  dest='lbl_smooth',  type=float,     default=0.1,        help='Label Smoothing')
    parser.add_argument('-num_workers', type=int,               default=10,                     help='Number of processes to construct batches')
    parser.add_argument('-seed',        dest='seed',            default=41504,  type=int,       help='Seed for randomization')

    parser.add_argument('-restore',     dest='restore',         action='store_true',            help='Restore from the previously saved model')
    parser.add_argument('-bias',        dest='bias',            action='store_true',            help='Whether to use bias in the model')

    parser.add_argument('-num_bases',   dest='num_bases',   default=-1,     type=int,   help='Number of basis relation vectors to use')
    parser.add_argument('-init_dim',    dest='init_dim',    default=100,    type=int,   help='Initial dimension size for entities and relations')
    parser.add_argument('-gcn_dim',     dest='gcn_dim',     default=200,    type=int,   help='Number of hidden units in GCN')
    parser.add_argument('-embed_dim',   dest='embed_dim',   default=None,   type=int,   help='Embedding dimension to give as input to score function')
    parser.add_argument('-gcn_layer',   dest='gcn_layer',   default=1,      type=int,   help='Number of GCN Layers to use')
    parser.add_argument('-gcn_drop',    dest='dropout',     default=0.1,    type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('-hid_drop',    dest='hid_drop',    default=0.3,    type=float, help='Dropout after GCN')

    # ConvE specific hyperparameters
    parser.add_argument('-hid_drop2',   dest='hid_drop2',   default=0.3,    type=float, help='ConvE: Hidden dropout')
    parser.add_argument('-feat_drop',   dest='feat_drop',   default=0.3,    type=float, help='ConvE: Feature Dropout')
    parser.add_argument('-k_w',         dest='k_w',         default=10,     type=int,   help='ConvE: k_w')
    parser.add_argument('-k_h',         dest='k_h',         default=20,     type=int,   help='ConvE: k_h')
    parser.add_argument('-num_filt',    dest='num_filt',    default=200,    type=int,   help='ConvE: Number of filters in convolution')
    parser.add_argument('-ker_sz',      dest='ker_sz',      default=7,      type=int,   help='ConvE: Kernel size to use')

    # Weight file
    parser.add_argument('-weight_file', dest='weight_file', default=None,   help='Path to file containing edge weights for training triples (one per line)')

    parser.add_argument('-logdir',      dest='log_dir',     default='./log/',               help='Log directory')
    parser.add_argument('-config',      dest='config_dir',  default='./config/',            help='Config directory')

    args = parser.parse_args()

    if not args.restore:
        args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')

    set_gpu(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = Runner(args)
    model.fit()