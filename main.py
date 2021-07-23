import logging
import random
from itertools import product

import torchvision
from torch.utils.data import DataLoader

from src import datasets, data
from src.cli import parse_dict_args, LOG
from src.run_context import RunContext


def hp_product():
    bs_hp = [32, 64, 128]
    n_labels_ratio_hp = [0.5, 0.4, 0.25, 0.1]
    wd_hp = [0, 1e-2, 1e-3, 1e-4]
    momentum_hp = [0.3, 0.5, 0.8, 0.9]
    hp_product_lst = list(product(bs_hp, n_labels_ratio_hp, wd_hp, momentum_hp))

    return hp_product_lst


def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


def partition_dict(dict_of_list, n):
    return {k: partition(lst, n) for k, lst in dict_of_list.items()}


def extract_fold(labeled_dict, unlabeled_dict, fold_idxs):
    def extract_dict(d):
        idxs = []

        for fold_idx in fold_idxs:
            for k in d.keys():
                idxs.extend(d[k][fold_idx])

        return idxs

    return extract_dict(labeled_dict), extract_dict(unlabeled_dict)


def create_loader(args, dataset, labeled_idxs, unlabeled_idxs, idxs_in_dict, eval=False):
    labeled, unlabeled = extract_fold(labeled_idxs, unlabeled_idxs, idxs_in_dict)
    sampler = data.TwoStreamBatchSampler(
        unlabeled, labeled, args.batch_size, args.labeled_batch_size)

    if not eval:
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=args.workers,
            pin_memory=True)

    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)

    return loader


def execute_model(args, train_loader, val_loader):
    return None, None


def nested_cross_validation(args, outer_k=10, inner_k=3):
    # create dataloaders
    dataset_config = datasets.__dict__[args.dataset](tnum=args.model_num)
    num_classes = dataset_config.pop('num_classes')

    train_dataset = torchvision.datasets.ImageFolder(dataset_config['datadir'], dataset_config['train_transformation'])
    eval_dataset = torchvision.datasets.ImageFolder(dataset_config['datadir'], dataset_config['eval_transformation'])
    ds_size = len(train_dataset.imgs)

    with open(args.labels) as f:
        labels = dict(line.split(' ') for line in f.read().splitlines())

    labeled_idxs, unlabeled_idxs = data.relabel_dataset_dict(train_dataset, labels)
    _, _ = data.relabel_dataset(eval_dataset, labels)

    labeled_idxs = partition_dict(labeled_idxs, outer_k)
    unlabeled_idxs = partition_dict(unlabeled_idxs, outer_k)

    # Outer cross-validation fold
    for test_idx in range(outer_k):
        train_val_idx = [i for i in range(outer_k) if i != test_idx]

        inner_fold_split = partition(train_val_idx, inner_k)

        # select 50 random params
        # for p in params do
        # update args
        # results = []
        for val_idx in range(inner_k):
            train_idx = [i for i in range(inner_k) if i != val_idx]

            train_idx = [j for i in train_idx for j in inner_fold_split[i]]
            val_idx = inner_fold_split[val_idx]

            train_loader = create_loader(args, train_dataset, labeled_idxs, unlabeled_idxs, train_idx)
            val_loader = create_loader(args, eval_dataset, labeled_idxs, unlabeled_idxs, val_idx, eval=True)

            # results.append(results)
            results = execute_model(args, train_loader, val_loader)

        # end for
        # select params[i] for i=argmax(results)
        # update args

        train_val_loader = create_loader(args, train_dataset, labeled_idxs, unlabeled_idxs, train_val_idx)
        test_loader = create_loader(args, eval_dataset, labeled_idxs, unlabeled_idxs, [test_idx], eval=True)

        results = execute_model(args, train_val_loader, test_loader)


def defaults(arch):
    args = {
        'model-arch': arch,

        # data
        'dataset': 'cifar100',
        'labels': '/home/naorko/DL/ssl-dual-student-torch/third_party/data-local/labels/cifar100/10000_balanced_labels/00.txt',

        # Technical Details
        'workers': 2,
        'checkpoint_epochs': 20,

        # optimization
        'batch-size': 100,
        'labeled-batch-size': 50,

        # optimizer
        'lr': 0.1,
        'nesterov': True,
        'weight-decay': 0,
        'momentum': 0.9,

        # architecture
        'arch': 'cnn13',
        'model_num': 2,

        # constraint
        'consistency_scale': 10.0,
        'consistency_rampup': 5,
        'stable_threshold': 0.8,
        'stabilization_scale': 100.0,
        'stabilization_rampup': 5,
        'logit_distance_cost': 0.01,
        'consistency': 100.0,  # mt-only

        'title': 'ms_cifar10_1000l_cnn13',
        'n_labels': 1000,
        'epochs': 1  # 300, TODO: More epochs?
    }
    return args


def run(title, n_labels, **kwargs):
    LOG.info('run title: %s', title)

    context = RunContext(__file__, "{}".format(n_labels))
    fh = logging.FileHandler('{0}/log.txt'.format(context.result_dir))
    fh.setLevel(logging.INFO)
    LOG.addHandler(fh)

    args = parse_dict_args(**kwargs)
    nested_cross_validation(args)


if __name__ == '__main__':
    args = defaults('ms')
    run(**args)
