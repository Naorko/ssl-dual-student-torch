import argparse
import logging
import re

from src import architectures
from src import datasets

LOG = logging.getLogger('main')


def create_parser():
    parser = argparse.ArgumentParser(description='Dual Student SSL PyTorch Version')

    # chosen architecture
    parser.add_argument('--model-arch', default='ms', type=str, choices=['ms', 'msi', 'mt'],
                        help='The chosen Model (ms | msi | mt)', metavar='ARCH')

    # data
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10', choices=datasets.__all__,
                        help='dataset: ' + ' | '.join(datasets.__all__) + ' (default: cifar10)')
    parser.add_argument('--n-labels', metavar='LABELS', default=4, type=int,
                        help='How many labels are in the dataset (default: 10000)')

    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--labels', default=None, type=str, metavar='FILE',
                        help='list of image labels (default: based on directory structure)')

    # optimization
    parser.add_argument('--batch-size', default=100, type=int,
                        metavar='N', help='mini-batch size (default: 100)')
    parser.add_argument('--labeled-batch-size', default=None, type=int,
                        metavar='N', help="labeled examples per minibatch (default: no constrain)")

    parser.add_argument('--lr', default=0.1, type=float,
                        metavar='LR', help='max learning rate')
    parser.add_argument('--initial-lr', default=0.0, type=float,
                        metavar='LR', help='initial learning rate when using linear rampup')
    parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                        help='length of learning rate rampup in the beginning')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for optimizer')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='use nesterov momentum', metavar='BOOL')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--checkpoint-epochs', default=1, type=int,
                        metavar='EPOCHS',
                        help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
    parser.add_argument('--validation-epochs', default=1, type=int,
                        metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
    parser.add_argument('--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    # archtecture
    parser.add_argument('--arch', '-a', metavar='ARCH', default='cnn13', choices=architectures.__all__,
                        help='model architecture: ' + ' | '.join(architectures.__all__))

    # constraint

    parser.add_argument('--consistency-scale', default=None, type=float, metavar='WEIGHT',
                        help='use consistency loss with given weight (default: None)')
    parser.add_argument('--consistency-rampup', default=30, type=int, metavar='EPOCHS',
                        help='length of the consistency loss ramp-up')
    parser.add_argument('--consistency', default=100.0, type=float, metavar='CONSISTENCY',
                        help='consistency')

    parser.add_argument('--stable-threshold', default=0.0, type=float, metavar='THRESHOLD',
                        help='threshold for stable sample')
    parser.add_argument('--stabilization-scale', default=None, type=float, metavar='WEIGHT',
                        help='use stabilization loss with given weight (default: None)')
    parser.add_argument('--stabilization-rampup', default=30, type=int, metavar='EPOCHS',
                        help='length of the stabilization loss ramp-up')

    parser.add_argument('--logit-distance-cost', default=-1, type=float, metavar='WEIGHT',
                        help='let the student model have two outputs and use an MSE loss　between '
                             'the logits with the given weight (default: only have one output)')
    parser.add_argument('--ema-decay', default=0.97, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.97)')  # MT- ONLY!
    parser.add_argument('--lr-rampdown-epochs', default=210, type=int, metavar='RAMPDOWN',
                        help='lr rampdown epochs (default: 210)')  # MT- ONLY!

    # for Multiple Student
    parser.add_argument('--model-num', default=2, type=int, metavar='MS',
                        help='number of the student models during training, which is required by '
                             ' multiple_student.py [set it to 2 is equal to Dual Student] (default: 2)')

    return parser


def parser_commandline_args():
    return create_parser().parse_args()


def parse_dict_args(**kwargs):
    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))

    LOG.info("Using these command line args: %s", " ".join(cmdline_args))

    return create_parser().parse_args(cmdline_args)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2epochs(v):
    try:
        if len(v) == 0:
            epochs = []
        else:
            epochs = [int(string) for string in v.split(",")]
    except:
        raise argparse.ArgumentTypeError(
            'Expected comma-separated list of integers, got "{}"'.format(v))
    if not all(0 < epoch1 < epoch2 for epoch1, epoch2 in zip(epochs[:-1], epochs[1:])):
        raise argparse.ArgumentTypeError(
            'Expected the epochs to be listed in increasing order')
    return epochs
