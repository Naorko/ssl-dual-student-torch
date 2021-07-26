import torchvision.transforms as transforms

from src import data
from src.utils import export


@export
def cifar100(tnum=2):
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
    train_transformation = data.TransformNTimes(
        transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]), n=tnum)

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/cifar100',
        'num_classes': 100
    }


@export
def mnist(tnum=2):
    channel_stats = dict(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
    train_transformation = data.TransformNTimes(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]), n=tnum)
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/mnist',
        'num_classes': 10,
    }


@export
def cinic10(tnum=2):
    channel_stats = dict(mean=[0.47889522, 0.47227842, 0.43047404],
                         std=[0.24205776, 0.23828046, 0.25874835])
    train_transformation = data.TransformNTimes(
        transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]), n=tnum)

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/cinic10',
        'num_classes': 10
    }


@export
def cub200(tnum=2):
    channel_stats = dict(mean=[0.47889522, 0.47227842, 0.43047404],
                         std=[0.24205776, 0.23828046, 0.25874835])
    train_transformation = data.TransformNTimes(
        transforms.Compose([
            transforms.Resize(size=(32, 32)),
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]), n=tnum)

    eval_transformation = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/cub200',
        'num_classes': 200
    }
