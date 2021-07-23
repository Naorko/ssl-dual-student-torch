def execute_ms(args, train_loader, val_loader):
    msi_flag = args.model_arch == 'msi'

    return {'accuracy': 1}


def execute_mt(args, train_loader, val_loader):
    return {'accuracy': 1}


def execute_model(args, train_loader, val_loader):
    if args.model_arch in ['ms', 'msi']:
        return execute_ms(args, train_loader, val_loader)
    elif args.model_arch == 'mt':
        return execute_mt(args, train_loader, val_loader)
    else:
        raise Exception('Unknown model architecture')
