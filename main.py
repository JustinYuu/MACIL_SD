from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import time
import math
import numpy as np
import random
import os
from avce_network import AVCE_Model, Single_Model
from avce_dataset import Dataset
from train import avce_train as train
from test import avce_test as test
import option
from utils import Prepare_logger, cosine_scheduler


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    global logger
    torch.multiprocessing.set_start_method('spawn')
    setup_seed(2333)
    args = option.parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    logger = Prepare_logger(eval=False)
    logger.info(args)
    train_loader = DataLoader(Dataset(args, test_mode=False),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=5, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model_av = AVCE_Model(args).cuda()
    model_v = Single_Model(args).cuda()

    total_params = sum(p.numel() for p in model_av.parameters())
    total_params += sum(p.numel() for p in model_v.parameters())
    logger.info(f'{total_params/1e6:.3f}M parameters.')
    total_trainable_params = sum(p.numel() for p in model_av.parameters() if p.requires_grad is True)
    total_trainable_params += sum(p.numel() for p in model_v.parameters() if p.requires_grad is True)
    logger.info(f'{total_trainable_params/1e6:.3f}M training parameters.')
    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    criterion = torch.nn.BCELoss()
    optimizer_av = optim.Adam(model_av.parameters(), lr=args.lr, weight_decay=0.000)
    optimizer_v = optim.Adam(model_v.parameters(), lr=args.lr / 5, weight_decay=0.000)
    scheduler_av = optim.lr_scheduler.CosineAnnealingLR(optimizer_av, T_max=60, eta_min=0)
    scheduler_v = optim.lr_scheduler.CosineAnnealingLR(optimizer_v, T_max=60, eta_min=0)
    gt = np.load(args.gt)
    best_av_auc = 0
    best_v_auc = 0
    best_epoch = 0
    av_auc, v_auc = test(test_loader, model_av, model_v, gt, 0)
    logger.info('Random initalization: offline av_auc:{0:.4}\n'.format(av_auc))
    for epoch in range(args.max_epoch):
        st = time.time()
        lamda_a2b = min(args.lamda_a2b, args.lamda_cof * epoch)
        lamda_a2n = min(args.lamda_a2n, args.lamda_cof * epoch)
        av_loss, v_loss = train(train_loader, model_av, model_v, optimizer_av, optimizer_v, criterion,
                                lamda_a2b, lamda_a2n, logger)
        scheduler_av.step()
        scheduler_v.step()

        with torch.no_grad():
            m = cosine_scheduler(base_value=args.m, final_value=1, curr_epoch=epoch, epochs=50)
            if m != 1.0:
                for param_av in model_av.named_parameters():
                    if 'sa_a' in param_av[0] or 'fc_a' in param_av[0]:
                        continue
                    for param_v in model_v.named_parameters():
                        if param_av[0] == param_v[0]:
                            param_av[1].data.mul_(m).add_((1 - m) * param_v[1].detach().data)
                            break
                        elif param_av[0] == 'att_mmil.fc.weight' and param_v[0] == 'fc.weight':
                            param_av[1].data.mul_(m).add_((1 - m) * param_v[1].detach().data)
                            break
                        elif param_av[0] == 'att_mmil.fc.bias' and param_v[0] == 'fc.bias':
                            param_av[1].data.mul_(m).add_((1 - m) * param_v[1].detach().data)
                            break

        av_auc, v_auc = test(test_loader, model_av, model_v, gt, epoch)
        if av_auc > best_av_auc:
            best_av_auc = av_auc
            best_v_auc = v_auc
            best_epoch = epoch
            torch.save(model_av.state_dict(), './ckpt/' + args.model_name + '.pkl')
        logger.info('av_loss:{:.4} | v_loss:{:.4}\n'.format(av_loss, v_loss))
        logger.info(
            'Epoch {}/{}: av_auc:{:.4} | v_auc:{:.4} | m={:.4}\n'.format(epoch, args.max_epoch, av_auc, v_auc, m))
    logger.info(
        'Best Performance in Epoch {}: av_auc:{:.4} | v_auc:{:.4}\n'.format(best_epoch, best_av_auc, best_v_auc))
