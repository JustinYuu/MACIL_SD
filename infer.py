from torch.utils.data import DataLoader
import torch
import numpy as np
from avce_network import AVCE_Model as Model
from avce_dataset import Dataset
from test import avce_test as test
import option
import time

if __name__ == '__main__':
    args = option.parser.parse_args()
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=5, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    model = Model(args)
    model = model.cuda()
    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(args.model_dir).items()})
    gt = np.load(args.gt)
    st = time.time()
    av_auc, _ = test(test_loader, model, None, gt, 0)
    print('Time:{}'.format(time.time() - st))
    print('offline av_auc:{0:.4}\n'.format(av_auc))
