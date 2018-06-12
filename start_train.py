import time
from utils.multibox_loss import MultiBoxLoss
from torch.autograd import Variable
import torch
import torch.optim as optim
from utils.ssd import ssd
import torch.backends.cudnn as cudnn
from utils.road_datareader import road_datareader
from torch.utils import data as tfd
from utils.logger import Logger
import os

"""

При сборке конфигурации сети использовались материалы:

    [1] https://arxiv.org/pdf/1512.02325.pdf for more details

    [2] https://github.com/amdegroot/ssd.pytorch

    [3] https://github.com/zengyu714/standard-panel-classification
"""

logger = Logger('./logs')
ckpt_name = 'overfit'

cudnn.benchmark = True
use_cuda = True

step = 0

def train(epoch):
    print('*** epoch: {} ***'.format(epoch))
    start = time.clock()
    net.train()
    train_loss = 0
    for batch_idx, (images, loc_targets, conf_targets) in enumerate(training_generator):
        images = images.cuda()
        loc_targets = loc_targets.cuda()
        conf_targets = conf_targets.cuda()

        images = Variable(images)
        loc_targets = Variable(loc_targets)
        conf_targets = Variable(conf_targets)

        optimizer.zero_grad()
        loc_preds, conf_preds = net(images)
        loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]

        print('train: loss {:.3f}'.format(train_loss / (batch_idx + 1)))
        info = {'train_loss': train_loss / (batch_idx + 1), 'lr': optimizer.param_groups[0]['lr']}
        global step
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step + 1)
        step += 1
    print('*** epoch ends, took {}s ***'.format(time.clock() - start))

def test(epoch):
    net.eval()
    valid_loss = 0
    for batch_idx, (images, loc_targets, conf_targets) in enumerate(trainval_generator):
        images = images.cuda()
        loc_targets = loc_targets.cuda()
        conf_targets = conf_targets.cuda()

        images = Variable(images, volatile=True)
        loc_targets = Variable(loc_targets)
        conf_targets = Variable(conf_targets)

        loc_preds, conf_preds = net(images)
        loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
        valid_loss += loss.data[0]
        print('test: loss {:.3f}'.format(valid_loss / (batch_idx + 1)))
        info = {'valid_loss': valid_loss / (batch_idx + 1)}
        global step
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step + 1)

    # Save checkpoint.
    global best_loss
    valid_loss /= len(trainval_generator)
    if valid_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'loss': valid_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + ckpt_name + '_epoch' + str(epoch) + '.pth')
        best_loss = valid_loss

train_data = road_datareader("./RoadDamageDataset/All", split='train')
trainval_data = road_datareader("./RoadDamageDataset/All", split='trainval')

params = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 4}

training_generator = tfd.DataLoader(train_data, **params)
trainval_generator = tfd.DataLoader(trainval_data, **params)

if __name__ == '__main__':

    torch.utils.backcompat.keepdim_warning.enabled = True
    torch.utils.backcompat.broadcast_warning.enabled = True

    net = ssd()
    net.load_state_dict(torch.load('./models/ssd.pth', map_location=lambda storage, loc: storage))
    net.cuda()

    criterion = MultiBoxLoss()
    optimizer = optim.SGD(net.parameters(), lr=10e-3, momentum=0.9, weight_decay=1e-4)

    for epoch in range(10):
        train(epoch)
        test(epoch)
