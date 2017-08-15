import torch
import time
from Network import NoduleNet
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from data_trainning_primitive import Luna16Dataset,ToTensor
# from data_trainning import Luna16Dataset, RandomCrop, RandomFlip, ToTensor
from data_testing import Luna16DatasetTest
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from constants import *
import matplotlib


if __name__ == '__main__':
    # root_path = '/Users/hyacinth/Downloads/luna16/'
    # cand_path = '/Users/hyacinth/Downloads/luna16/candidates.csv'

    transformed_dataset = Luna16Dataset(csv_file=cand_path, root_dir=root_path, subset=[0],
                                        transform = transforms.Compose([ToTensor()]))
    train_dataloader = DataLoader(transformed_dataset, batch_size=64, shuffle=True, num_workers=8)

    test_dataset = Luna16Dataset(csv_file=cand_path, root_dir=root_path, subset=[0],
                                            transform=transforms.Compose([ToTensor()]))
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

    classes = {'Not Nodule', 'Is Nodule'}

    net = NoduleNet()
    net = net.cuda()
    net.load_state_dict(torch.load('./weight/weights0.save'))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    # start epoch...
    for epoch in range(1):
        print('epoch ' + str(epoch) + ' start' , file=f)
        print('epoch ' + str(epoch) + ' start')

        # subecho is to walk through all data augmentation
        for subepoch in range(1):

            # start time
            localtime = time.asctime(time.localtime(time.time()))
            print("subepoch start time :", localtime, file=f)
            print("subepoch start time :", localtime)
            print('subepoch ' + str(subepoch) + ' start', file=f)
            print('subepoch ' + str(subepoch) + ' start')

            # init
            cnt = 0
            running_loss = 0.0
            running_correctness = 0

            # walk through dataloader
            for i, data in enumerate(test_dataloader):
                cnt += 1
                localtime = time.asctime(time.localtime(time.time()))
                print("batch start time :", localtime, file=f)
                print("batch start time :", localtime)
                print('batch ' + str(cnt), file=f)
                print('batch ' + str(cnt))

                # get inputs, set to GPU
                inputs, labels = data['cube'], data['label']
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

                # get outputs
                outputs = net(inputs)

                # statistics

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                running_loss += loss.data[0]
                running_correctness += torch.sum(preds == labels.data)
                localtime = time.asctime(time.localtime(time.time()))
                print('preds is: '+str(preds))
                print('label is: ' + str(labels.data))
                print("batch end time :", localtime, file=f)
                print("batch end time :", localtime)

                # running_loss += loss.data[0]
                if i % 64 == 63:  # print every 64 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                            (epoch + 1, subepoch + 1, running_loss / 64), file=f)
                    print('[%d, %5d] loss: %.3f' %
                            (epoch + 1, subepoch + 1, running_loss / 64))
                    running_loss = 0.0

            correct = running_loss / len(test_dataloader)
            print('correct is: ' + str(correct))

            # save model
            # torch.save(net.state_dict(), './weight/weights.save')
            print('model saved', file=f)
            print('model saved')

            # print time
            localtime = time.asctime(time.localtime(time.time()))
            print("subepoch end time :", localtime, file=f)
            print("subepoch end time :", localtime)
