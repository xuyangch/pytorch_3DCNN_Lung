import torch
import time
from Network import NoduleNet
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import data
# from data import Luna16Dataset,ToTensor
# from data_trainning import Luna16Dataset, RandomCrop, RandomFlip, ToTensor
from data_testing import Luna16DatasetTest
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from constants import *
import matplotlib


if __name__ == '__main__':

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            data.RandomCrop((36,36,36)),
            data.RandomFlip((36,36,36)),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            data.CenterCrop((20, 36, 36)),
            transforms.ToTensor(),
        ]),
    }

    transformed_dataset = Luna16Dataset(csv_file=cand_path, root_dir=root_path, subset=[0],
                                        transform = transforms.Compose([ToTensor()]))
    train_dataloader = DataLoader(transformed_dataset, batch_size=64, shuffle=True, num_workers=8)

    test_dataset = Luna16DatasetTest(csv_file=cand_path, root_dir=root_path, subset=[0],
                                            transform=transforms.Compose(
                                                [ToTensor()]))
    test_dataloader = DataLoader(transformed_dataset, batch_size=1, shuffle=True, num_workers=8)


    classes = {'Not Nodule', 'Is Nodule'}

    net = NoduleNet()
    net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    for epoch in range(1):
        print('epoch ' + str(epoch) + ' start' , file=f)
        print('epoch ' + str(epoch) + ' start')
        for subepoch in range(1):
            localtime = time.asctime(time.localtime(time.time()))
            print("subepoch start time :", localtime, file=f)
            print("subepoch start time :", localtime)
            cnt = 0
            print('subepoch ' + str(subepoch) + ' start', file=f)
            print('subepoch ' + str(subepoch) + ' start')
            running_loss = 0.0
            for i, data in enumerate(train_dataloader):
                cnt += 1
                localtime = time.asctime(time.localtime(time.time()))
                print("batch start time :", localtime, file=f)
                print("batch start time :", localtime)
                print('batch ' + str(cnt), file=f)
                print('batch ' + str(cnt))
                inputs, labels = data['cube'], data['label']

                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                # inputs = inputs.float()
                # labels = labels.float()
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.data[0]
                localtime = time.asctime(time.localtime(time.time()))
                print("batch end time :", localtime, file=f)
                print("batch end time :", localtime)
                # running_loss += loss.data[0]
                if i % 64 == 63:  # print every 64 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                            (epoch + 1, subepoch + 1, running_loss / 64), file=f)
                    print('[%d, %5d] loss: %.3f' %
                            (epoch + 1, subepoch + 1, running_loss / 64))
                    running_loss = 0.0



            # save model
            torch.save(net.state_dict(), './weight/weights.save')
            print('model saved', file=f)
            print('model saved')
            localtime = time.asctime(time.localtime(time.time()))
            print("subepoch end time :", localtime, file=f)
            print("subepoch end time :", localtime)

            #test
            # for i, data in enumerate(test_dataloader):
            #     inputs, labels = data['cube'], data['label']
            #     inputs, labels = Variable(inputs), Variable(labels)
            #     outputs = net(inputs)


                # fig = plt.figure()
                # img = inputs[0][0][10].data.numpy()
                # plt.imshow(img, cmap='gray')
                # plt.show()