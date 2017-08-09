import torch

from Network import NoduleNet
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from data_trainning import Luna16Dataset, RandomCrop, RandomFlip, ToTensor
from data_testing import Luna16DatasetTest
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

if __name__ == '__main__':
    root_path = '/Users/hyacinth/Downloads/luna16/'
    cand_path = '/Users/hyacinth/Downloads/luna16/candidates.csv'

    transformed_dataset = Luna16Dataset(csv_file=cand_path, root_dir=root_path, subset=[0],
                                        transform = transforms.Compose([RandomFlip((20,36,36)), RandomCrop((20,36,36)),
                                                                        ToTensor()]))
    train_dataloader = DataLoader(transformed_dataset, batch_size=64, shuffle=True, num_workers=4)

    test_dataset = Luna16DatasetTest(csv_file=cand_path, root_dir=root_path, subset=[0],
                                            transform=transforms.Compose(
                                                [ToTensor()]))
    test_dataloader = DataLoader(transformed_dataset, batch_size=1, shuffle=True, num_workers=4)

    classes = {'Not Nodule', 'Is Nodule'}

    net = NoduleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    for epoch in range(8):
        for i in range(1000):
            running_loss = 0.0
            for i, data in enumerate(train_dataloader):
                inputs, labels = data['cube'], data['label']
                inputs, labels = Variable(inputs), Variable(labels)
                # inputs = inputs.float()
                # labels = labels.float()
                print(type(inputs.data))
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.data[0]

                # running_loss += loss.data[0]
                if i % 100 == 99:  # print every 100 mini-batches
                     print('[%d, %5d] loss: %.3f' %
                           (epoch + 1, i + 1, running_loss / 100))
                     running_loss = 0.0
        # save model
        torch.save(net.state_dict(), '')
        #test
        for i, data in enumerate(test_dataloader):
            inputs, labels = data['cube'], data['label']
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = net(inputs)


        print('iter' + str(epoch)+'finished')


                # fig = plt.figure()
                # img = inputs[0][0][10].data.numpy()
                # plt.imshow(img, cmap='gray')
                # plt.show()