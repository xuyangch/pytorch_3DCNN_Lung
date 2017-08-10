from Network import NoduleNet, test_data
from data_trainning import *
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

if __name__ == '__main__':
    # set constant
    # root_path = '/Users/hyacinth/Downloads/luna16/'
    # cand_path = '/Users/hyacinth/Downloads/luna16/candidates.csv'
    alpha = 0.25

    fine_tune_dataset = Luna16Dataset(csv_file=cand_path, root_dir=root_path, subset=[0],
                                        transform = transforms.Compose([ToTensor()]))

    fine_tune_dataloader = DataLoader(fine_tune_dataset, batch_size=1, shuffle=True, num_workers=4)
    classes = {'Not Nodule', 'Is Nodule'}

    net = NoduleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    # implement incremental learning
    R = np.zeros([len(fine_tune_dataloader),alpha*1000,alpha*1000])
    for i, data in enumerate(fine_tune_dataloader):
        inputs, labels = data['cube'], data['label']
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        pred = test_data(net, inputs[0])
        if np.mean(pred) > 0.5:
            s = pred.sort(reverse=True)[0 : alpha*1000]
        else:
            s = pred.sort(reverse=False)[0 : alpha*1000]


    for epoch in range(8):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
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
            # print statistics
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

            # running_loss += loss.data[0]
            # if i % 100 == 99:  # print every 100 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 100))
            #     running_loss = 0.0


            # fig = plt.figure()
            # img = inputs[0][0][10].data.numpy()
            # plt.imshow(img, cmap='gray')
            # plt.show()