from Network import NoduleNet, test_data
from data_trainning import *
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from constants import *

def entropy(s, j, l):
    assert (j == l)
    p = np.array([s[j],1-s[j]])
    entr = 0
    entr += (-(s*np.log(s))).sum()
    return entr

def diversity(s, j, l):
    pj = np.array([s[j], 1 - s[j]])
    pl = np.array([s[l], 1 - s[l]])
    diver = 0
    diver += ((pj-pl) * np.log(pj/pl)).sum()
    return diver

if __name__ == '__main__':
    fine_tune_dataset = Luna16Dataset(csv_file=cand_path, root_dir=root_path, subset=[0],
                                        transform = transforms.Compose([ToTensor()]))

    fine_tune_dataloader = DataLoader(fine_tune_dataset, batch_size=1, shuffle=False, num_workers=8)
    classes = {'Not Nodule', 'Is Nodule'}

    net = NoduleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    # implement incremental learning
    R = np.zeros([len(fine_tune_dataloader),alpha*1000,alpha*1000])
    scores = []
    for i, data in enumerate(fine_tune_dataloader):
        inputs, labels = data['cube'], data['label']
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        pred = test_data(net, inputs[0])
        if np.mean(pred) > 0.5:
            s = pred.sort(reverse=True)[0 : alpha*1000]
        else:
            s = pred.sort(reverse=False)[0 : alpha*1000]
        # Create matrix R
        for j in range(alpha*1000):
            for l in range(alpha*1000):
                if (j != l):
                    R[i,j,l] = diversity(s,j,l)
                else:
                    R[i,j,l] = entropy(s,j,l)

        scores.append([i,R.sum(axis=(1,2))])

    scores = np.sort(scores, axis=0)[::-1]
    scores = scores[:beta]

    # train on the top b candidates
    for i in range(len(scores)):
        data = dataloader[i]
        inputs, labels = data['cube'], data['label']
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        print(type(inputs.data))
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        running_loss = 0.0
