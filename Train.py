import torch
from tqdm import tqdm

from CCRF import CCRF

from torch.utils.data import DataLoader
import torch.optim as optim
from Dataloader import ObjDataset, extractObject, my_collate
# from model import vgg

Outdoor_root = 'D:/CNN_CCRF/CNN&CCRF/KITTI/data'
Outdoor_Train_Set = ObjDataset(root_path=Outdoor_root, transform='train', isNYU=False, loader=extractObject)
Indoor_root = 'D:/CNN_CCRF/CNN&CCRF/_NYU_DEPTH/data'
Indoor_Train_Set = ObjDataset(root_path=Indoor_root, transform='train', isNYU=True, loader=extractObject)


def Train(model, batchsize=1, epochs=1):
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(device)
    model.to(device)
    Indoor_train_DataLoader = DataLoader(Indoor_Train_Set, batch_size=1, shuffle=True, collate_fn=my_collate)
    Outdoor_train_DataLoader = DataLoader(Outdoor_Train_Set, batch_size=1, shuffle=True, collate_fn=my_collate)

    TrainLoader = {
        'Indoor': Indoor_train_DataLoader,
        'Outdoor': Outdoor_train_DataLoader
    }
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    lost_funciton = CCRF()
    # Train the model,
    for Loader in TrainLoader:
        running_set_loss = 0.0
        for epoch in range(epochs):
            model.train()
            running_epoch_loss = 0.0
            for step, train_data in enumerate(tqdm(TrainLoader[Loader])):
                images, labels = train_data
                z, R = model(images.to(device))
                loss = lost_funciton(labels.to(device), z.to(device), R.to(device))
                # loss.requires_grad_(True)  # 加入此句就行了
                # print(loss)
                loss.backward(retain_graph=True)
                if (step + 1) % batchsize == 0 or step == len(TrainLoader[Loader]) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                running_epoch_loss += loss
                pass
            print('Epoch {} loss is : {:.3f}'.format(epoch + 1, running_epoch_loss / len(TrainLoader[Loader])))
            running_set_loss += running_epoch_loss
        torch.save(model.state_dict(), './net')
        print('{} set loss is : {:.3f}'.format(Loader, running_set_loss / (epochs * len(TrainLoader[Loader]))))
        pass
    print('Training Finished!')
    torch.save(model.state_dict(), './net')
    pass
