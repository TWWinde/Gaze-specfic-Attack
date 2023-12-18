from copy import deepcopy

import torch, os, classify, utils, sys
import numpy as np
import torch.nn as nn
from utils import *
from sklearn.model_selection import train_test_split

from Celeba import facenet

dataset_name = "celeba"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_path = "./result"
log_path = os.path.join(root_path, "evaluation_classifier_logs")
model_path = os.path.join(root_path, "evaluation_classifier_ckp")
os.makedirs(model_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)


def init_dataloader1(args, file_path, batch_size=64, mode="gan", iterator=False):
    tf = time.time()

    if mode == "attack":
        shuffle_flag = False
    else:
        shuffle_flag = True

    if args['dataset']['name'] == "celeba":
        data_set = dataloader.ImageFolder(args, file_path, mode)

    else:
        data_set = dataloader.GrayFolder(args, file_path, mode)

    if iterator:
        data_loader = torch.utils.data.DataLoader(data_set,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle_flag,
                                                  drop_last=True,
                                                  num_workers=0,
                                                  pin_memory=True).__iter__()
    else:
        data_loader = torch.utils.data.DataLoader(data_set,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle_flag,
                                                  drop_last=True,
                                                  num_workers=2,
                                                  pin_memory=True)
        interval = time.time() - tf
        print('Initializing data loader took %ds' % interval)

    return data_set, data_loader


def test(model, criterion, dataloader):
    tf = time.time()
    model.eval()
    loss, cnt, ACC = 0.0, 0, 0

    for img, iden in dataloader:
        img, iden = img.to(device), iden.to(device)
        bs = img.size(0)
        iden = iden.view(-1)

        out_prob = model(img)[-1]
        out_iden = torch.argmax(out_prob, dim=1).view(-1)
        ACC += torch.sum(iden == out_iden).item()
        cnt += bs

    return ACC * 100.0 / cnt


def train_reg(args, model, criterion, optimizer, trainloader, testloader, n_epochs):
    best_ACC = 0.0
    model_name = args['dataset']['model_name']

    # scheduler = MultiStepLR(optimizer, milestones=adjust_epochs, gamma=gamma)

    for epoch in range(n_epochs):
        tf = time.time()
        ACC, cnt, loss_tot = 0, 0, 0.0
        model.train()

        for i, (img, iden) in enumerate(trainloader):
            img, iden = img.to(device), iden.to(device)
            bs = img.size(0)
            iden = iden.view(-1)

            feats, out_prob = model(img)
            cross_loss = criterion(out_prob, iden)
            loss = cross_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()
            loss_tot += loss.item() * bs
            cnt += bs

        train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt
        test_acc = test(model, criterion, testloader)

        interval = time.time() - tf
        if test_acc > best_ACC:
            best_ACC = test_acc
            best_model = deepcopy(model)

        if (epoch + 1) % 10 == 0:
            torch.save({'state_dict': model.state_dict()},
                       os.path.join(model_path, "allclass_epoch{}.tar").format(epoch))

        print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval,
                                                                                                   train_loss,
                                                                                                   train_acc, test_acc))
        # scheduler.step()

    print("Best Acc:{:.2f}".format(best_ACC))
    return best_model, best_ACC


def main(args, model_name, trainloader, testloader):
    n_classes = args["dataset"]["n_classes"]
    mode = args["dataset"]["mode"]
    if model_name == "VGG16":
        if mode == "reg":
            net = classify.VGG16(n_classes)
        elif mode == "vib":
            net = classify.VGG16_vib(n_classes)

    elif model_name == "FaceNet":
        net = classify.FaceNet(n_classes)
        BACKBONE_RESUME_ROOT = os.path.join(root_path, "backbone_ir50_ms1m_epoch120.pth")
        print("Loading Backbone Checkpoint ")
        utils.load_my_state_dict(net.feature, torch.load(BACKBONE_RESUME_ROOT))
        # utils.weights_init_classifier(net.fc_layer)

    elif model_name == "FaceNet_all":
        net = classify.FaceNet(202599)
        BACKBONE_RESUME_ROOT = os.path.join(root_path, "backbone_ir50_ms1m_epoch120.pth")
        print("Loading Backbone Checkpoint ")
        utils.load_my_state_dict(net.feature, torch.load(BACKBONE_RESUME_ROOT))
        # utils.weights_init_classifier(net.fc_layer)

    elif model_name == "FaceNet64":
        net = facenet.FaceNet64(n_classes)
        BACKBONE_RESUME_ROOT = os.path.join(root_path, "backbone_ir50_ms1m_epoch120.pth")
        print("Loading Backbone Checkpoint ")
        utils.load_my_state_dict(net.feature, torch.load(BACKBONE_RESUME_ROOT))
        # net.fc_layer.apply(net.weight_init)

    elif model_name == "IR50":
        if mode == "reg":
            net = classify.IR50(n_classes)
        elif mode == "vib":
            net = classify.IR50_vib(n_classes)
        BACKBONE_RESUME_ROOT = "ir50.pth"
        print("Loading Backbone Checkpoint ")
        load_my_state_dict(net.feature, torch.load(BACKBONE_RESUME_ROOT))

    elif model_name == "IR152":
        if mode == "reg":
            net = classify.IR152(n_classes)
        else:
            net = classify.IR152_vib(n_classes)

        BACKBONE_RESUME_ROOT = os.path.join(root_path,
                                            "Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth")
        print("Loading Backbone Checkpoint ")
        utils.load_state_dict(net.feature, torch.load(BACKBONE_RESUME_ROOT))

    else:
        print("Model name Error")
        exit()

    optimizer = torch.optim.SGD(params=net.parameters(),
                                lr=args[model_name]['lr'],
                                momentum=args[model_name]['momentum'],
                                weight_decay=args[model_name]['weight_decay'])

    epochs = args[model_name]["epochs"]
    criterion = nn.CrossEntropyLoss().cuda()
    net = torch.nn.DataParallel(net).to(device)

    mode = args["dataset"]["mode"]
    n_epochs = args[model_name]['epochs']
    best_ACC = 0
    print("Start Training!")

    if mode == "reg":
        best_model, best_acc = engine.train_reg(args, net, criterion, optimizer, trainloader, testloader, n_epochs)
    elif mode == "vib":
        best_model, best_acc = engine.train_vib(args, net, criterion, optimizer, trainloader, testloader, n_epochs)

    torch.save({'state_dict': best_model.state_dict()},
               os.path.join(model_path, "{}_{:.2f}_allclass.tar").format(model_name, best_acc))


if __name__ == '__main__':
    dataset_name = "evaluation_classifier"
    file = "./" + dataset_name + ".json"
    args = utils.load_params(json_file=file)
    model_name = args['dataset']['model_name']

    log_file = "{}.txt".format(model_name)
    utils.Tee(os.path.join(log_path, log_file), 'w')

    print(log_file)
    print("---------------------Training [%s]---------------------" % model_name)
    utils.print_params(args["dataset"], args[model_name], dataset=args['dataset']['name'])

    train_file = args['dataset']['train_file_path']
    test_file = args['dataset']['test_file_path']
    _, trainloader = init_dataloader1(args, train_file, mode="train")
    _, testloader = init_dataloader1(args, test_file, mode="test")

    main(args, model_name, trainloader, testloader)
