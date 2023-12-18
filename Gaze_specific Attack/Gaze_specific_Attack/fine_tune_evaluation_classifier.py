import torch.optim as optim
import utils
from facenet import FaceNet64
from utils import *
from gaze_estimation.config import get_default_config
from gaze_estimation.datasets import create_dataset
from torch.utils.data import DataLoader
'''
fine tune FaceNet as evaluation classifier on mpiifacegaze dataset
'''
config = get_default_config()
config.merge_from_file('./configs/mpiifacegaze/resnet_simple_14_train.yaml')  # MPIIFaceGaze
config.freeze()
train_dataset, val_dataset = create_dataset(config, True, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], True)
train_loader = DataLoader(
    train_dataset,
    batch_size=config.train.batch_size,
    shuffle=True,
    num_workers=config.train.train_dataloader.num_workers,
    pin_memory=config.train.train_dataloader.pin_memory,
    drop_last=config.train.train_dataloader.drop_last,
)

if __name__ == "__main__":

    file = "./evaluation_classifier.json"
    model_path = './result/FaceNet64.tar'
    args = utils.load_params(json_file=file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch_size = args["FaceNet64"]["batch_size"]
    # batch_size = 64

    model = FaceNet64(num_classes=10)
    model = nn.DataParallel(model).to(device)
    # T = torch.load(target_path)
    # print(T.keys())
    ckp_model = torch.load(model_path)['state_dict']  # FaceNet64
    utils.load_my_state_dict(model, ckp_model)
    for p in model.parameters():
        p.requires_grad_(True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args["FaceNet64"]["lr"], momentum=0.9)

    for epoch in range(args["FaceNet64"]["epochs"]):

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for i, data in enumerate(train_loader, 0):
            print(f'{i}/{len(train_loader)}', end='\r')
            images, poses, gazes, ids = data
            # for i in range(len(ids)):
            #     ids[i] = [0,2,3,6,8,9,10,11,12,14].index(ids[i])
            images = images.to(device)
            ids = ids.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs[1], ids)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs[1].data, 1)
            correct_predictions += (predicted == ids).sum().item()
            total_samples += ids.size(0)

        accuracy = correct_predictions / total_samples
        print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / (i + 1)} Training Accuracy: {accuracy * 100:.2f}%')

        # Validation loop
        # correct = 0
        # total = 0
        # i = 0
        # with torch.no_grad():
        #     for data in testloader:
        #         print(f'{i}/{len(testloader)}', end='\r')
        #         inputs, labels, files = data
        #         inputs = inputs.to(device)
        #         labels = labels.to(device)
        #         outputs = model(inputs)
        #         _, predicted = torch.max(outputs[1].data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()
        #         i += 1
        # print(f'Accuracy: {correct / total}, wrong count: {total-correct}')

        torch.save(model, './result/FaceNet64_face.tar')

    print('Finished Training')
