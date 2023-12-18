import torch.optim as optim
import utils
from utils import *
from gaze_estimation.config import get_default_config
from gaze_estimation.datasets import create_dataset
from torch.utils.data import DataLoader

'''
 Train a VGG16 model to classify the identity of NVGaze in order to get identity loss during attack.
 The example is NVGaze dataset.
 
'''

config = get_default_config()
config.merge_from_file('./configs/NVGaze.yaml')  # MPIIFaceGaze
config.freeze()
train_dataset, val_dataset = create_dataset(config, True, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], True)
train_loader = DataLoader(
    train_dataset,
    batch_size=config.train.batch_size,
    shuffle=True,
    num_workers=config.train.train_dataloader.num_workers,
    pin_memory=config.train.train_dataloader.pin_memory,
    drop_last=config.train.train_dataloader.drop_last,
)


if __name__ == "__main__":
    dataset_name = "gaze"
    # file = "./Celeba/" + dataset_name + ".json"
    file = "./gaze.json"

    args = utils.load_params(json_file=file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch_size = args["VGG16"]["batch_size"]
    print(batch_size)
    # batch_size = 64

    model = classify.VGG16(n_classes=14)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args["VGG16"]["lr"], momentum=0.9)

    for epoch in range(args["VGG16"]["epochs"]):

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for i, data in enumerate(train_loader, 0):
            print(f'{i}/{len(train_loader)}', end='\r')
            images, gazes, identity = data
            ids = torch.nn.functional.one_hot(identity, num_classes=14)
            ids = ids.to(torch.float32)
            images = images.to(device)
            ids = ids.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            #print(outputs[1].shape)
            loss = criterion(outputs[1], ids)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            identity = identity.to(device)
            _, predicted = torch.max(outputs[1].data, 1)
            correct_predictions += (predicted == identity).sum().item()
            total_samples += ids.size(0)
        accuracy = correct_predictions / total_samples
        print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / (i + 1)} Training Accuracy: {accuracy * 100:.2f}%')

        torch.save(model, './result/Identity_classifier_NVGaze.tar')

    print('Finished Training')
