import torch.optim as optim
import torch.utils.data
import utils
from utils import *

'''
 Train a VGG16 model to classify the identity in order to get identity loss during attack.
 The example is MPIIGaze dataset.

'''

if __name__ == "__main__":
    dataset_name = "gaze"
    # file = "./Celeba/" + dataset_name + ".json"
    file = "/projects/tang/GMI-Attack/GMI-Attack/Celeba/gaze.json"
    # / MPIIGaze / Data / Normalized ["p00", "p02", "p03", "p06", "p08", "p09", "p10", "p11", "p12", "p14"]
    args = utils.load_params(json_file=file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch_size = args["VGG16"]["batch_size"]
    # batch_size = 64

    gaze_dataset, dataloader = init_gaze_data('/projects/tang/GMI-Attack/GMI-Attack/Celeba/data/MPIIGaze/Data'
                                              '/Normalized', [0,2,3,6,8,9,10,11,12,14])  # Utils
    train_size = int(args["dataset"]["train_percentage"] * len(gaze_dataset))
    test_size = len(gaze_dataset) - train_size
    

    train_dataset, test_dataset = torch.utils.data.random_split(gaze_dataset, [train_size, test_size])
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=2)

    model = classify.VGG16(n_classes=10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args["VGG16"]["lr"], momentum=0.9)

    for epoch in range(args["VGG16"]["epochs"]):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            print(f'{i}/{len(trainloader)}', end='\r')
            inputs, labels, files = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs[1], labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i == 100:
                torch.save(model, args["VGG16"]["save_path"])

        print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / (i + 1)}')
        #Validation loop
        correct = 0
        total = 0
        i = 0
        with torch.no_grad():
            for data in testloader:
                print(f'{i}/{len(testloader)}', end='\r')
                inputs, labels, files = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs[1].data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                i += 1
        print(f'Accuracy: {correct / total}, wrong count: {total-correct}')
        torch.save(model, args["VGG16"]["save_path"])

    print('Finished Training')