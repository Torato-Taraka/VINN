import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from .. import get_dataset, get_model, train, ConfuseMatrix


def trainer(model_name, dataset_name, classes=1000, transform=None,
            train_size=64, test_size=50, lr=0.001, epochs=100, gpu=True):
    # device
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')

    # dataset
    dataset = get_dataset(dataset_name)
    train_dataset = dataset(train=True, transform=transform, download=True, root='../data')
    test_dataset = dataset(train=False, transform=transform, download=True, root='../data')
    train_loader = DataLoader(train_dataset, batch_size=train_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

    # model
    model = get_model(model_name)().to(device)

    # train
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    scheduler = None

    metrics = ConfuseMatrix(classes=classes, names=("accuracy",))

    # tensorboard = SummaryWriter(comment=f"{model_name}_{dataset_name}")
    tensorboard = None

    train(model=model, train_loader=train_loader, test_loader=test_loader,
          criterion=criterion, optimizer=optimizer,
          scheduler=scheduler, epochs=epochs,
          metrics=metrics, device=device, tensorboard=tensorboard)

    torch.save(model, f"./trained/{model_name}_{dataset_name}.pth")
