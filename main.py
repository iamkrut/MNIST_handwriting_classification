import torch
from torchvision import transforms
import torchvision.datasets as datasets
from PIL import Image
import matplotlib.pyplot as plt
from model import Model
import numpy as np


mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                                                        transforms.Normalize((0.1307,), (0.3081,))]))

mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                                                        transforms.Normalize((0.1307,), (0.3081,))]))

mnist_valset, mnist_testset = torch.utils.data.random_split(mnist_testset, [int(0.9 * len(mnist_testset)), int(0.1 * len(mnist_testset))])

train_dataloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(mnist_valset, batch_size=32, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(mnist_testset, batch_size=1, shuffle=False)

print("Training dataset size: ", len(mnist_trainset))
print("Validation dataset size: ", len(mnist_valset))
print("Testing dataset size: ", len(mnist_testset))

# visualize data
# fig=plt.figure(figsize=(5, 2))
# for i in range(1, 5):
#     img = mnist_trainset[i][0]
#     fig.add_subplot(1, 5, i)
#     plt.title(mnist_trainset[i][1].item())
#     plt.imshow(img)
# plt.show()

no_epochs = 100
model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if (torch.cuda.is_available()):
    model.cuda()

train_loss = list()
val_loss = list()
for epoch in range(no_epochs):
    total_train_loss = 0
    total_val_loss = 0

    # training
    for itr, (image, label) in enumerate(train_dataloader):

        if (torch.cuda.is_available()):
            image = image.cuda()
            label = label.cuda()

        optimizer.zero_grad()

        pred = model(image)

        loss = criterion(pred, label)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    total_train_loss = total_train_loss / (itr + 1)
    train_loss.append(total_train_loss)

    # validation
    for itr, (image, label) in enumerate(val_dataloader):

        if (torch.cuda.is_available()):
            image = image.cuda()
            label = label.cuda()

        pred = model(image)

        loss = criterion(pred, label)
        total_val_loss += loss.item()

    total_val_loss = total_val_loss / (itr + 1)
    val_loss.append(total_val_loss)

    print('Epoch: {}/{}, Train Loss: {:.4f}, Val Loss: {:.4f}'
          .format(epoch + 1, no_epochs, total_train_loss, total_val_loss))

plt.plot(np.arange(1, no_epochs+1), train_loss, label="Train loss")
plt.plot(np.arange(1, no_epochs+1), val_loss, label="Validation loss")
plt.xlabel('Loss')
plt.ylabel('Epochs')
plt.title("Loss Plots")
# plt.show()
plt.savefig('loss.png')


