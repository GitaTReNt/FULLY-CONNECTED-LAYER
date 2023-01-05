import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
#超参数的设置
#mistgpu服务器使用指南：继承全局站点包
epochs = 30
batch_size = 16 #8-2，16-4，32-8，batchsize和output的数量差了4倍
lr = 0.0001  # 防止忽略细节
device = torch.device('cuda')
print(device)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=4)
        self.conv2 = torch.nn.Conv2d(10, 8, kernel_size=4)
        self.fc1 = nn.Linear(8, 256)#4x4x8   89888,
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        self.AA2D = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # 卷积-relu激活-maxpooling池化
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.AA2D(x) # -1自适应 0行数 ps：nn.Linear()结构，输入输出都是维度为一的值 view实现类似reshape的功能
        x = self.AA2D(x)
        x = torch.squeeze(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

model = Net()

model.to(device)    #模型加载到设备上训练

img_valid_pth = "/home/mist/data/val"
imgs_train_pth = "/home/mist/data/train"

transform = transforms.Compose([
    transforms.Resize((224,224)),  # 随机裁剪+resize
    transforms.ToTensor(),  # rgb归一化
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 使用论文的参数
])

dataset_to_train = datasets.ImageFolder(imgs_train_pth, transform)
dataset_to_valid = datasets.ImageFolder(img_valid_pth, transform)
print(dataset_to_train.class_to_idx)    # 安排 labels
# labels = torch.Tensor(labels).long()
train_loader = torch.utils.data.DataLoader(dataset_to_train, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset_to_valid, batch_size=batch_size)




#优化器一类的
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.5)
criterion = nn.CrossEntropyLoss()


def train(epoch):

    loss_total = 0
    correct = 0
    #total = len(train_loader.dataset)
    model.train() # 进行训练
    #loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):   # 开始迭代
        data = Variable(data).to(device)
        target = Variable(target).to(device) #这里输出的是8个图片的判断值，其中1是狗 0是猫
        #targets = target.to(device, dtype=torch.float)
        #print(target.shape)
        #data = data.flatten()
        #data = np.reshape(data, 16)
        #print(data.shape)
        output = model(data)#这里output只输出了两个结果，而不是batchsize

        #target2 = target.unsqueeze(0)#target是batchsize个数，1/0表示的是标签值，而output是一个2，2的tensor、、、、我们需要的output是一个【batchsize，2】的输出
        ##output2 = output.flatten()
       # print(output2.shape)#batch：8，2-2、、batch：16，4-2
        optimizer.zero_grad()
        _,predict_label = torch.max(output.data, 1)
        loss = criterion(output, target)#4,8 mismatch
        loss.backward()

        correct += torch.sum(predict_label == target)

        optimizer.step()

        loss_total += loss.data.item()
        # if (batch_idx + 1) % 5 == 0:
        #    print('Train : Epoch =  {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
        #               100. * (batch_idx + 1) / len(train_loader), loss.item()))
        # losses =  loss_total / len(train_loader)
        # i += 1
        # loop.set_description(f'Epoch [{epoch}/{epochs}]')
        # loop.set_postfix(loss=loss.item() / (batch_idx + 1), acc=correct / total)
        # print('\n', "train :  epoch =", epoch,
        # " learn rate =" , optimizer.param_groups[0]['lr'],
        # " loss =", losses /(batch_idx + 1) , " accuracy =", correct / total, '\n')
        if (batch_idx + 1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
        average_loss = loss_total / len(train_loader)
        print('epoch:{},loss:{}'.format(epoch, average_loss))



def valid(model,valid_loader):
    loss_total = 0
    correct = 0
    total = len(valid_loader.dataset)
    model.eval()
    print(total, len(valid_loader))
    with torch.no_grad():

        for data, target in valid_loader:

            data, target = Variable(data).to(device), Variable(target).to(device)

            output = model(data)

            loss = criterion(output, target)

            _, pred = torch.max(output.data, 1)

            correct += torch.sum(pred == target)

            print_loss = loss.data.item()

            loss_total += print_loss

        correct = correct.data.item()

        accuracy = correct / total

        losses = loss_total / len(valid_loader)

        scheduler.step(print_loss / epoch + 1)

        print('\nvalidation: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            losses, correct, len(valid_loader.dataset), 100 * accuracy))

for epoch in range(1, epochs + 1):

    train(epoch)

    valid(model, valid_loader)

torch.save(model, 'model.pth')
