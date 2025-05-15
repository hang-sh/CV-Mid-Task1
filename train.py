import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os, csv
from data_process import get_datasets
from test import test_model

# 解压数据集文件
# file_path = r'.\\caltech-101\\101_ObjectCategories.tar.gz'
# with tarfile.open(file_path, 'r:gz') as tar:
#    tar.extractall(path=r'.\\data\\')

# 如果csv文件不存在，则创建并写入表头
csv_file_path = './experiment_results.csv'
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Experiment Name', 'Learning Rate', 'Batch Size', 'Weight Decay', 'Step Size', 'Gamma',
                         'Best Val Accuracy', 'Test Accuracy'])
# 保存训练好的模型
save_dir = './saved_models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)    

data_path = '.\\data\\101_ObjectCategories'
train_data, val_data, _, classes = get_datasets(data_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(classes)

# 超参数配置
lr = 0.01
batch_size = 32
weight_decay = 0.0001
step_size = 10
gamma = 0.9
epochs = 25

# 记录训练过程中的各种信息
exp_name = f"resnet18_lr{lr}_bs{batch_size}_wd{weight_decay}_ss{step_size}_g{gamma}"
log_dir = f'./runs/{exp_name}'
filename = f'{exp_name}_best.pth'
save_path = os.path.join(save_dir, filename)
writer = SummaryWriter(log_dir) 

# 加载预训练的ResNet18模型
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# 加载随机初始化的ResNet18模型
# model = models.resnet18(weights=None)

# 修改输出层
n_hidden = model.fc.in_features
model.fc = nn.Linear(n_hidden, num_classes)
model = model.to(device)

# 数据加载
TrainLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
ValLoader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay) # 随机梯度下降
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma) # 学习率下降
best_val_acc = 0.0

print(f"开始训练: {exp_name}")
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    val_loss = 0.0
    correct = 0
    for images, labels in TrainLoader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 在每个epoch结束时计算平均训练loss并记录
    avg_train_loss = train_loss / len(TrainLoader)
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    
    # 验证集评估
    model.eval()
    with torch.no_grad():
        for images, labels in ValLoader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, y_pred = torch.max(outputs, 1)
            correct += sum(y_pred == labels).item() / len(labels)

    avg_val_loss = val_loss / len(ValLoader)
    val_acc = correct / len(ValLoader)  
    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, Val Accuracy: {val_acc}")
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"保存最佳模型到: {save_path}")
    
    scheduler.step()

# 模型测试
test_acc = test_model(data_path, save_path)  
print(f"Test Accuracy: {test_acc}")

# 记录实验结果到CSV
with open(csv_file_path, mode='a', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow([exp_name, lr, batch_size, weight_decay, step_size, gamma, best_val_acc, test_acc])













