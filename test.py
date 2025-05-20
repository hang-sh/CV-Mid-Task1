import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from data_process import get_datasets

# 测试集评估
def test_model(data_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载测试集
    _, _, test_data = get_datasets(data_path)
    TestLoader = DataLoader(test_data, batch_size=32, shuffle=True)
    # 加载模型
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # model = models.resnet18(weights=None)
    # 修改输出层
    n_hidden = model.fc.in_features
    model.fc = nn.Linear(n_hidden, 101)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    correct = 0
    # 开始测试
    with torch.no_grad():
        for images, labels in TestLoader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, y_pred = torch.max(outputs, 1)
            correct += sum(y_pred == labels).item() / len(labels)

    test_acc = correct / len(TestLoader)  # 计算测试集上的准确率
    return test_acc

if __name__ == "__main__":
   data_path = '.\\data\\101_ObjectCategories' 
   model_path = 'saved_models\\resnet18_lr0.001_bs32_wd0.0005_best.pth'   # 模型路径
   test_acc = test_model(data_path, model_path)
   print(f"Test Accuracy: {test_acc}")


