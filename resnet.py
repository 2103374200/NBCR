import os
import torchvision.models as models
from sklearn.metrics import  matthews_corrcoef, confusion_matrix
from torch.optim import Adam
from torch.optim.lr_scheduler import  LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, roc_auc_score
import data_read_save

seed = 443
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


trainfile_path = './data/trainset.txt'
testfile_path = './data/testset.txt'
# 读取数据
seqs=data_read_save.read_sequences(trainfile_path);
test_seqs = data_read_save.read_sequences(testfile_path)
# 将数据转换为DataFrame
df = pd.DataFrame(seqs, columns=['id', 'sequence'])
df=df['sequence'].tolist()
test_df = pd.DataFrame(test_seqs, columns=['id', 'sequence'])
test_df=test_df['sequence'].tolist()

# 前2206个样本的标签为1，后2206个样本的标签为0
labels = torch.cat((torch.ones(2206, dtype=torch.long), torch.zeros(2206, dtype=torch.long)))
# 生成测试集的标签
test_labels = torch.cat((torch.ones(552, dtype=torch.long), torch.zeros(552, dtype=torch.long)))

# 初始化模型和分词器
model_name1 = './500_multi'
model_name2 = './bert2'
tokenizer1 = AutoTokenizer.from_pretrained(model_name1,trust_remote_code=True)
tokenizer2 = AutoTokenizer.from_pretrained(model_name2,trust_remote_code=True)

class NerDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        item1 = self.data[idx]
        label = self.labels[idx]
        return item1,label

    def __len__(self):
        return len(self.labels)


train_dataset = NerDataset(df, labels)
test_dataset = NerDataset(test_df, test_labels)


import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoModel

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 加载bert模型
        self.bert1 = AutoModelForMaskedLM.from_pretrained(model_name1, trust_remote_code=True)
        self.bert2 = AutoModel.from_pretrained(model_name2)
        self.resnet = models.resnet18(weights=None,num_classes=1000)
        # self.resnet=resnest18()
        # 使用二维卷积替代LSTM，并添加池化层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 2048), stride=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 1026), stride=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.dropout = nn.Dropout(0.5)

        # 最后的预测层
        self.relu = nn.ReLU()
        self.fc1 = torch.nn.Linear(9472, 1024)
        self.fc2 = torch.nn.Linear(1000, 2)
    def forward(self, inputs1, inputs2):

        hidden_states1 = self.bert1(inputs1)[0]
        hidden_states2 = self.bert2(inputs2)[0]
        hidden_states = torch.cat((hidden_states2, hidden_states1), dim=2)
        # 调整维度以适应二维卷积
        hidden_states = hidden_states.unsqueeze(1)
        # 使用二维卷积层和池化层
        conv_out = self.conv1(hidden_states)
        conv_out = self.relu(conv_out)
        conv_out = self.pool1(conv_out)
        conv_out = self.conv2(conv_out)
        conv_out = self.relu(conv_out)
        conv_out = self.pool2(conv_out)

        # conv_out=self.resnet(conv_out)
        conv_out = self.resnet(conv_out)

        outputs = self.fc2(conv_out)
        outputs = self.dropout(outputs)
        outputs = outputs.softmax(dim=1)
        return outputs



def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs1 = tokenizer1(texts,max_length=198, padding="max_length", truncation=True,  return_tensors="pt")["input_ids"]

    inputs2 = tokenizer2(texts, max_length=198, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]

    # inputs["labels"] = torch.tensor(labels)
    return inputs1,inputs2,torch.tensor(labels)

trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_func)
testloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_func)

model = MyModel()
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=2e-6)
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1 if epoch > 4 else 1)

# 验证循环
def evaluate(epoch,model, trainloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_prob=[]
    all_labels = []
    with torch.no_grad():
        for inputs1, inputs2, labels in tqdm(trainloader):
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
            outputs = model(inputs1, inputs2)
            prob=outputs[:,1]
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_prob.extend(prob.cpu().numpy())
    avg_loss = total_loss / len(testloader)
    accuracy = 100 * correct / total

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
    SEN = tp / (tp + fn)
    SPE = tn / (tn + fp)
    MCC = matthews_corrcoef(all_labels, all_predictions)
    AUROC = roc_auc_score(all_labels, all_prob)
    ACC = accuracy_score(all_labels, all_predictions)
    print(f'xunlian集平均损失avg_loss: {avg_loss}')
    print(f'xunlian准确率accuracy: {ACC}%')
    print(f'敏感度SEN: {SEN}')
    print(f'特异度SPE: {SPE}')
    print(f'MCC: {MCC}')
    print(f'AUROC: {AUROC}')

def train(model, trainloader, criterion,scheduler):
    # 训练循环
    num_epochs = 12  # 根据需要修改这个值
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs1, inputs2, labels in tqdm(trainloader):
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
            outputs = model(inputs1, inputs2)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(trainloader)
        accuracy = 100 * correct / total
        print(f'第 {epoch + 1} 轮,训练集上的平均损失: {avg_loss}')
        print(f'第 {epoch + 1} 轮,训练集上的准确率: {accuracy}%')

        save_dir = f'./model/resnet'
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        epoch_model_path = f'{save_dir}/model_epoch_{epoch + 1}.pth'

        torch.save(model.state_dict(), epoch_model_path)
        print(f'Model for epoch {epoch} saved to {epoch_model_path}')
        scheduler.step()  # 在每个epoch之后调整学习率



# 测试循环
def test(epoch,model, testloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_prob=[]
    all_labels = []
    with torch.no_grad():
        for inputs1, inputs2, labels in tqdm(testloader):
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
            outputs = model(inputs1, inputs2)
            prob=outputs[:,1]
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_prob.extend(prob.cpu().numpy())
    avg_loss = total_loss / len(testloader)
    accuracy = 100 * correct / total
    # 将数组中的每个元素转换为字符串，并用逗号连接
    data_str = ','.join(map(str, all_prob))
    # 打开一个文件来写入数据
    with open('./probs/resnet_test.txt', 'w') as f:
        f.write(data_str)
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
    SEN = tp / (tp + fn)
    SPE = tn / (tn + fp)
    MCC = matthews_corrcoef(all_labels, all_predictions)
    AUROC = roc_auc_score(all_labels, all_prob)
    ACC = accuracy_score(all_labels, all_predictions)
    print(f'测试集平均损失avg_loss: {avg_loss}')
    print(f'测试集准确率accuracy: {ACC}%')
    print(f'敏感度SEN: {SEN}')
    print(f'特异度SPE: {SPE}')
    print(f'MCC: {MCC}')
    print(f'AUROC: {AUROC}')


