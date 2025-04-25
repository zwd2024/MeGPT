import os
import torch.nn as nn
import torch
from pprint import pprint
from configs.config import parser
from dataset.data_module import DataModule
from lightning_tools.callbacks import add_callbacks
from models.R2GenGPT import R2GenGPT
from lightning.pytorch import seed_everything
import lightning.pytorch as pl
from sklearn.cluster import KMeans
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image
import numpy as np

# 生成提示词
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        # 使用预训练的ResNet模型
        self.model = models.resnet50(pretrained=True)
        # 替换最后的全连接层以适应新的分类任务
        self.model.fc = nn.Identity()  # 使用Identity层，直接输出特征

    def forward(self, x):
        return self.model(x)
# 定义数据加载器
class IU_XRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.patient_folders = [os.path.join(root_dir, folder) for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]

    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, idx):
        patient_folder = self.patient_folders[idx]
        frontal_image_path = os.path.join(patient_folder, '0.png')
        lateral_image_path = os.path.join(patient_folder, '1.png')

        frontal_image = Image.open(frontal_image_path).convert('RGB')
        lateral_image = Image.open(lateral_image_path).convert('RGB')

        if self.transform:
            frontal_image = self.transform(frontal_image)
            lateral_image = self.transform(lateral_image)

        # 将正面和侧面的图像组合成一个张量
        combined_image = torch.cat((frontal_image, lateral_image), dim=1)

        return combined_image
# 定义数据预处理
def get_data_loader(root_dir, batch_size=1, shuffle=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = IU_XRayDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# 使用KMeans聚类生成标签
def generate_labels(features, n_clusters=12):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels

# 推理函数
def classify_images(dataloader, classifier, n_clusters=12):
    all_features = []
    for combined_image in dataloader:
        # 提取特征
        with torch.no_grad():
            features = classifier(combined_image)
            all_features.append(features.cpu().numpy())

    # 将所有特征拼接成一个数组
    all_features = np.concatenate(all_features, axis=0)

    # 使用KMeans聚类生成标签
    labels = generate_labels(all_features, n_clusters=n_clusters)

    return labels

def train_classifier_model(dataloader, classifier, device, num_epochs=10, n_clusters=12):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.0001)
    classifier.train()  # 设置模型为训练模式
    for epoch in range(num_epochs):
        running_loss = 0.0
        all_features = []  # 用于存储所有特征
        all_labels = []  # 用于存储所有标签

        # 提取特征
        for batch in dataloader:
            inputs = batch  # 只有输入数据
            inputs = inputs.to(device)  # 将数据移动到指定设备

            with torch.no_grad():
                features = classifier(inputs)  # 提取特征
                all_features.append(features.cpu().numpy())
        # 将所有特征拼接成一个数组
        all_features = np.concatenate(all_features, axis=0)
        # 使用 K-Means 聚类生成标签
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(all_features)

        # 将标签转换为张量
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        # 训练模型
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch  # 只有输入数据
            inputs = inputs.to(device)  # 将数据移动到指定设备

            optimizer.zero_grad()  # 清空梯度
            outputs = classifier(inputs)  # 前向传播

            # 获取当前批次的标签
            batch_labels = labels[batch_idx * dataloader.batch_size:(batch_idx + 1) * dataloader.batch_size]
            loss = criterion(outputs, batch_labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')

# 保存模型的函数
def save_model(model, path):
    torch.save(model.state_dict(), path)

def train(args):
    dm = DataModule(args)
    callbacks = add_callbacks(args)

    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval = args.val_check_interval,
        limit_val_batches = args.limit_val_batches,
        max_epochs = args.max_epochs,
        num_sanity_val_steps = args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks["callbacks"], 
        logger=callbacks["loggers"]
    )

    if args.ckpt_file is not None:
        #print("ckpt_file:", args.ckpt_file)
        model = R2GenGPT.load_from_checkpoint(args.ckpt_file, strict=False)
    else:
        model = R2GenGPT(args)

    if args.test:
        #print("model:", model)
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        #print("model:", model)
        trainer.fit(model, datamodule=dm)

def main():
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # classifier = ImageClassifier().to(device)
    # dataloader = get_data_loader('data/iu_xray/images/', batch_size=1, shuffle=True)
    # train_classifier_model(dataloader, classifier, device, num_epochs=10)
    # save_model(classifier, '/home/zhengweidong/projects/R2GenGPT/models/classifier_model.pth')
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(42, workers=True)
    train(args)


if __name__ == '__main__':
    #补充
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 或者 "true" 根据你的需求
    main()