import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset.dataset import MyDataset
from model.model import MLPClassifier
from sentence_transformers import SentenceTransformer
from audiodiffusion.audio_encoder import AudioEncoder
from utils.oss_utils import *
import matplotlib.pyplot as plt

text_oss_path = 'nuna_algorithm_simulation_data/boweihan/data/synthesis_data/dialog_data/'
audio_meta_path = 'synthesis_oss_log_v2.csv'
train_dir = './dataset/train/audio/'

num_epochs = 300
batch_size = 64  
num_classes = 30
lr = 0.0001

audio_model_path = 'audio_model.pth'
text_model_path = 'text_model.pth'
mlp_model_path = 'mlp_model.pth'
tencent_oss = TencentOss()

def train():
    dataset = MyDataset([], [], [])
    dataset.load_train_dataset(text_oss_path, audio_meta_path, train_dir)
    # 划分训练集和验证集
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    validation_size = dataset_size - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(validation_dataset)}")

    audio_model = AudioEncoder.from_pretrained("teticio/audio-encoder")
    text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    mlp_model = MLPClassifier(100+384, hidden_sizes=[256, 128], output_size=num_classes)
    criterion = nn.CrossEntropyLoss()
    

    # 消融实验
    # for param in audio_model.parameters():
    #     param.requires_grad = False
    # for param in text_model.parameters():
    #     param.requires_grad = False
    # optimizer = optim.Adam(mlp_model.parameters(), lr=lr)
    # optimizer = optim.Adam(list(audio_model.parameters()) + list(mlp_model.parameters()), lr=lr)
    # optimizer = optim.Adam(list(text_model.parameters()) + list(mlp_model.parameters()), lr=lr)
    optimizer = optim.Adam(list(audio_model.parameters()) + list(text_model.parameters()) + list(mlp_model.parameters()), lr=lr)

    # audio_model.load_state_dict(torch.load(audio_model_path))
    # text_model.load_state_dict(torch.load(text_model_path))
    # mlp_model.load_state_dict(torch.load(mlp_model_path))

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # 训练模型
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # 训练阶段
        audio_model.train()
        text_model.train()
        mlp_model.train()

        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for audio_path, sentence, labels in train_dataloader:
            optimizer.zero_grad()

            embedding1 = audio_model.encode(audio_path)
            embedding2 = torch.tensor(text_model.encode(sentence))
            combined_embedding = torch.cat((embedding1, embedding2), dim=-1)
            outputs = mlp_model(combined_embedding)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_accuracy = correct_train / total_train
        avg_train_loss = running_loss / len(train_dataloader)
        train_accs.append(train_accuracy)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        audio_model.eval()
        text_model.eval()
        mlp_model.eval()

        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for audio_path, sentence, labels in validation_dataloader:
                embedding1 = audio_model.encode(audio_path)
                embedding2 = torch.tensor(text_model.encode(sentence))
                combined_embedding = torch.cat((embedding1, embedding2), dim=-1)
                outputs = mlp_model(combined_embedding)

                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_accuracy = correct_val / total_val
        avg_val_loss = val_loss / len(validation_dataloader)
        val_accs.append(val_accuracy)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(audio_model.state_dict(), audio_model_path)
            torch.save(text_model.state_dict(), text_model_path)
            torch.save(mlp_model.state_dict(), mlp_model_path)
            print(f"model saved successfully")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png') 

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_accs, label='Train Accuracy', color='blue')
    plt.plot(range(1, num_epochs + 1), val_accs, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_curve.png')   

if __name__ == "__main__":
    start_time = time.time()

    train()

    end_time = time.time()
    elapsed_time = end_time - start_time 
    print(f"\nElapsed time: {elapsed_time:.6f} seconds")
