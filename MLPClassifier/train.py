import subprocess
import os
from moviepy import AudioFileClip
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model.model import MLPClassifier

from threading import Lock

lock = Lock()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 基于sherpa-onnx的event分类模型参数
executable_path = "./sherpa-onnx/bin/sherpa-onnx-offline-audio-tagging" # sherpa-onnx可执行文件
sherpa_onnx_model_path = "./sherpa-onnx/sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.int8.onnx" # 模型架构和参数文件
sherpa_onnx_labels_path = "./sherpa-onnx/sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/class_labels_indices.csv" # sherpa-onnx分类标签文档
# 训练集/验证集及其label地址
train_audio_dir = './dataset/train/audio'
train_meta_path = './dataset/train/meta.txt'
validation_audio_dir = './dataset/validate/audio'
validation_meta_path = './dataset/validate/meta.txt'
# 分窗大小
window_duration = 5
match_threshold = [0.8, 0.2]
num_epochs = 100
# 设置模型保存路径
model_path = 'best_model.pth'

# 数据集的label到四个场景分类的映射: 0-working, 1-commuting, 2-entertainment, 3-home
label_scene_map = {
    'bus':1,
    'cafe/restaurant':2,
    'car':1,
    # 'city_center':,
    # 'forest_path':,
    # 'grocery_store':,
    'home':3,
    # 'library':,
    'metro_station':1,
    'office':0,
    # 'residential_area':,
    'train':1,
    'tram':1,
    # Lakeside beach (outdoor)
    # Urban park (outdoor)
}

# 调用sherpa-onnx判断音频对应的event
def find_match_event(audio_part_path):
    command = [
        executable_path,
        f"--zipformer-model={sherpa_onnx_model_path}",
        f"--labels={sherpa_onnx_labels_path}",
        audio_part_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        events = []
        for line in result.stderr.strip().split('\n'):
            if 'AudioEvent(name="' in line:
                parts = line.split('AudioEvent(name="')
                if len(parts) > 1:
                    name_part = parts[1].split('"', 1)[0]
                    prob_part = parts[1].split('prob=', 1)[1].split(')')[0]
                    events.append((name_part, float(prob_part)))
        events = sorted(events, key=lambda x: x[1], reverse=True)

        match_events = []
        for event in events:
            if(event[1]>=match_threshold[0]):
                match_events.append(event)
        if(len(match_events)<2):
            match_events = events[:1]
            if events[1][1] >= match_threshold[1]:
                match_events.append(events[1])

        return match_events

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the command: {e}")
        print(f"Error output: {e.stderr}")
        return []

# 存储音频切片，并判断event类型
def process_audio_segment(audio_segment):
    try:
        temp_audio_file = f"temp_audio_segment_{uuid.uuid4()}.wav"
        with lock:
            audio_segment.write_audiofile(temp_audio_file, codec='pcm_s16le')
        match_events = find_match_event(temp_audio_file)
        return match_events
    except Exception as e:
        logger.error(f"Error writing audio segment: {e}")
        return []
    finally:
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)  

# 对音频分窗处理，统计该音频中各event数量
def process_audio(audio_path, event_index, window_duration=5):
    audio = AudioFileClip(audio_path)
    audio_duration = audio.duration 
    try:
        all_match_events = []
        audio_segments = []
        for start_time in range(0, int(audio_duration), window_duration):
            end_time = min(start_time + window_duration, audio_duration)
            audio_segments.append(audio.subclipped(start_time, end_time))

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_audio_segment, audio_segment)for audio_segment in audio_segments]
            for future in as_completed(futures):
                all_match_events.extend(future.result())

        ret = [0 for i in range(len(event_index))]
        for event in all_match_events:
            ret[event_index[event[0]]] += 1
        for i in range(len(event_index)):
            if ret[i] != 0:
                ret[i] = ret[i] / math.ceil(int(audio_duration)/window_duration) * math.ceil(180/window_duration)
        return ret
    except Exception as e:
        logger.error(f"Error processing video {audio_path}: {e}")

# 读入训练集的label字典
def load_labels_to_dict(meta_file_path):
    labels_dict = {}
    try:
        with open(meta_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    filename, label = parts
                    if label in label_scene_map:
                        labels_dict[filename] = label
        return labels_dict
    except FileNotFoundError:
        print(f"Error: The file {meta_file_path} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def load_event_types():
    df = pd.read_csv(sherpa_onnx_labels_path, header=None)
    return dict(zip(df.iloc[1:,2], df.iloc[1:,0].astype(int)))

def load_dataset(audio_dir, meta_path, event_index):
    inputs = []
    labels = []
    find_labels = load_labels_to_dict(meta_path)
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if f"audio/{file}" in find_labels:
                inputs.append(process_audio(os.path.join(root, file), event_index, window_duration))
                labels.append(label_scene_map[find_labels[f"audio/{file}"]])
                if(len(labels)==80):
                    break
    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)
    return inputs, labels


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_path):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_accuracy = correct_train / total_train
        avg_train_loss = running_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_accuracy = correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        
        # 如果验证损失有改进，则保存模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"model saved successfully--{model_path}")

def train():
    event_index = load_event_types()

    # 模拟一些数据
    input_size = len(event_index) 
    batch_size = 16  
    num_classes = 4 

    # 创建模拟数据集
    train_inputs, train_labels = load_dataset(train_audio_dir, train_meta_path, event_index)
    validation_inputs, validation_labels = load_dataset(validation_audio_dir, validation_meta_path, event_index)
    print(f"{train_inputs.size()}+{train_labels.size()}+{validation_inputs.size()}+{validation_labels.size()}")

    # 创建DataLoader
    train_dataset = TensorDataset(train_inputs, train_labels)
    val_dataset = TensorDataset(validation_inputs, validation_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = MLPClassifier(input_size, hidden_sizes=[256, 128], output_size=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)



    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_path)

if __name__ == "__main__":
    start_time = time.time()

    train()

    end_time = time.time()
    elapsed_time = end_time - start_time 
    print(f"\nElapsed time: {elapsed_time:.6f} seconds")
