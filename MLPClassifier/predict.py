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
# 测试集及其label地址
predict_audio_dir = './dataset/predict/audio'
model_path = 'best_model.pth'

scenes = ['working', 'commuting', 'entertainment', 'home']
# 分窗大小
window_duration = 5
match_threshold = [0.8, 0.2]

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

def load_event_types():
    df = pd.read_csv(sherpa_onnx_labels_path, header=None)
    return dict(zip(df.iloc[1:,2], df.iloc[1:,0].astype(int)))

def load_dataset(audio_dir, event_index):
    inputs = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            inputs.append(process_audio(os.path.join(root, file), event_index, window_duration))
    inputs = torch.tensor(inputs)
    return inputs

def predict():
    event_index = load_event_types()

    input_size = len(event_index) 
    batch_size = 16  
    num_classes = 4  
    # 创建模拟数据集
    test_inputs = load_dataset(predict_audio_dir, event_index)  # 100个样本

    # 创建DataLoader
    test_dataset = TensorDataset(test_inputs, torch.zeros_like(test_inputs[:, 0])) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = MLPClassifier(input_size, hidden_sizes=[256, 128], output_size=num_classes)

    # 加载最佳模型
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    all_predictions = []
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
    
    cnt = 0
    print("\npredict result:")
    for root, dirs, files in os.walk(predict_audio_dir):
        for file in files:
            scene_index = int(all_predictions[cnt])
            scene = scenes[scene_index]
            print(f"\"{file}\": {scene}")
            cnt += 1

    return all_predictions


if __name__ == "__main__":
    start_time = time.time()

    predict()

    end_time = time.time()
    elapsed_time = end_time - start_time  
    print(f"\nElapsed time: {elapsed_time:.6f} seconds")