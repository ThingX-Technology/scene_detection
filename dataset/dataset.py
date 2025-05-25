from torch.utils.data import Dataset
import json
from utils.oss_utils import *
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

scene_list = [
    'Bedroom',
    'kitchen',
    'living room',
    'Office',
    'factory',
    'School',
    'university',
    'Supermarkets',
    'malls',
    'Parks',
    'trails',
    'Restaurants',
    'coffee shops',
    'Gyms',
    'fitness studios',
    'Bus stops',
    'train stations',
    'Public libraries',
    'university libraries',
    'Hospitals',
    'clinics',
    'Churches',
    'mosques',
    'synagogues',
    'Rec centers',
    'town halls',
    'Banks',
    'ATMs',
    'Post offices',
    'Gas stations'
]

def read_jsonl_file(file_path):
    all_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            all_data.append(data)
    return all_data

def get_scene_encoding(scene):
    for i, type in enumerate(scene_list):
        if type == scene:
            return torch.tensor(i, dtype=torch.long)
    print(f"error! fail to find the scene: \"{scene}\"")
    return torch.tensor(i, dtype=torch.long)


class MyDataset(Dataset):
    def __init__(self, audio_paths, sentences, labels):
        self.audio_paths = audio_paths
        self.sentences = sentences
        self.labels = labels
        self.tencent_oss = TencentOss()

    def load_train_dataset(self, text_oss_dir, audio_meta_path, save_path):   
        audio_paths = []
        sentences = []
        labels = []

        audio_path_list = {}
        df = pd.read_csv(audio_meta_path)
        for scene in scene_list:
            filtered_df = df.loc[df['noise_scene'] == scene, 'mixed_output_file'].tolist()
            audio_path_list[scene] = filtered_df

        text_data_json = self.tencent_oss.list_directory(text_oss_dir)
        text_list, additional_info = text_data_json
        for item in text_list:
            if item['Size'] != '0':
                text_oss_path = item['Key']
                json_file_name = os.path.basename(item['Key'])
                json_save_path = os.path.join(save_path, json_file_name)
                self.tencent_oss.download_file(text_oss_path, json_save_path)
                sentences_datas = read_jsonl_file(json_save_path)
                for data in sentences_datas:
                    conversations = json.loads(data['conversation'])
                    scene = data['scene'].strip()
                    for conversation in conversations:
                        oss_audio_path = audio_path_list[scene].pop(0)
                        local_audio_path = os.path.join(save_path, os.path.basename(oss_audio_path))
                        self.tencent_oss.download_file(oss_audio_path, local_audio_path)
                        audio_paths.append(local_audio_path)
                        sentences.append(conversation['content'])
                        labels.append(get_scene_encoding(scene))
                        # self.audio_paths = audio_paths
                        # self.sentences = sentences
                        # self.labels = labels  
                        # return None      
                os.remove(json_save_path)
                # return None
        self.audio_paths = audio_paths
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        return self.audio_paths[idx], self.sentences[idx], self.labels[idx]
    

if __name__ == "__main__":
    text_oss_path = 'nuna_algorithm_simulation_data/boweihan/data/synthesis_data/dialog_data/'
    audio_meta_path = 'synthesis_oss_log_v2.csv'
    train_dir = './dataset/train/audio/'

    dataset = MyDataset([], [], [])
    dataset.load_train_dataset(text_oss_path, audio_meta_path, train_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)