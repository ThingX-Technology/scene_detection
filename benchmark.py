import json
from utils.oss_utils import *
import os
import random
import pandas as pd

tencent_oss = TencentOss()

scene_list = {
    'Bedroom': 5001,
    'kitchen': 5002,
    'living room': 5003,
    'Office': 5004,
    'factory': 5005,
    'School': 5006,
    'university': 5007,
    'Supermarkets': 5008,
    'malls': 5009,
    'Parks': 5010,
    'trails': 5011,
    'Restaurants': 5012,
    'coffee shops': 5013,
    'Gyms': 5014,
    'fitness studios': 5015,
    'Bus stops': 5016,
    'train stations': 5017,
    'Public libraries': 5018,
    'university libraries': 5019,
    'Hospitals': 5020,
    'clinics': 5021,
    'Churches': 5022,
    'mosques': 5023,
    'synagogues': 5024,
    'Rec centers': 5025,
    'town halls': 5026,
    'Banks': 5027,
    'ATMs': 5028,
    'Post offices': 5029,
    'Gas stations': 5030
}

def read_jsonl_file(file_path):
    all_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            all_data.append(data)
    return all_data

def load_train_dataset(text_oss_dir, audio_meta_path, save_path):   
        audio_oss_paths = []
        texts = []
        scenes = []
        scene_ids = []
        times = 100

        audio_oss_path_list = {}
        conversation_list = {}
        audio_df = pd.read_csv(audio_meta_path)
        for scene in scene_list.keys():
            filtered_df = audio_df.loc[audio_df['noise_scene'] == scene,  'mixed_output_file'].tolist()
            audio_oss_path_list[scene] = filtered_df
            conversation_list[scene] = []

        text_data_json = tencent_oss.list_directory(text_oss_dir)
        text_list, additional_info = text_data_json
        cnt = 0
        for item in text_list:
            if item['Size'] != '0':
                text_oss_path = item['Key']
                json_file_name = os.path.basename(item['Key'])
                json_save_path = os.path.join(save_path, json_file_name)
                tencent_oss.download_file(text_oss_path, json_save_path)
                sentences_datas = read_jsonl_file(json_save_path)
                for data in sentences_datas:
                    scene = data['scene'].strip()
                    if(type(data['conversation']) == str):
                        conversations = json.loads(data['conversation'])
                        for conversation in conversations:
                            conversation_list[scene].append(conversation['content'])
                    else:
                        for key, conversation in data['conversation'].items():
                            conversation_list[scene].append(conversation['content'])
                cnt += 1
                if cnt>100:
                    break

        for scene, scene_id in scene_list.items():
            for time in range(times):
                if audio_oss_path_list[scene]:
                    index = random.randrange(len(audio_oss_path_list[scene])) 
                    audio_oss_path = audio_oss_path_list[scene].pop(index)  
                    audio_oss_paths.append(audio_oss_path)
                else:
                    print("error, don't find audio")
                if conversation_list[scene]:
                    index = random.randrange(len(conversation_list[scene])) 
                    text = conversation_list[scene].pop(index) 
                    texts.append(text)
                else:
                    print(f"error, don't find conversation in scene-{scene}")
                scenes.append(scene)
                scene_ids.append(scene_id)

        df = pd.DataFrame({
            'audio_oss_path': audio_oss_paths,
            'conversation': texts,
            'scene': scenes,
            'scene_id': scene_ids
        })
        df.to_csv('benchmark_v1.csv', index=False)
        return None

if __name__ == "__main__":
    text_oss_dir = 'nuna_algorithm_simulation_data/boweihan/data/synthesis_data/dialog_data/'
    audio_meta_path = 'synthesis_oss_log_v2.csv'
    save_path = './dataset/train/audio/'

    load_train_dataset(text_oss_dir, audio_meta_path, save_path)