import time
import torch
from model.model import MLPClassifier
from sentence_transformers import SentenceTransformer
from audiodiffusion.audio_encoder import AudioEncoder
from utils.oss_utils import *
import torch.nn.functional as F
import pandas as pd

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

predict_dir = './dataset/predict/audio/'
benchmark_path = 'benchmark_v1.csv'

batch_size = 64  
num_classes = 30

audio_model_path = './records/audio+text+mlp/v3-0.5344-0.9024-0.8944-0.8333/audio_model.pth'
text_model_path = './records/audio+text+mlp/v3-0.5344-0.9024-0.8944-0.8333/text_model.pth'
mlp_model_path = './records/audio+text+mlp/v3-0.5344-0.9024-0.8944-0.8333/mlp_model.pth'
tencent_oss = TencentOss()

audio_model = AudioEncoder.from_pretrained("teticio/audio-encoder")
text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
mlp_model = MLPClassifier(100+384, hidden_sizes=[256, 128], output_size=num_classes)
audio_model.load_state_dict(torch.load(audio_model_path))
text_model.load_state_dict(torch.load(text_model_path))  
mlp_model.load_state_dict(torch.load(mlp_model_path))

def predict_single_sample(audio_path, sentence):
    embedding1 = audio_model.encode([audio_path])
    embedding2 = torch.tensor(text_model.encode([sentence]))
    combined_embedding = torch.cat((embedding1, embedding2), dim=-1)
    outputs = mlp_model(combined_embedding)

    probabilities = F.softmax(outputs, dim=1)
    index = torch.argmax(probabilities, dim=1).item()
    return scene_list[index], 5001 + index

def predict_muti_samples(benchmark_path, local_save_path):
    df = pd.read_csv(benchmark_path)
    audio_oss_paths = df['audio_oss_path'].tolist()
    conversations = df['conversation'].tolist()
    results = []
    for i in range(len(audio_oss_paths)):
        audio_oss_path = audio_oss_paths[i]
        local_audio_path = os.path.join(predict_dir, os.path.basename(audio_oss_path))
        tencent_oss.download_file(audio_oss_path, local_audio_path)

        scene, scene_id = predict_single_sample(local_audio_path, conversations[i])
        results.append({
            # 'audio_oss_path': audio_oss_paths[i],
            # 'conversation': conversations[i],
            'predicted_scene': scene,
            'predicted_scene_id': scene_id
        })
        os.remove(local_audio_path)
    return results

if __name__ == "__main__":
    start_time = time.time()

    result = predict_muti_samples(benchmark_path, predict_dir)
    print(result)

    end_time = time.time()
    elapsed_time = end_time - start_time 
    print(f"\nElapsed time: {elapsed_time:.6f} seconds")
