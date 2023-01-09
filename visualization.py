import argparse
import os
from utils import get_visualization

parser = argparse.ArgumentParser(description="run")

parser.add_argument('--path_video', type=str, default='video/', help='Path to all videos')
parser.add_argument('--path_report', type=str, default='report/', help='Path to save the report')
parser.add_argument('--path_save_video', type=str, default='result_videos/', help='Path to save the result videos')
parser.add_argument('--type_lips_model', type=int, default=0, help='0: 6 Emotion, 1: VALENCE, 2: BINARY')


args = parser.parse_args()

def pred_all_video():
    path_all_videos = os.listdir(args.path_video)
    if not os.path.exists(args.path_save_video):
        os.makedirs(args.path_save_video)
    for id, cr_path in enumerate(path_all_videos):
        print('{}/{}'.format(id+1, len(path_all_videos)))
        print('Name video: ', os.path.basename(cr_path))
        name_video = os.path.basename(cr_path)
        name_report = os.path.basename(cr_path)[:-4] + '.csv'
        
        detect = get_visualization.VideoCamera(path_video=os.path.join(args.path_video, cr_path),
                                            path_report=os.path.join(args.path_report, name_report),
                                            path_save=os.path.join(args.path_save_video, name_video[:-4]),
                                            type_lips_model=args.type_lips_model)
        detect.get_video()
        print('Ressult saved in: ', os.path.join(args.path_save_video,name_video[:-4] + '.mp4'))
        
if __name__ == "__main__":
    pred_all_video()