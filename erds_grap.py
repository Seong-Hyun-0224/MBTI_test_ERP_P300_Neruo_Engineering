import os
from typing import List
from datetime import datetime

import numpy as np
import pandas as pd

from src.task import grap_task
from src.analysis import AnalyzeEEG
from src.plot import PlotEEG

def custom(
    channels: List,
    value_dir: str,
    data_list: List,
    result_dir: str
):
    
    name_list= ['Mu', 'Beta', 'Gamma']  
    
    for i in range(len(data_list)):
        data = data_list[i]
        integral_b_channel_0 = []
        integral_b_channel_1 = []

        for channel in range(len(channels)):
            if not os.path.exists(f"{value_dir}/erds"):
                os.makedirs(f"{value_dir}/erds")
            erds_avg_evoked_list = data['erds_avg_evoked_list']
            erds_times_list = data['erds_times_list']
            erds_df = pd.DataFrame(
                zip(erds_avg_evoked_list[0][channel], erds_times_list[0]),
                columns=["avg_evoked", "order"]
            )

            erds_df.to_csv(
                f"{value_dir}/erds/EEG_Fp{channel+1}_{name_list[i]}.csv",
                encoding="utf-8-sig",
                index=False,
            )

            start_time = 0.0
            for j in range(len(erds_times_list[0])):
                if erds_times_list[0][j] == 0.0:
                    start_time = j
                
                if erds_times_list[0][j] == 1.0:
                    end_time = j

        
            for k in range(len(erds_times_list[0])):
                if erds_times_list[0][k] == 0.0:
                    start_time2 = k
                if erds_times_list[0][k] == 0.4:
                    end_time2 = k
            print(start_time2,end_time2)
            max=start_time2
            min=end_time2
            for o in range(len(erds_avg_evoked_list[0][channel][start_time2:end_time2])):
                if erds_avg_evoked_list[0][channel][o+start_time2]>erds_avg_evoked_list[0][channel][max]:
                    max=o+start_time2
                if erds_avg_evoked_list[0][channel][o+start_time2]<erds_avg_evoked_list[0][channel][min]:
                    min=o+start_time2
            print("최댓값,최솟값",max,min)
            sec=erds_times_list[0][int(min+max)]/2
            print("시간",sec)

        for channel in range(len(channels)):
            if not os.path.exists(f"{value_dir}/erds_whole"):
                os.makedirs(f"{value_dir}/erds_whole")
            erds_whole_avg_evoked_list = data['erds_whole_avg_evoked_list']
            erds_whole_times_list = data['erds_whole_times_list']
            erds_whole_df = pd.DataFrame(
                zip(erds_whole_avg_evoked_list[0][channel], erds_whole_times_list[0]),
                columns=["avg_evoked", "order"]
            )
            erds_whole_df.to_csv(
                f"{value_dir}/erds_whole/EEG_Fp{channel+1}_{name_list[i]}.csv",
                encoding="utf-8-sig",
                index=False,
            )
            
        erds_avg_evoked_list = data['erds_avg_evoked_list']
        erds_times_list = data['erds_times_list']
        erds_whole_avg_evoked_list = data['erds_whole_avg_evoked_list']
        erds_whole_times_list = data['erds_whole_times_list']

        plot_eeg = PlotEEG(
            channels=channels,
            result_dir=result_dir,
            is_show=False,
            is_save=True,
            eeg=data['eeg'],
            eeg_times=data['eeg_times'],
            eeg_filename="eeg_raw",
        )
        plot_eeg.plot_eeg()
        plot_eeg.plot_electrode(
            erds_avg_evoked_list[0],
            erds_times_list[0],
            filename=f"grap_erds_electrode_{name_list[i]}",
        )
        plot_eeg.plot_electrode(
            erds_whole_avg_evoked_list[0],
            erds_whole_times_list[0],
            filename=f"grap_erds_whole_electrode_{name_list[i]}",
        )
    

def erds_grap(
    screen_width: int,
    screen_height: int,
    fs: int,
    channels: List,
    isi: int,
    obstacle_playing_time: int,
    background_path: str,
    image_folder: str,
    num_trials: int,
    num_images: int,
    event_save_path: str,
    result_dir: str,
    value_dir: str,
):
    today = str(datetime.now().date())
    if not os.path.exists(f"./data/{today}"):
        os.makedirs(f"./data/{today}")
    if not os.path.exists(f"./event/{today}"):
        os.makedirs(f"./event/{today}")

    grap_task(
        screen_width=screen_width,
        screen_height=screen_height,
        isi=isi,
        obstacle_playing_time=obstacle_playing_time,
        background_path=background_path,
        image_folder=image_folder,
        num_trials=num_trials,
        num_images=num_images,
        event_save_path=f"{event_save_path}/{today}",
    )

    rawdata_folders = os.listdir("C:/MAVE_RawData")

    text_file_name = f"C:/MAVE_RawData/{rawdata_folders[-1]}/Rawdata.txt"
    data_df = pd.read_csv(text_file_name, delimiter="\t")

    record_start_time = data_df.iloc[0, 0]
    hour = str(record_start_time).split(":")[0]
    min = str(record_start_time).split(":")[1]
    sec = str(record_start_time).split(":")[2].split(".")[0]

    data_df = data_df[channels]
    data_file_path = f"./data/{today}/Rawdata_{hour}.{min}.{sec}.csv"
    data_df.to_csv(data_file_path, index=False)

    event_paths = os.listdir(f"./event/{today}")
    event_file = f"./event/{today}/{event_paths[-1]}"

    analyze_eeg = AnalyzeEEG(channels=channels, fs=fs)

    
    data_list = analyze_eeg.custom_anaylyze(
        eeg_filename=data_file_path,
        event_filename=event_file,
        result_dir=result_dir
    )

    custom(
        channels=channels,
        value_dir=value_dir,
        data_list=data_list,
        result_dir=result_dir
    )


if __name__ == "__main__":
    import argparse
    import ast

    def parse_list(string: str):
        try:
            parsed_list = ast.literal_eval(string)
            if isinstance(parsed_list, list):
                return parsed_list
            else:
                raise argparse.ArgumentTypeError("Invalid list format")
        except (ValueError, SyntaxError):
            raise argparse.ArgumentTypeError("Invalid list format")

    parser = argparse.ArgumentParser(
        description="Insert arguments for function of erds grap"
    )
    parser.add_argument(
        "--screen_width",
        type=int,
        default=1920,
        help="Set screen width of grap task",
    )
    parser.add_argument(
        "--screen_height",
        type=int,
        default=1080,
        help="Set screen height of grap task",
    )
    parser.add_argument(
        "--fs", type=int, default=256, help="Get resolution of EEG device"
    )
    parser.add_argument(
        "--channels",
        type=parse_list,
        default="['EEG_Fp1', 'EEG_Fp2']",
        help="Get channels of EEG device",
    )
    parser.add_argument(
        "--isi",
        type=int,
        default=7000,
        help="Set inter-stimulus interval of grap task",
    )
    parser.add_argument(
        "--obstacle_playing_time",
        type=int,
        default=1500,
        help="Set obstacle playing time of grap task",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./images",
        help="Get image data path to use in the task",
    )
    parser.add_argument(
        "--backgrounds_order",
        type=int,
        default=1,
        help="Set order of upper clothes to use in the task",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=3,
        help="Set number of trials to use in the task",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=3,
        help="Set number of clothes to use in the task",
    )
    parser.add_argument(
        "--event_save_path",
        type=str,
        default="./event",
        help="Set a record of events file saving path",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./plot",
        help="Set a EEG, ERDS plots saving path",
    )
    parser.add_argument(
        "--value_dir",
        type=str,
        default="./value",
        help="Set a EEG, ERDS values saving path",
    )
    parser.add_argument(
        "--dir_num",
        type=int,
        default=0,
        help="Set a EEG, ERDS plots and values detailed saving path",
    )
    args = parser.parse_args()

    erds_grap(
        screen_width=args.screen_width,
        screen_height=args.screen_height,
        fs=args.fs,
        channels=args.channels,
        isi=args.isi,
        obstacle_playing_time=args.obstacle_playing_time,
        background_path=f"{args.image_path}/backgrounds/B{args.backgrounds_order}.mp4",
        image_folder=f"{args.image_path}/obstacles",
        num_trials=args.num_trials,
        num_images=args.num_images,
        event_save_path=f"{args.event_save_path}",
        result_dir=f"{args.result_dir}/grap/{args.dir_num}",
        value_dir=f"{args.value_dir}/grap/{args.dir_num}",
    )