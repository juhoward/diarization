from pathlib import Path
from yt_dlp import YoutubeDL
import locale
import os

# video_url = "https://youtu.be/hpZFJctBUHQ"
# video_url ="https://youtu.be/YqL6IMGE5os"
video_url = "https://youtu.be/kcULvDBJOVs"
output_path = "./content/transcript/" #@param {type:"string"}
output_path = str(Path(output_path))
audio_title = "Sample Order Taking" #@param {type:"string"}

locale.getpreferredencoding = lambda: "UTF-8"
Path(output_path).mkdir(parents=True, exist_ok=True)
video_title = ""
video_id = ""


with YoutubeDL() as ydl: 
    info_dict = ydl.extract_info(video_url, download=False)
    video_title = info_dict.get('title', None)
    video_id = info_dict.get('id', None)
    print("Title: " + video_title) # <= Here, you got the video title

os.system(f"yt-dlp -xv --ffmpeg-location ffmpeg-master-latest-linux64-gpl/bin --audio-format wav  -o {str(output_path) + '/'}input.wav -- {video_url}")

