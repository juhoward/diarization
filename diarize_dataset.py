from pydub import AudioSegment
from pyannote.audio import Pipeline

from pathlib import Path
import locale
import os
import re
import whisper
import torch
import json
from datetime import timedelta
from glob import glob
from diarization_utils import *

datadir = '/home/digitalopt/proj/datasets/Recordings/'
output_root = "/home/digitalopt/proj/diarization/recordings/" #@param {type:"string"}
output_root = str(Path(output_root))
audio_title = "Sample Order Taking" #@param {type:"string"}

spacermilli = 2000
spacer = AudioSegment.silent(duration=spacermilli)
# access_token = "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"
# pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token= (access_token) or True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pipeline.to(device)
# # identify paths to all mp3 recordings
# recordings = glob(datadir + '*/*.mp3')
# print(f'found {len(recordings)} recordings\nexample:{recordings[0]}')
# for r in recordings[:1]:
#     audio = AudioSegment.from_mp3(r) 
#     audio = spacer.append(audio, crossfade=0)
#     output_path = output_root + '/' + r.split('/')[-2] + '/' + r.split('/')[-1].split('.')[0]
#     Path(output_path).mkdir(parents=True, exist_ok=True)
#     # write padded .wav file
#     audio.export(output_path + '/input_prep.wav', format='wav')
#     DEMO_FILE = {'uri': 'blabla', 'audio': 'input_prep.wav', 'num_speakers':3}
#     dz = pipeline(DEMO_FILE)

#     # write diarization file
#     with open(output_path + "/diarization.txt", "w") as text_file:
#         text_file.write(str(dz))
# transcription with whisper
model = whisper.load_model('large', device = device)
output_root = "/home/digitalopt/proj/diarization/recordings/"
dz_paths = glob(output_root + '*/*/diarization.txt')
print(f'found {len(dz_paths)} diarizations')
for dpath in dz_paths[1:]:
    print(f'reading {dpath}')
    dzs = open(dpath).read().splitlines()
    groups = []
    g = []
    lastend = 0
    for d in dzs:   
        if g and (g[0].split()[-1] != d.split()[-1]): #same speaker
            groups.append(g)
            g = []
        
        g.append(d)
        
        end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=d)[1]
        end = millisec(end)
        if (lastend > end):     #segment engulfed by a previous segment
            groups.append(g)
            g = [] 
        else:
            lastend = end
        if g:
            groups.append(g)
    # if duplicate groups are present, remove them
    groups = deduplicate(groups)
    fpath = '/'.join(dpath.split('/')[:-1])
    print(f'reading wav file: {fpath + "/input_prep.wav"}')
    audio = AudioSegment.from_wav(fpath + "/input_prep.wav")
    gidx = -1
    # segment input_prep file by speaking groups
    for g in groups:
        start = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
        end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[-1])[1]
        start = millisec(start) #- spacermilli
        end = millisec(end)  #- spacermilli
        gidx += 1
        audio[start:end].export(fpath + '/' + str(gidx) + '.wav', format='wav')

    # save transcriptions as json files
    for i in range(len(groups)):
        audiof = fpath + '/' + str(i) + '.wav'
        result = model.transcribe(audio=audiof, language='en', word_timestamps=True)#, initial_prompt=result.get('text', ""))
        with open(fpath + '/' + str(i)+'.json', "w") as outfile:
            json.dump(result, outfile, indent=4)
    speakers = {'SPEAKER_00':('SPEAKER_00', '#e1ffc7', 'darkgreen'), 'SPEAKER_01':('SPEAKER_01', 'white', 'darkorange') }
    def_boxclr = 'white'
    def_spkrclr = 'orange'
    html = list(preS)
    txt = list("")
    gidx = -1
    for g in groups:  
        shift = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
        shift = millisec(shift) - spacermilli #the start time in the original video
        shift = max(shift, 0)
        
        gidx += 1
        
        captions = json.load(open(fpath + '/' + str(gidx) + '.json'))['segments']

        if captions:
            speaker = g[0].split()[-1]
            boxclr = def_boxclr
            spkrclr = def_spkrclr
            if speaker in speakers:
                speaker, boxclr, spkrclr = speakers[speaker] 
        
            html.append(f'<div class="e" style="background-color: {boxclr}">\n');
            html.append('<p  style="margin:0;padding: 5px 10px 10px 10px;word-wrap:normal;white-space:normal;">\n')
            html.append(f'<span style="color:{spkrclr};font-weight: bold;">{speaker}</span><br>\n\t\t\t\t')
            
            for c in captions:
                start = shift + c['start'] * 1000.0 
                start = start / 1000.0   #time resolution ot youtube is Second.            
                end = (shift + c['end'] * 1000.0) / 1000.0      
                txt.append(f'[{timeStr(start)} --> {timeStr(end)}] [{speaker}] {c["text"]}\n')

                for i, w in enumerate(c['words']):
                    if w == "":
                        continue
                    start = (shift + w['start']*1000.0) / 1000.0        
                    #end = (shift + w['end']) / 1000.0   #time resolution ot youtube is Second.  
                    html.append(f'<a href="#{timeStr(start)}" id="{"{:.1f}".format(round(start*5)/5)}" class="lt" onclick="jumptoTime({int(start)}, this.id)">{w["word"]}</a><!--\n\t\t\t\t-->')
                #html.append('\n')      
                html.append('</p>\n')
                html.append(f'</div>\n')

            html.append(postS)

            with open(f"{fpath}/capspeaker.txt", "w", encoding='utf-8') as file:
                s = "".join(txt)
                file.write(s)
                print(f'captions saved to /{fpath}/capspeaker.txt:')
            with open(f"{fpath}/capspeaker.html", "w", encoding='utf-8') as file:    #TODO: proper html embed tag when video/audio from file
                s = "".join(html)
                file.write(s)
                print(f'html saved to {fpath}/capspeaker.html:')