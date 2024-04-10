from collections import defaultdict
import spacy
import spacy_transformers

class CaptionCleaner(object):
    def __init__(self, NER=True):
        self.load_NER_model() if NER else None
    '''
    FILE OPERATIONS.
    '''
    def read_captions(self, path):
        self.fname = path
        captions =  open(path).read().splitlines()
        captions = [i.strip() for i in captions]
        if captions[-1] == '':
            captions = captions[:-2]
        print(f'captions read. length: {len(captions)}')
        return captions
    
    def write_captions(self, txt, fname):
        with open(fname, 'w') as file:
            output = '\n'.join(txt)
            file.write(output)
    '''
    NAME SCRUBBING.
    '''
    def load_NER_model(self, model="en_core_web_trf"):
        # to optimize for speed, use "en_core_web_sm" model.
        self.NER = spacy.load(model)
        print(f'spacy model {model} loaded')

    def remove_names(self, txt):
        for idx in range(len(txt)):
            names = self.detect_names(txt[idx])
            for name in names:
                txt[idx] = txt[idx].replace(name, "[NAME]")
        return txt

    def detect_names(self, line):
        document = self.NER(line)
        return [ent.text for ent in document.ents if ent.label_ == 'PERSON']
    '''
    CAPTION CLEANING
    '''
    def remove_time(self, txt):
        output = []
        for line in txt:
            output.append(line[32:])
        return output

    def deduplicate(self, txt:list):
        prev = ''
        nodupes = []
        for line in txt:
            if line != prev:
                nodupes.append(line)
                prev = line
        return nodupes

    def condense(self, txt):
        condensed = []
        old_speaker = ''
        line = ''
        for idx in range(len(txt)):
            speaker = txt[idx][1:11]
            if old_speaker == speaker:
                line += txt[idx][12:]
            else:
                if line != '':
                    condensed.append(line)
                old_speaker = speaker
                line = f'[{speaker}]'
                line += txt[idx][12:]
        print(f'removed {len(txt) - len(condensed)} lines.')
        return condensed

    def clean(self, txt, full_clean=True):
        if full_clean:
            txt = self.remove_time(txt)
            txt = self.deduplicate(txt)
            txt = self.condense(txt)
        self.speaker_cnt(txt)
        if len(list(self.speakers.keys())) > 1:
            txt = self.remap_speaker_names(txt, for_train=True)
            if self.NER:
                print('removing names...')
                self.remove_names(txt)
            return txt
        else:
            print(f'Skipping a file. Too few speakers in \n{self.fname}')

    '''
    SPEAKER IDENTIFICATION & RE-MAPPING
    '''
    def speaker_cnt(self, txt):
        self.speakers = defaultdict(int)
        cnt = 0
        for line in txt:
            cnt +=1
            speaker = line[1:11]
            self.speakers[speaker] += len(line[14:].split(' '))
            # if len(line[14:].split(' ')) == 1:
            #     print(f'Offending line found: {cnt}, content: {line[14:]}')
        print(f'Dialogue contains : {len(self.speakers.keys())} speakers.')
        for i in self.speakers.keys():
            print(f'speaker: {i}, words spoken: {self.speakers[i]}')
        if len(list(self.speakers.keys())) > 3:
            print(f'More than three speakers in captions! Only 3 speakers allowed.')

    def remap_speaker_names(self, captions, for_train = True):
        # coded mapping
        new_speaker= dict(SPEAKER_00='', 
                          SPEAKER_01='',
                          SPEAKER_02='')
        if for_train:
            # interpretable mapping for training roles.
            new_speaker= dict(LocalTech='', 
                              Patient__='',
                              Assistant='')
        # list of tuples
        sorted_speakers = sorted(self.speakers.items(), key= lambda x: x[1])
        print(f'sorted speakers by word count: \n{sorted_speakers}')
        s_cnt = len(list(self.speakers.keys()))
        if s_cnt > 3:
            print(f'{self.fname} is invalid! Only 3 speakers allowed. Removing speakers by lowest word count.')
            new_idx = s_cnt - 3
            # discard speaker names with lowest word counts.
            s2 = [i[0] for i in sorted_speakers[new_idx:]]
            print(f'truncated speaker list: {s2}')
        else:
            s2 = [i[0] for i in sorted_speakers]
        # if only two speakers, remove local tech before speaker assignment
        if s_cnt == 2:
            new_speaker.pop(list(new_speaker.keys())[0])
        # a list of new_speaker names
        s1 = list(new_speaker.keys())
        # for X speakers in dataset
        for idx in range(len(s2)):
            # assign new speaker names to the top X speakers in captions by word count
            new_speaker[s1[idx]] = s2[idx]
        # reverse key:value pairs for easy replacement in captions.
        new_speaker = {v:k for (k,v) in new_speaker.items()}
        print(f'speaker mapping = <speaker in data> : <new speaker label>\n{new_speaker}')
        s1 = list(new_speaker.keys())
        # use mapping to rename speakers in each caption line.
        try:
            for idx in range(len(captions)):
                captions[idx] = captions[idx].replace(captions[idx][1:11], new_speaker[captions[idx][1:11]])
        except KeyError:
                # keys that aren't present in the mapping will be assigned SPEAKER_00
                captions[idx] = captions[idx].replace(captions[idx][1:11], new_speaker[s1[0]])
        return captions

    '''
    TRAINING DATA PREPARATION
    '''
    def to_llama_dialogue(self, txt):
        dialogue = []
        for l in txt:
            speaker = l[1:11]
            content = l[14:]
            line = {"role":speaker, "content":content}
            dialogue.append(line) 
        return dialogue

def deduplicate(groups):
    old = []
    deduped = []
    for g in groups:
        if old != g:
            old = g
            deduped.append(g)
    return deduped

def millisec(timeStr):
  spl = timeStr.split(":")
  s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
  return s

def timeStr(t):
  return '{0:02d}:{1:02d}:{2:06.2f}'.format(round(t // 3600), 
                                            round(t % 3600 // 60), 
                                            t % 60)

preS = f'<!DOCTYPE html>\n<html lang="en">\n\n<head>\n\t<meta charset="UTF-8">\n\t'\
    '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n\t'\
    '<meta http-equiv="X-UA-Compatible" content="ie=edge">\n\t'\
    '<title>{video_title}</title>\n\t'\
    '<style>\n\t\tbody {\n\t\t\tfont-family: sans-serif;\n\t\t\tfont-size: 14px;'\
    '\n\t\t\tcolor: #111;\n\t\t\tpadding: 0 0 1em 0;\n\t\t\tbackground-color: '\
    '#efe7dd;\n\t\t}\n\n\t\ttable {\n\t\t\tborder-spacing: 10px;\n\t\t}\n\n\t\tth'\
    ' {\n\t\t\ttext-align: left;\n\t\t}\n\n\t\t.lt {\n\t\t\tcolor: inherit;\n\t\t\t'\
    'text-decoration: inherit;\n\t\t}\n\n\t\t.l {\n\t\t\tcolor: #050;\n\t\t}\n\n\t\t.s'\
    '{\n\t\t\tdisplay: inline-block;\n\t\t}\n\n\t\t.c {\n\t\t\tdisplay: inline-block;'\
    '\n\t\t}\n\n\t\t.e {\n\t\t\t/*background-color: white; Changing background color */'\
    '\n\t\t\tborder-radius: 10px;\n\t\t\t/* Making border radius */\n\t\t\twidth: 50%;'\
    '\n\t\t\t/* Making auto-sizable width */\n\t\t\tpadding: 0 0 0 0;\n\t\t\t'\
    '/* Making space around letters */\n\t\t\tfont-size: 14px;\n\t\t\t'\
    '/* Changing font size */\n\t\t\tmargin-bottom: 0;\n\t\t}\n\n\t\t.t '\
    '{\n\t\t\tdisplay: inline-block;\n\t\t}\n\n\t\t#player-div {\n\t\t\tposition: '\
    'sticky;\n\t\t\ttop: 20px;\n\t\t\tfloat: right;\n\t\t\twidth: 40%\n\t\t}\n\n\t\t'\
    '#player {\n\t\t\taspect-ratio: 16 / 9;\n\t\t\twidth: 100%;\n\t\t\theight: '\
    'auto;\n\n\t\t}\n\n\t\ta {\n\t\t\tdisplay: inline;\n\t\t}\n\t</style>\n\t<script>'\
    '\n\t\tvar tag = document.createElement(\'script\');\n\t\ttag.src = "https://www.youtube.com/iframe_api";'\
    '\n\t\tvar firstScriptTag = document.getElementsByTagName(\'script\')[0];'\
    '\n\t\tfirstScriptTag.parentNode.insertBefore(tag, firstScriptTag);\n\t\tvar player;'\
    '\n\t\tfunction onYouTubeIframeAPIReady() {\n\t\t\t'\
    'player = new YT.Player(\'player\', {\n\t\t\t\t//height: \'210\',\n\t\t\t\t//width: '\
    '\'340\',\n\t\t\t\tvideoId: \'{video_id}\',\n\t\t\t});\n\n\n\n\t\t\t// '\
    'This is the source "window" that will emit the events.\n\t\t\tvar '\
    'iframeWindow = player.getIframe().contentWindow;\n\t\t\tvar lastword = null;\n\n\t\t\t'\
    '// So we can compare against new updates.\n\t\t\tvar lastTimeUpdate = "-1";\n\n\t\t\t'\
    '// Listen to events triggered by postMessage,\n\t\t\t// this is how different windows '\
    'in a browser\n\t\t\t// (such as a popup or iFrame) can communicate.\n\t\t\t'\
    '// See: https://developer.mozilla.org/en-US/docs/Web/API/Window/postMessage\n\t\t\t'\
    'window.addEventListener("message", function (event) {\n\t\t\t\t'\
    '// Check that the event was sent from the YouTube IFrame.\n\t\t\t\tif '\
    '(event.source === iframeWindow) {\n\t\t\t\t\tvar data = JSON.parse(event.data);'\
    '\n\n\t\t\t\t\t// The "infoDelivery" event is used by YT to transmit any\n\t\t\t\t\t'\
    '// kind of information change in the player,\n\t\t\t\t\t// such as the current time or a playback quality change.'\
    '\n\t\t\t\t\tif (\n\t\t\t\t\t\tdata.event === "infoDelivery" &&\n\t\t\t\t\t\tdata.info '\
    '&&\n\t\t\t\t\t\tdata.info.currentTime\n\t\t\t\t\t) {\n\t\t\t\t\t\t'\
    '// currentTime is emitted very frequently (milliseconds),\n\t\t\t\t\t\t'\
    '// but we only care about whole second changes.\n\t\t\t\t\t\tvar ts = '\
    '(data.info.currentTime).toFixed(1).toString();\n\t\t\t\t\t\tts = '\
    '(Math.round((data.info.currentTime) * 5) / 5).toFixed(1);\n\t\t\t\t\t\t'\
    'ts = ts.toString();\n\t\t\t\t\t\tconsole.log(ts)\n\t\t\t\t\t\tif (ts !== lastTimeUpdate)'\
    '{\n\t\t\t\t\t\t\tlastTimeUpdate = ts;\n\n\t\t\t\t\t\t\t// It\'s now up to you to format the time.'\
    '\n\t\t\t\t\t\t\t//document.getElementById("time2").innerHTML = time;\n\t\t\t\t\t\t\t'\
    'word = document.getElementById(ts)\n\t\t\t\t\t\t\tif (word) {\n\t\t\t\t\t\t\t\tif (lastword)'\
    '{\n\t\t\t\t\t\t\t\t\tlastword.style.fontWeight = \'normal\';\n\t\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\t\t'\
    'lastword = word;\n\t\t\t\t\t\t\t\t//word.style.textDecoration = \'underline\';\n\t\t\t\t\t\t\t\t'\
    'word.style.fontWeight = \'bold\';\n\n\t\t\t\t\t\t\t\tlet toggle = document.getElementById("autoscroll");'\
    '\n\t\t\t\t\t\t\t\tif (toggle.checked) {\n\t\t\t\t\t\t\t\t\tlet position = word.offsetTop - 20;'\
    '\n\t\t\t\t\t\t\t\t\twindow.scrollTo({\n\t\t\t\t\t\t\t\t\t\ttop: position,\n\t\t\t\t\t\t\t\t\t\t'\
    'behavior: \'smooth\'\n\t\t\t\t\t\t\t\t\t});\n\t\t\t\t\t\t\t\t}\n\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t}'\
    '\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t})\n\t\t}\n\t\tfunction jumptoTime(timepoint, id) '\
    '{\n\t\t\tevent.preventDefault();\n\t\t\thistory.pushState(null, null, "#" + id);\n\t\t\t'\
    'player.seekTo(timepoint);\n\t\t\tplayer.playVideo();\n\t\t}\n\t</script>\n</head>\n\n<body>\n\t<h2>'\
    '{video_title}</h2>\n\t<i>Click on a part of the transcription, to jump to its video, '\
    'and get an anchor to it in the address\n\t\tbar<br><br></i>\n\t<div id="player-div">\n\t\t'\
    '<div id="player"></div>\n\t\t<div><label for="autoscroll">auto-scroll: </label>\n\t\t\t'\
    '<input type="checkbox" id="autoscroll" checked>\n\t\t</div>\n\t</div>\n  '
postS = '\t</body>\n</html>'