from glob import glob
from diarization_utils import CaptionCleaner

output_root = "/home/digitalopt/proj/diarization/recordings/"
cap_paths = glob(output_root + '*/*/capspeaker.txt')
cap_paths.sort()
print(f'found {len(cap_paths)} caption texts')
# loads NER model by default, but pass NER=False when instantiating if not desired.
cleaner = CaptionCleaner()
for p in cap_paths:
    print(f'reading {p}')
    captions = cleaner.read_captions(p)
    cleaned = cleaner.clean(captions)
    if cleaned:
        outfile = '/'.join(p.split('/')[:-1]) + '/clean_captions.txt'
        cleaner.write_captions(cleaned, outfile)
