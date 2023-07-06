import pickle
import spacy


intents=[14,20,28,29,35,36,43,45,46,57,58,7,10]
intents_names={
    "14":"audio_volume_up",
    "20":"play_audiobook",
    "28":"music_settings",
    "29":"audio_volume_other",
    "35":"audio_volume_down",
    "36":"play_radio",
    "43":"music_likeness",
    "45":"play_music",
    "46":"audio_volume_mute",
    "57":"music_query",
    "58":"play_podcasts",
    "7":"music_dislikeness",
    "10":"add_to_playlist"
    }
svm_model = pickle.load(open('src/models/svm_model.sav', 'rb'))
ner_model = pickle.load(open('src/models/ner_model.sav', 'rb'))

def classify(command):
    return svm_model.predict([command])


def ner(command):
    ents={}
    doc = ner_model(command)
    for ent in doc.ents:
        ents[ent.text]=ent.label_
    return ents