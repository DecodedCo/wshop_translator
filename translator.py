from transformers import pipeline
import yaml, json
from yaml.loader import SafeLoader
from tqdm import tqdm


def open_workshop(path):
    with open(path) as f:
        return yaml.load(f, Loader=SafeLoader)

def translator_pipe(language="SPA"):
    if language == "SPA":
        return pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
    elif language == "POR":
        return pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-pt")
    else:
        print("Sorry, this language is not supported. Please try again and use SPA for Spanish or POR for portuguese.")
        

def extractor(data):
    full_notes = []
    for idx, slide in enumerate(tqdm(data["slides"])):
        if "notes" in slide.keys():
            one_note = slide["notes"].split("\n")
            for num, sentence in enumerate(one_note):
                if sentence != None and len(sentence) > 3:
                    if len(sentence) < 400:
                        one_note[num] = translator(sentence)[0]['translation_text']
                    elif len(sentence) > 400:
                        group_sent = sentence.split(".")
                        for num2, sent2 in enumerate(group_sent):
                            group_sent[num2] = translator(sent2)[0]['translation_text']
                        group_sent2 = ".".join(sent for sent in group_sent)
                        one_note[num] = group_sent2
                else:
                    one_note[num] = sentence
            slide["notes"] = "\n".join(sent for sent in one_note)
        if "text" in slide.keys():
            text = slide["text"]
            slide["text"] = translator(text)[0]['translation_text']
        if "title" in slide.keys():
            title = slide["title"]
            slide["title"] = translator(title)[0]['translation_text']
        data["slides"][idx] = slide
    return data

if __name__ == "__main__":
    