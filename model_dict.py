from models import BasicBertForClassification, BertFeaturesForSequenceClassification
from transformers import AutoTokenizer

# When you add a model or a domain to your app Just import your model and add the path to it
models_dic = {
    "Crisis":{
    "crisis_binary": {
        'models': {
            "flaubert-base-cased": {
                "model": BasicBertForClassification,
                "path": "../models_weights/Crisis_Binary/Crisis_Binary_flaubert_base.pth",
                "tokenizer_base": "flaubert-base-cased",
            },
              "flaubert-finetuned-focalloss": {
                "model": BasicBertForClassification,
                "path": "../models_weights/Crisis_Binary/flaubert_finetuned_binary.pth",
                "tokenizer_base": "flaubert-base-cased",
            }
        },
        "labels_dic": {
            0: 'Message-Utilisable',
            1: 'Message-NonUtilisable'
        }
    },
    "crisis_Three_Class": {
        'models': {
            "flaubert-base-cased": {
                "model": BasicBertForClassification,
                "path": "../models_weights/Crisis_Ternary/Crisis_ThreeClass_flaubert_base.pth",
                "tokenizer_base": "flaubert-base-cased",
            },
            "flaubert-finetuned-focalloss": {
                "model" : BasicBertForClassification,
                "path": "../models_weights/Crisis_Ternary/flaubert_finetuned_ternary.pth",
                "tokenizer_base": "flaubert-base-cased"

            }
        },
        "labels_dic": {
            0: 'Message-InfoUrgent',
            1: 'Message-NonUtilisable',
            2: 'Message-InfoNonUrgent', }
    },
    "crisis_MultiClass": {
        'models': {
            "bert_base_multiligual_cased": {
                "model": BasicBertForClassification,
                "path": "../models_weights/Crisis_Multiclass/Crisis_MultiClass_flaubert_base.pth",
                "tokenizer_base": "flaubert-base-cased",
            },
             "flaubert-finetuned-focalloss": {
                "model": BasicBertForClassification,
                "path": "../models_weights/Crisis_Multiclass/flaubert_finetuned_multiclass.pth",
                "tokenizer_base": "flaubert-base-cased",
            },

        },
        "labels_dic": {
            0: 'Degats-Materiels',
            1: 'Degats-Humains',
            2: 'AutresMessages',
            3: 'Message-NonUtilisable',
            4: 'Avertissement-conseil',
            5: 'Soutiens',
            6: 'Critiques'}
    }}
}

def get_model(domain, model_name):
    models = {}
    for d in list(models_dic.values()):
       models.update(d)
    model = models[domain]['models'][model_name]["model"].load(
        models[domain]['models'][model_name]["path"])

    if "features" in models[domain]['models'][model_name]:
        features = models[domain]['models'][model_name]["features"]
    else:
        features = []
    Tokenizer = AutoTokenizer.from_pretrained(
        models[domain]['models'][model_name]["tokenizer_base"])

    return model, Tokenizer, models[domain]["labels_dic"], features



if __name__ == '__main__':
    print("TEST")
    models = {}
    for d in list(models_dic.values()):
       models.update(d)
    print(models)