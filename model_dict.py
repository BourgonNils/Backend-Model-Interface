from models import BasicBertForClassification, BertFeaturesForSequenceClassification
from transformers import AutoTokenizer

# When you add a model or a task to your app Just import your model and add the path to it
models_dic = {

    "Crisis": {

        "Predict utility": {
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
            },
            "description": "Predict if the tweet is about a crisis"
        },
        "Predict urgency": {
            'models': {
                "flaubert-base-cased": {
                    "model": BasicBertForClassification,
                    "path": "../models_weights/Crisis_Ternary/Crisis_ThreeClass_flaubert_base.pth",
                    "tokenizer_base": "flaubert-base-cased",
                },
                "flaubert-finetuned-focalloss": {
                    "model": BasicBertForClassification,
                    "path": "../models_weights/Crisis_Ternary/flaubert_finetuned_ternary.pth",
                    "tokenizer_base": "flaubert-base-cased"
                }
            },
            "labels_dic": {
                0: 'Message-InfoUrgent',
                1: 'Message-NonUtilisable',
                2: 'Message-InfoNonUrgent', },

            "description": "Predict if a tweet is about a crisis and asses the urgency"

        },

        "Predict category": {
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
                6: 'Critiques'},

            "description": "Predict if a tweet is about a crisis and determine the type of information"

        }
    },

    "Psycho": {
        "TestTaskPsycho": {
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
            },
            "description": "DummyTask"
        }

    },
    "Speech act": {
        "Predict speech act": {
            "models": {
                "best-SA": {
                    "model": BasicBertForClassification,
                    "path": "../models_weights/SA/BestSA.pth",
                    "tokenizer_base": "camembert-base",
                },
            },
            "labels_dic": {0: 'Jussif', 1: 'Assertif', 2: 'Subjectif', 3: 'Interrogatif'},
            "description": "Predict the speech act within the speech act. The tweet can be : Assertive, Interrogative, Jussive or Subjective"


        }
    }
}


def get_model(task, model_name):
    models = {}
    for d in list(models_dic.values()):
        models.update(d)

    if model_name == "":
        print("\nUsing default model !\n")
        model_name = list(models[task]['models'].keys())[0]

    model = models[task]['models'][model_name]["model"].load(
        models[task]['models'][model_name]["path"])

    if "features" in models[task]['models'][model_name]:
        features = models[task]['models'][model_name]["features"]
    else:
        features = []

    Tokenizer = AutoTokenizer.from_pretrained(
        models[task]['models'][model_name]["tokenizer_base"])

    return model, Tokenizer, models[task]["labels_dic"], features


if __name__ == '__main__':
    print("TEST")
    models = {}
    for d in list(models_dic.values()):
        models.update(d)
    print(models)
