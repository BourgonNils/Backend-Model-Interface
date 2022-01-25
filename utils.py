from preprocessing.text_preprocessing import TextPreprocessing
import re
from nltk.corpus import stopwords

COL_USED = ['text','link','likes','retweets','date']

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def get_data_word_cloud(text_series):
    text = ""
    for row_text in text_series:
        text += " " +row_text

    # TODO Remove stop words
    text = text.lower()
    #Remove URL 
    text = re.sub(r'http(\S)+', '', text)
    text = re.sub(r'http ...', '', text)
    text = re.sub('["$#%()*+,-@./:;?![\]^_`{|}~\n\t’\']', ' ', text)
    text = re.sub(r' +', ' ', text)

    wordlist = text.split(' ')
    filteredList = [w for w in wordlist if not w in stopwords.words('french')]
    dictionary = wordListToFreqDict(filteredList)
    sorteddict = sortFreqDict(dictionary)
    
    ret = []
    #Put in required format for word cloud
    for i in sorteddict[:25] :
        ret.append({'text' : i[1], 'value':i[0]})

    return ret


def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux


def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(list(zip(wordlist,wordfreq)))