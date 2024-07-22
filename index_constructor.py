import os
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import nltk
from bs4 import BeautifulSoup
from pathlib import Path
import json
from collections import defaultdict
import math
import json

nltk.download('wordnet')
# (GLOBAL VARIABLES) Initialize dictionaries based on the first character of the token
# index structure {"word":[[folder, file, TF, TF_IDF], ...]}
a = defaultdict(list)
b = defaultdict(list)
c = defaultdict(list)
d = defaultdict(list)
e = defaultdict(list)
f = defaultdict(list)
g = defaultdict(list)
h = defaultdict(list)
i = defaultdict(list)
j = defaultdict(list)
k = defaultdict(list)
l = defaultdict(list)
m = defaultdict(list)
n = defaultdict(list)
o = defaultdict(list)
p = defaultdict(list)
q = defaultdict(list)
r = defaultdict(list)
s = defaultdict(list)
t = defaultdict(list)
u = defaultdict(list)
v = defaultdict(list)
w = defaultdict(list)
x = defaultdict(list)
y = defaultdict(list)
z = defaultdict(list)
misc = defaultdict(list)

# List of tf-idfs for each word in the document for calculating doc_length
list_of_tfidf = defaultdict(list)   # docID: [TF_IDF, TF_IDF, TF_IDF, ...]  -->  0/131: [1.22212, 1.311112, 0.98399, ...]

# Doc_length for each document (to be used later for cosine similarity)
doc_length = {}         # docID: doc_length  -->  0/131: 5.0012852

totalDocs = 0

# docID count dictionary (to be used for PageRank)
# {docID: count}
pageRankDict = {}


# calculates the file size in kilobytes
def getFileSizeInKB(filePath):
    fObj = Path(filePath)
    size = fObj.stat().st_size
    return size/1024

# Takes in a list of tokens and return a dictionary with the keys as the token and value as the word frequency in the document
# wordFreqDict: {token: wordFreq}
def computerWordFreq(tokens):
    wordFreqDict = defaultdict(int)
    for i in tokens:
        wordFreqDict[i] += 1
    return wordFreqDict

def computeTF(wordFreq):
    if wordFreq > 0:
        return 1 + math.log10(wordFreq)
    else:
        return 0

def computeIDF(numDocIDs, totalDocs):
    N = totalDocs
    return math.log10(N / float(numDocIDs))

def computeTagsImportance(word, lemmatized_bold, lemmatized_title, lemmatized_h1, lemmatized_h2, lemmatized_h3, lemmatized_anchor):
    ratingSum = 0
    if word in lemmatized_title:
        ratingSum += 0.08
    if word in lemmatized_h1:
        ratingSum += 0.04
    if word in lemmatized_h2:
        ratingSum += 0.03
    if word in lemmatized_h3:
        ratingSum += 0.02
    if word in lemmatized_bold:
        ratingSum += 0.01
    if word in lemmatized_anchor:
        ratingSum += 0.01
    return ratingSum

def computePageRank(docid):
    try:
        # returns the number of inlinks of a website/docid then divide it by 100 to make it a smaller number
        return pageRankDict[docid] / 100
    except:   
        return 0

def computeDocLength():
    for k, v in list_of_tfidf.items():
        result = sum(i*i for i in v)
        result = math.sqrt(result)
        # Add each document length for each document to the doc_length dictionary
        doc_length[k] = result

# takes in a token and returns the corresponding file name
def sortToken(token):
    firstChar = token[0].lower()
    if 97 <= ord(firstChar) <= 122:
        txtFileName = firstChar
    else:
        txtFileName = 'misc'
    return txtFileName

def generateJSONFile():
    # Cite: https://www.geeksforgeeks.org/how-to-convert-python-dictionary-to-json/
    for alphabet in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'misc', 'doc_length']:
        fileName = alphabet + '.json'
        with open(fileName, 'a', encoding = 'utf-8') as outfile:
            json.dump(globals()[alphabet], outfile, indent = 4)


def constructIndex(bkDict):
    flippedBK = {y:x for x,y in bkDict.items()}
    stop_words = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    global totalDocs
    totalKB = 0
    uniqueWords = set()

    for root, dirs, files in os.walk("./WEBPAGES_RAW", topdown=True):
        for f in files:
            if not f.endswith('.json') and not f.endswith('.tsv')and not f.endswith('.DS_Store'):
                dirPath = os.path.join(root, f)
                docID = dirPath[15:]

                with open(dirPath, "r", encoding = "utf-8") as file:

                    text_with_tags = file.read()

                    # Source: https://stackoverflow.com/questions/24856035/how-to-detect-with-python-if-the-string-contains-html-code
                    if bool(BeautifulSoup(text_with_tags, "html.parser").find()):       # Check if text_with_tags is actually HTML file, if it is, then parse it
                        # Count Number of Documents
                        totalDocs += 1

                        # Count total size of kilobytes
                        totalKB += getFileSizeInKB(dirPath)

                        text_with_tags = BeautifulSoup(text_with_tags, 'html.parser')
                      

                        # Count number of outlinks in pageRankDict (for calculating PageRank later)
                        for url in text_with_tags.find_all('a'):
                            try: 
                                # remove "http://" or "https://" so we can  match in bookkeeping
                                link = url.get('href')
                                if link.startswith("http://"):
                                    link = link.replace("http://", "")
                                elif link.startswith("https://"):
                                    link = link.replace("https://", "")
                                
                                # find the docid for the link and keep count in pageRankDict
                                docid = flippedBK[link]
                                if docid in pageRankDict:
                                    pageRankDict[docid] += 1
                                else:
                                    pageRankDict[docid] = 1
                            except:
                                pass
                                    
        
                        # Get the list of bold, title, h1, h2, h3, and anchor tags
                        # then converts it to a string with the tokens separated by a space
                        listOfBold = text_with_tags.find_all('b')                           # listOfBold is of type bs4.element.ResultSet
                        listOfBold = " ".join([elem.get_text() for elem in listOfBold])     # We change type bs4.element.ResultSet into string so we can tokenize it

                        listOfTitle = text_with_tags.find_all('title')
                        listOfTitle = " ".join([elem.get_text() for elem in listOfTitle])

                        listOfH1 = text_with_tags.find_all('h1')
                        listOfH1 = " ".join([elem.get_text() for elem in listOfH1])

                        listOfH2 = text_with_tags.find_all('h2')
                        listOfH2 = " ".join([elem.get_text() for elem in listOfH2])

                        listOfH3 = text_with_tags.find_all('h3')
                        listOfH3 = " ".join([elem.get_text() for elem in listOfH3])

                        listOfAnchors = text_with_tags.find_all('a')
                        listOfAnchors = " ".join([elem.get_text() for elem in listOfAnchors])

                        # Get everything except HTML tags
                        text_with_no_tags = text_with_tags.get_text().lower()

                        # TOKENIZING PROCESS
                        # Tokenizing the whole text
                        # Cite: https://www.nltk.org/_modules/nltk/tokenize/regexp.html
                        tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')        #Forms tokens out of alphabetic sequences, money expressions, and any other non-whitespace sequences
                        tokens = tokenizer.tokenize(text_with_no_tags)
                        # Tokenizing the bolded text
                        bolded_tokens = tokenizer.tokenize(listOfBold)
                        # Tokenizing the title
                        title_tokens = tokenizer.tokenize(listOfTitle)
                        # Tokenizing h1
                        h1_tokens = tokenizer.tokenize(listOfH1)
                        # Tokenizing h2
                        h2_tokens = tokenizer.tokenize(listOfH2)
                        # Tokenizing h3
                        h3_tokens = tokenizer.tokenize(listOfH3)
                        # Tokenizing anchors
                        anchor_tokens = tokenizer.tokenize(listOfAnchors)

                        # LEMMATIZING PROCESS
                        lemmatizer = WordNetLemmatizer()

                        # list comprehension where lemmatized_words is a list of the lemmatized tokens
                        lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
                        lemmatized_bold = [lemmatizer.lemmatize(word) for word in bolded_tokens]
                        lemmatized_title = [lemmatizer.lemmatize(word) for word in title_tokens]
                        lemmatized_h1 = [lemmatizer.lemmatize(word) for word in h1_tokens]
                        lemmatized_h2 = [lemmatizer.lemmatize(word) for word in h2_tokens]
                        lemmatized_h3 = [lemmatizer.lemmatize(word) for word in h3_tokens]
                        lemmatized_anchor = [lemmatizer.lemmatize(word) for word in anchor_tokens]
                    

                        # dictionary of the tokens frequencies in current document i.e {word: freq, ...}
                        wordFreqDict = computerWordFreq(lemmatized_words)

                        # Prevent repeating docID for the same word in current document ex. 0/100 0/100 0/ 100/ 2/250 0/100
                        repeated_tokens = set()

                        # Token Info format: [docID:string, TF:float, tagsImportanceRating:int, TF_IDF:float, score:float]
                        tokenInfo = list()
                        for word in lemmatized_words:
                            if word.isalnum() and word not in stop_words and word not in repeated_tokens:
                                repeated_tokens.add(word)
                                uniqueWords.add(word)
                                # compute the TF for current word in current document
                                TF = computeTF(wordFreqDict[word])

                                # compute the tags and anchor words importance rating
                                tagsImportanceRating = computeTagsImportance(word, lemmatized_bold, lemmatized_title, lemmatized_h1, lemmatized_h2, lemmatized_h3, lemmatized_anchor)

                                # Add word to the appropriate index
                                tokenInfo = [docID, TF, tagsImportanceRating, 0, 0]

                                # Cite: https://www.codespeedy.com/convert-string-into-variable-name-in-python/#:~:text=Declaring%20Variable%20Name%20Dynamically&text=Now%2C%20by%20using%20globals(),in%20the%20global%20%2F%20local%20namespace.
                                (globals()[sortToken(word)])[word].append(tokenInfo)       # sortToken(word) returns a character or 'misc'

                        # print("dirPath:", dirPath, "\tdocID:", docID, "\tCount:", totalDocs, '\ttotalKBsize:', totalKB, '\ttotalUniqueWords:', len(uniqueWords))
    print("Count:", totalDocs, '\ttotalKBsize:', totalKB, '\ttotalUniqueWords:', len(uniqueWords))

    # Calculating idf and tf-idf and the current score for each document
    for alphabet in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'misc']:
        for token, info in globals()[alphabet].items():
            numDocIDs = len(info)
            # TF_IDF = 0
            for doc in info:
                # Calculate TF_IDF
                TF_IDF = doc[1] * computeIDF(numDocIDs, totalDocs)
                doc[3] = TF_IDF

                # docID
                doc_name = doc[0]

                # compute pageRank 
                pageRank = computePageRank(doc_name)

                # Calculate the overall score. TFIDF 70%, HTML tag rating 10%, page rank 15% = 100%
                score = (0.70 * TF_IDF) + (0.10 * doc[2]) + (0.20 * pageRank)
                doc[4] = score

                # Record tf-idf for each word in the document
                list_of_tfidf[doc_name].append(TF_IDF)

    # Change the list of tf-idf for each word in the document to one document length (sqrt(x^2 + y^2 + ...))
    computeDocLength()
    # print(pageRankDict)