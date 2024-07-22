import os
import re
from bs4 import BeautifulSoup
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import json
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import defaultdict
import pickle

stop_words = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", 
                "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", 
                "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", 
                "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", 
                "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", 
                "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", 
                "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", 
                "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", 
                "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", 
                "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", 
                "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", 
                "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", 
                "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", 
                "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", 
                "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", 
                "your", "yours", "yourself", "yourselves"]

class Milestone2:
    def __init__(self) -> None:
        # self.retrieval(base_folder)
        self.list_document = []

        self.lemma = WordNetLemmatizer()
        self.doc_count = 0
        self.index = {}
        self.length = defaultdict(list)
        self.counted_terms = defaultdict(set)
        self.scores = {}
        self.sorted_scores = []


    def retrieval(self, base_folder):
        documents = []
        # Loop through all sub-folders inside the base folder
        for sub_folder in sorted(os.listdir(base_folder)):
            sub_folder_path = os.path.join(base_folder, sub_folder)
            
            # Check if the current item is a sub-folder
            if os.path.isdir(sub_folder_path):
                # Loop through all files inside the sub-folder
                for file in sorted(os.listdir(sub_folder_path)):
                    file_path = os.path.join(sub_folder_path, file)
                    
                    # Check if the current item is a file
                    if os.path.isfile(file_path):
                        # Append the file path to the list of documents
                        documents.append(file_path)
        return documents

    def tokenizer(self, raw_contet):
        
        tokens = []
        word = ""
        for i in raw_contet:
                if (i >= 'a' and i <= 'z') or (i >= 'A' and i <= 'Z') or (i >= '0' and i <= '9'):
                    word += i.lower()
                elif word != "":
                    #check if word is stop word and length of word
                    if word not in stop_words and len(word) >= 3:
                        #transfer word to base form and adding word in token list
                        tokens.append(self.lemma.lemmatize(word))
                    word = ""
        return tokens

    def save_index(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.index, f)
            pickle.dump(self.length, f)
            pickle.dump(self.scores, f)

    def load_index(self, filename):
        with open(filename, "rb") as f:
            self.index = pickle.load(f)
            self.length = pickle.load(f)
            self.scores = pickle.load(f)

    def handel_document(self, documents):
        
        doc_text = []
        for i, document in enumerate(documents):
            with open(document, "r", encoding="utf8") as f:
                self.doc_count += 1
                self.scores[document] = 0.0
                self.length[document] = []
                #read the content of file, inculding <tag>
                content = f.read()
                # self.fileName = document
                soup = BeautifulSoup(content, 'html.parser')
                

                # extract title text
                title = soup.find("title")
                headings = soup.find_all(['h1', 'h2', 'h3', 'h4'])
                anchor = soup.find('a')
                bolds = soup.find_all('b')

                if title:
                    title_text = title.get_text()
                    doc_text.append(title_text)
                    title_tokens = self.tokenizer(title_text)
                    for word in title_tokens:
                        if word not in self.index:
                            self.index[word] = {}
                        if document not in self.index[word]:
                            self.index[word][document] = [1,0,0,0] # (tf, idf, tfidf, html_tag)
                            self.index[word][document][1] += 1 # idf
                        else:
                            self.index[word][document][0] += 1 # tf
                        self.index[word][document][3] += 1.0 # weight for title tag
                    
                # extract heading text and add to doc_text with higher weight
                if headings:
                    for heading in headings:
                        heading_text = heading.get_text()
                        doc_text.append(heading_text)
                        heading_tokens = self.tokenizer(heading_text)
                        for word in heading_tokens:
                            if word not in self.index:
                                self.index[word] = {}
                            if document not in self.index[word]:
                                self.index[word][document] = [1,0,0,0] # (tf, idf, tfidf, html_tag)
                                self.index[word][document][1] += 1 # idf
                            else:
                                self.index[word][document][0] += 1 # tf
                            self.index[word][document][3] += .75 # weight for heading tags

                # extract anchor text
                if anchor:
                    anchor_text = anchor.get_text()
                    doc_text.append(anchor_text)
                    anchor_tokens = self.tokenizer(anchor_text)
                    for word in anchor_tokens:
                        if word not in self.index:
                            self.index[word] = {}
                        if document not in self.index[word]:
                            self.index[word][document] = [1,0,0,0] # (tf, idf, tfidf, html_tag)
                            self.index[word][document][1] += 1 # idf
                        else:
                            self.index[word][document][0] += 1 # tf
                        self.index[word][document][3] += .5 # weight for title tag

                # extract bold text and add to doc_text with higher weight
                if bolds:
                    for bold in bolds:
                        bold_text = bold.get_text()
                        doc_text.append(bold_text)
                        bold_tokens = self.tokenizer(bold_text)
                        for word in bold_tokens:
                            if word not in self.index:
                                self.index[word] = {}
                            if document not in self.index[word]:
                                self.index[word][document] = [1,0,0,0] # (tf, idf, tfidf, html_tag_score)
                                self.index[word][document][1] += 1 # idf
                            else:
                                self.index[word][document][0] += 1 # tf
                            self.index[word][document][3] += .25 # weight for bold tag



                # contentNotTag = re.sub(r'<title>.*?</title>', '', content)
                # contentNotTag = re.sub(r'<h\d>.*?</h\d>|<b>.*?</b>|<a.*?>.*?</a>', '', contentNotTag)
                # contentNotTag = re.sub(r'<[^>]+>', '', contentNotTag)
                
                contentNotTag = re.sub('<[^<]+?>', '', content)
                
                doc_text.append(contentNotTag)

                # tokenize
                tokens = self.tokenizer(contentNotTag)

                for word in tokens:
                    if word not in self.index:
                        self.index[word] = {}
                    if document not in self.index[word]:
                        # initialize
                        self.index[word][document] = [1,0,0,0] # (tf, idf, tfidf, html_tag_score)
                        self.index[word][document][1] += 1 # idf
                    else:
                        self.index[word][document][0] += 1 # tf
                    

                print('document {} **************************'.format(document))
                print('----------------------------------------------------------------------------------------------')
                
        
        # calculate raw tf-idf
        for word in self.index.keys():
            n = len(self.index[word])
            idf = math.log(self.doc_count/n) if n > 0 else 0.0
            for doc, nums in self.index[word].items():
                nums[1] = idf
                nums[2] = nums[0] * nums[1] # tf-idf = tf * idf
                self.length[doc].append(nums[2]**2)
        
        # calculate doc length
        for doc, vals in self.length.items():
            # sum of all tf-idf^2
            length = sum([x**2 for x in vals])
            length = math.sqrt(length)
            self.length[doc] = length
        
        
    def get_paths(self, query):
        
        # lemmatize each query term
        query_terms = [self.lemma.lemmatize(word.lower()) for word in query.split()]
        paths = []
        

        # calculating cosine:
        for term in query_terms:
            # try/except: check if term exists in document
            try:
                for doc,info in self.index[term].items():
                    w_tq = query.count(term) * self.index[term][doc][1] # tf * idf of term in query
                    self.scores[doc] += info[2] * w_tq # tfidf * w_tq
            except:
                continue
        
        for doc in self.length:
            if self.length[doc] != 0:
                self.scores[doc] = self.scores[doc]/self.length[doc] # divide scores by doc weight
                # check if term exists in doc
                try:
                    for term in query_terms:
                        self.scores[doc] += self.index[term][doc][3] # add html weights
                except:
                    continue
            else:
                self.scores[doc] = 0
        
        # sort based on scores
        self.sorted_scores = sorted(self.scores.items(), key=lambda item: item[1], reverse=True)

        paths = [path.strip('WEBPAGES_RAW/').strip('/') for (path, score) in self.sorted_scores if score != 0]
        return paths

    def get_urls(self, paths):
        # Open the JSON file
        with open('WEBPAGES_RAW/bookkeeping.json', "r") as file:
            # Load the contents of the file into a Python object
            data = json.load(file)
        
        urls_list = []
        for path in paths:
            urls_list.append(data[path])

        # write urls onto file
        file_name = input('Enter File Name: ')
        f = open(file_name, "w",encoding="UTF-8")
        for i, url in enumerate(urls_list):
            if i == 20:
                break
            f.write(url + " (" + str(self.sorted_scores[i][1]) + ")" + '\n')
        f.close()
        return urls_list

        
                
    def report(self):
            # write index onto text file

            def format_list(lst):
                return "[" + ", ".join([str(x) for x in lst]) + "]"
            
            f = open("index1.txt", "w", encoding="UTF-8")
            
            for word in self.index.keys():
                docs = list(self.index[word].keys())
                tfs = [self.index[word][doc][0] for doc in docs]
                idfs = [self.index[word][doc][1] for doc in docs]
                tf_idfs = [self.index[word][doc][2] for doc in docs]
                tags = [self.index[word][doc][3] for doc in docs]
                word_info = {
                    "documents": format_list(docs),
                    "tfs": format_list(tfs),
                    "idfs": format_list(idfs),
                    "tf-idfs": format_list(tf_idfs),
                    "tag scores": format_list(tags)
                }
                f.write(f"{word}:\n")
                f.write(json.dumps(word_info, indent=4))
                f.write("\n")
            f.close()

