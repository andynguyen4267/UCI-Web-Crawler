import os
import re
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import pickle
import math
import heapq
import json
import itertools
from collections import Counter, defaultdict
from bs4 import BeautifulSoup

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

class Milestone1:
    def __init__(self) -> None:
        # single word dictionary
        self.single_word_dict = defaultdict(list)
        self.single_title_word = {} # {term: [d1,d2]}
        self.single_h_tag_word = {}
        self.single_ab_tag_word = {}
        
        #dictionary of bigram
        self.bigram_word_dict = defaultdict(list)
        self.bigram_h_tag_word = {}
        self.bigram_title_word = {}
        self.bigram_ab_tag_word = {}
        self.bigram_doc_length = {}

        self.lemma = WordNetLemmatizer()
        self.total_docs = 37497
        self.totaldocs = 0
        self.doc_description = {}

    # retrieval all files in foders
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

    # stemming and lematizing tokens
    def tokenizer(self, raw_content): 
        tokens = []
        word = ""
        for i in raw_content:
            if (i >= 'a' and i <= 'z') or (i >= 'A' and i <= 'Z') or (i >= '0' and i <= '9'):
                word += i.lower()
            elif word != "":
                #check if word is stop word and length of word
                if word not in stop_words and len(word) >= 3:
                    #transfer word to base form and adding word in token list
                    tokens.append(self.lemma.lemmatize(word))
                word = ""
        return tokens

    def unigram_dict(self, fileName, tokens): #Unigram
        count = Counter(tokens)
        for word in count:
            self.single_word_dict[word].append([fileName, count[word]])

    def bigram_dict(self, fileName, bigram):
        count = Counter(bigram)
        for word in count:
            self.bigram_word_dict[word].append([fileName, count[word]])

    def return_dict(self):
        return self.bigram_word_dict, self.single_word_dict
    
    def get_description(self):
        return self.doc_description

    # open file and handle the document
    def handle_document(self, documents):
        for i, document in enumerate(documents):
            with open(document, "r", encoding="utf8") as f:
                #read the content of file, inculding <tag>
                print("PROCESSING ",document)
                content = f.read()
                soup = BeautifulSoup(content, 'html.parser')
                
                #Extract the title tags
                title_tag = soup.find('title')
                if title_tag is not None:
                    title_bigram = list(nltk.bigrams(self.tokenizer(title_tag.text)))
                    for i in title_bigram:
                        if i in self.bigram_title_word:
                            self.bigram_title_word[i].append(document) 
                        else:
                            self.bigram_title_word[i] = [document]

                    title_single = list(self.tokenizer(title_tag.text))
                    for i in title_single:
                        if i in self.single_title_word:
                            self.single_title_word[i].append(document) 
                        else:
                            self.single_title_word[i] = [document]

                #Extract the h tags
                h_tag = soup.find_all(['h1', 'h2', 'h3', 'h4' ])
                text = ""
                for tag in h_tag:
                    text += tag.get_text()
                h_bigram = list(nltk.bigrams(self.tokenizer(text))) 
                for i in h_bigram:
                    if i in self.bigram_h_tag_word:
                        self.bigram_h_tag_word[i].append(document) 
                    else:
                        self.bigram_h_tag_word[i] = [document]
                        
                h_single = list(self.tokenizer(text))
                for i in h_single:
                    if i in self.single_h_tag_word:
                        self.single_h_tag_word[i].append(document) 
                    else:
                        self.single_h_tag_word[i] = [document]

                #Extract the a, b tags
                ab_tag = soup.find_all(['b', 'a'])
                ab_words = ""
                for i in ab_tag:
                    ab_words += " " + i.get_text()
                ab_bigram = list(nltk.bigrams(self.tokenizer(ab_words))) 
                for i in ab_bigram:
                    if i in self.bigram_ab_tag_word:
                        self.bigram_ab_tag_word[i].append(document) 
                    else:
                        self.bigram_ab_tag_word[i] = [document]

                ab_single = list(self.tokenizer(ab_words))
                for i in ab_single:
                    if i in self.single_ab_tag_word:
                        self.single_ab_tag_word[i].append(document) 
                    else:
                        self.single_ab_tag_word[i] = [document]
                
                # reomove all tags
                contentNotTag = re.sub('<[^<]+?>', '', content) 

                # tokenizer
                tokens = self.tokenizer(contentNotTag)

                # create short description
                desc = tokens[:100] # extract first 100 words
                self.doc_description[document] = ' '.join(desc)
            
                # single word
                self.unigram_dict(document, tokens)

                # bigram words
                bigrams_list = list(nltk.bigrams(tokens))
                self.bigram_dict(document, bigrams_list)

    def calculate_tf_idf(self, posting_dict):  
        doc_length = {}
        tf_idf_doc_dict = {}
        for term, value in posting_dict.items(): #{ term: [ [doc, tf], [doc, tf]]}
            sub_dict = {}     
            for item in value:
                priority = 0
                tf_idf = (1+math.log10(item[1])) * math.log10(self.total_docs / len(posting_dict[term]))
                # set priority score
                if ((term in self.single_title_word) and (item[0] in self.single_title_word[term])) or ((term in self.bigram_title_word) and (item[0] in self.bigram_title_word[term])):  
                    priority = 1
                elif ((term in self.single_h_tag_word) and (item[0] in self.single_h_tag_word[term])) or ((term in self.bigram_h_tag_word) and (item[0] in self.bigram_h_tag_word[term])):
                    priority = 0.75
                elif ((term in self.single_ab_tag_word) and (item[0] in self.single_ab_tag_word[term])) or ((term in self.bigram_ab_tag_word) and (item[0] in self.bigram_ab_tag_word[term])):
                    priority = 0.5
                sub_dict[item[0]] = [tf_idf, priority] #{doc: [tfidf, priority]}
                if item[0] in doc_length:   # loop through every term and retrieve tfidf^2 for the same doc
                    doc_length[item[0]] += math.pow(tf_idf, 2)
                else:
                    doc_length[item[0]] = math.pow(tf_idf, 2)
            tf_idf_doc_dict.update({term: sub_dict}) # {term: {doc: [tfidf, priority]}}
        return tf_idf_doc_dict, doc_length # tfidf [term: tfidf] doclenth [doc: doc length]
    
    def calculate_normalzie(self, tf_idf_dict, doc_length):
        for subdict in tf_idf_dict.values():
            for key, value in doc_length.items():
                if key in subdict:
                    subdict[key].append(subdict[key][0]/math.sqrt(value)) #value[0]: tf_idf, value[1]: priority score, value[2]: normalize of term
        return tf_idf_dict

    def get_paths(self, query):
        key = self.lemma.lemmatize(query.lower())
        value = self.single_word_dict[key]
        paths = [path[0].replace('\\', '/').split('/')[-2:] for path in value]
        paths = ['/'.join(path) for path in paths]
        return paths

    def get_urls(self, paths):
        # Open the JSON file
        with open('WEBPAGES_RAW/bookkeeping.json', "r") as file:
            # Load the contents of the file into a Python object
            data = json.load(file)

        urls_list = []
        for path in paths:
            urls_list.append(data[path])

        file_name = input('Enter File Name: ')
        f = open(file_name, "w",encoding="UTF-8")
        for url in urls_list:
            f.write(url + '\n')
        f.close()
                    
    def report(self, single_index, bigram_index):
        try:
            f3 = open("SINGLE_INDEX-VIEW.csv", "w",encoding="UTF-8")
            for token, posting in single_index.items():
                f3.write('{}: {}\n'.format(token, posting))
            f3.close()

            f2 = open("BIGRAM_INDEX-VIEW.csv", "w",encoding="UTF-8")
            for token, posting in bigram_index.items():
                f2.write('{}: {}\n'.format(token, posting))
            f2.close()

            single_index_file = open("SINGLE-INDEX.PKL", "wb")
            pickle.dump(single_index, single_index_file)
            single_index_file.close()

            desc_file = open("META.PKL", "wb")
            pickle.dump(self.doc_description, desc_file)
            desc_file.close()

            bigram_index_file = open("BIGRAM-INDEX.PKL", "wb")
            pickle.dump(bigram_index, bigram_index_file)
            bigram_index_file.close()

        except (Exception) as e:
            print(e)

class Milestone2:
    def __init__(self):
        self.lemma = WordNetLemmatizer()

        self.single_index = {}
        self.bigram_index = {}
        self.total_docs = 37497
        self.doc_description = {}

    def open_files(self) -> None:
        try:
            with open('SINGLE-INDEX.PKL', 'rb') as f1:
                self.single_index = pickle.load(f1)
            with open('BIGRAM-INDEX.PKL', 'rb') as f2:
                self.bigram_index = pickle.load(f2)
            with open('META.PKL', 'rb') as f3:
                self.doc_description = pickle.load(f3)
        except (Exception) as e:
            print (e)
        

    # stemming and lematizing token
    def tokenizer(self, raw_contet:str) -> list: 
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
        if word not in stop_words and len(word) >= 3:
            tokens.append(self.lemma.lemmatize(word))
        return tokens
    
    # calculate tf of query
    def check_frequency(self, query_list:list) -> dict:
        """Calculate raw_tf of query


        Return: dict {term: raw_TF}
        
        """
        frequency_dict = {}
        for word in query_list:
            if word in frequency_dict:
                frequency_dict[word] += 1
            else:
                frequency_dict[word] = 1 
        return frequency_dict


    def get_unigram_data(self, query: list) -> dict:
        """ Retrieve unigram posting data for query terms

            Return: {doc: [tfidf, norm, quality]}
        """
        posting_data = {}
        for term in query:
            if term in self.single_index:
                posting_data.update({term: self.single_index[term]})
        return posting_data
    
    def get_bigram_data(self, query: list) -> dict:
        """ Retrieve bigram posting data for query terms

            Return: {doc: [tfidf, norm, quality]}
        """

        posting_data = {}
        for bigram in query:
            if bigram in self.bigram_index:
                # add element to dict
                posting_data.update({bigram: self.bigram_index[bigram]})

        return posting_data
    

    def normalize_query(self, tf_idf_query_dict:dict, query_length: float) -> dict: 
        #tf_idf_query_dict = {term: tfidf}
        for key, value in tf_idf_query_dict.items():
            tf_idf_query_dict[key] = value / query_length
        return tf_idf_query_dict

    def calculate_tf_idf(self, query: list, is_bigram: bool) -> dict:
        """calculate tf_idf and normalize query"""
        tf_idf_query_dict= {}
        query_data = {} # An index of query
        query_length_sqr = 0
        
        # EXTRACT POSTING LISTS AND ADD TO QUERY_DATA BASED ON QUERY'S TERM
        if is_bigram:
            query_data = self.get_bigram_data(query)

            if len(query_data) == 0: 
                
                # convert bigram into unigram [(computer, science), (science, major)] 
                # -> [computer, science, major]
                query = list(itertools.chain(*query)) 
                is_bigram = False

        else:
            query_data = self.get_unigram_data(query)

        if len(query_data) == 0: #NOT FOUND IN BOTH INDEXES
            return {}, {} # Return 2 empty dicts            

        #  frequency_dict {term: raw_tf}
        frequency_dict = self.check_frequency(query) 
        
        # CALCULATING TF-IDF AND NORMALIZE
        for term in query:
            if term in query_data:
                tf = 1 + math.log10(frequency_dict[term])
                idf = math.log10(self.total_docs/len(query_data[term]))
                tf_idf = tf * idf

                #create tf_idf dict for query term = {query_term: tfidf}
                tf_idf_query_dict[term] = tf_idf
                
                # calculate length of query
                query_length_sqr += math.pow(tf_idf,2)     
        
        query_length = math.sqrt(query_length_sqr) 
        tf_idf_query_dict = self.normalize_query(tf_idf_query_dict, query_length) 
        
        #return two dicts passed to func() calculate_cosine_score below
        #   tf_idf_query_dict -> {query_term: normed_tfidf}
        #   {doc_query_term: {doc: [tfidf,score,norm]}}
        return tf_idf_query_dict, query_data
    
    # calculate cosine scores
    def calculate_cosine_score(self, tf_idf_query_dict: dict, query_data: dict) -> list:
        score_dict = {}
        for key, normalize_query in tf_idf_query_dict.items():
            sub_dict = query_data[key] # query_data[key] = posting list for query term
            for doc_key, value in sub_dict.items(): # value = [tfidf, score, norm]
                if doc_key in score_dict:
                    score_dict[doc_key] += ((normalize_query * value[2]) + value[1])  # dot product of norm query and tfidf + preexisting score
                else:
                    score_dict[doc_key] = ((normalize_query * value[2]) + value[1]) #[tfidf,score,norm]

        # return top 20 largest cosine scores of document by using heap 
        url_results = heapq.nlargest(20, score_dict.items(), key=lambda x: x[1])

        # export to CSV file raw and unranked net scores for query and save as CSV file
        f3 = open('SCORE_TABLE.CSV', "w",encoding="UTF-8")
        for doc, score in score_dict.items():
            f3.write('{}: {}\n'.format(doc, score))
        f3.close()



        
        # return list of tuples (doc, score)
        return url_results
    
    def get_urls(self, score_list):
        # Open the JSON file
        with open('WEBPAGES_RAW/bookkeeping.json', "r") as file:
            # Load the contents of the file into a Python object
            data = json.load(file)

        output_list = []
        for item in score_list: # item = WEBPAGES_RAW/65/04
            path = item[0].split('\\')
            output_list.append(path[1] + '/' + path[2])

        urls_list = []
        for path in output_list:
            urls_list.append(data[path])

        f3 = open('TOP-20-URLS.CSV', "w",encoding="UTF-8")
        for url in urls_list:
            f3.write('{}\n'.format(url))
        f3.close()

        return urls_list

    def get_description(self, score_list) -> list:
        desc = []
        for docID in score_list:
            desc.append(self.doc_description[docID[0]])
        return desc
    
    def get_title(self, score_list) -> list:
        title = []
        for docID in score_list:

            # dont say this part
            path = docID[0].split('\\')
            path = '/'.join(path)
            #dont say this part

            with open(path, "r", encoding="utf8") as f:
                #read the content of file, inculding <tag>
                content = f.read()
                soup = BeautifulSoup(content, 'html.parser')
                
                #Extract the title tags
                title_tag = soup.find('title')
                if title_tag is not None:
                    title.append(title_tag.text[:80]) # title length
                else:
                    title.append("NO TITLE AVAILABLE")
        
        return title