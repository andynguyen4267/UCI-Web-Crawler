import json
from index_constructor import *
from collections import defaultdict


def normalizeScore(docID, tfidf, dl=0):
    if docID =='query':         #If docID == 'query', use the doc_length specified in the argument
        doc_length = dl
        normalized = tfidf / doc_length
        return normalized
    else:                       #Else, get the doc_length from the json file  -->  if docID == "0/132" get the doc_length from json file for that particular document
        doc_length = docLength[docID]
        normalized = tfidf / doc_length
        return normalized


def searchForQuery(userInput, docuLength, bookkeeping):
    #words' normalized scores for each document  -->  {'0/27': {'computer': 0.09617813036613498, 'science': 0.10540192798327865}, '0/11': {'computer': 0.06946354220425044, 'science': 0.07612532334537692}}
    #to be compared with the query normalized scores later
    doc_normalized_score = defaultdict(dict)
    #words' normalized scores in the query
    #to be compared with each document that has the words
    queryResult = defaultdict(list)
    #queryLength to be used in the calculation of cosine similarity for the query
    queryLength = 0
    #Dictionary of docs, ranked by their normalized scores (highest comes first)
    rankedDocs = {}
    global docLength
    docLength = docuLength

    # Get user input and make it into a list
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')        
    tokens = tokenizer.tokenize(userInput.lower())
    lemmatizer = WordNetLemmatizer()
    userInputList = [lemmatizer.lemmatize(word) for word in tokens] # lemmatized user input
    # userInputList = userInput.lower().split(' ')
    
    stop_words = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    # dictionary of the tokens frequencies in the query i.e {word: freq, ...}
    wordFreqDict = computerWordFreq(userInputList)
    # Prevent repeating words in query
    query_repeated_tokens = set()


    '''QUERY NORMALIZATION PROCESS USING COSINE SIMILARITY'''
    #Get TF for each word in the query
    for word in userInputList:
        if word.isalnum() and word not in stop_words and word not in query_repeated_tokens:
            query_repeated_tokens.add(word)
            #compute the TF for current word in the query
            TF = computeTF(wordFreqDict[word])
            #Add word to the query dictionary (to be compared with docIDs in doc_normalized_score)  i.e {word: TF, ...}
            queryResult[word].append(TF)
            #The queryResult looks like this: {word: [TFIDF], word2:[TFIDF]}  -->  {computer: [1.213132], science: [1.543511]}

    # Calculate query length
    # Cite: VectorSpaceModel.pdf Slide 22 (The one under the table (Query Length))
    for k, v in queryResult.items():
        queryLength += (v[0]*v[0])
    queryLength = math.sqrt(queryLength)

    # Normalize queryResult dictionary with cosine similarity and add it to doc_normalized_score (for easy comparison with the other docs)
    # Cite: VectorSpaceModel.pdf Slide 22 (Left side table (Query))
    for k, v in queryResult.items():
        doc_normalized_score['query'][k] = normalizeScore('query', v[0], queryLength)      #queryLength is optional argument but we put it because it is a query (see normalizeScore() function for better understanding)
        #The doc_normalized_score looks like this:
        #{'query': {'computer': 0.7071067811865475,
        #           'science': 0.7071067811865475},
        #(other docs to be added here later)}


    # DOCS NORMALIZATION PROCESS USING COSINE SIMILARITY
    # For each word, find all the corresponding docIDs
    # Cite: VectorSpaceModel.pdf Slide 22 (Right side table (Document1))
    for word in userInputList:
        fileName = sortToken(word) + '.json'
        with open(fileName) as file:
            jIndex = json.load(file)
            try:
                postings = jIndex[word]   # { word : [ [docID, TF, tagsImportanceRating, TF_IDF, score], [docID, TF, tagsImportanceRating, TF_IDF, score] ] }
                #Normalize the docs with cosine similarity
                for doc in postings:
                    # calculating anchor words weight
                    # doc[4] += (0.1 * doc[5].count(word))
                    doc_normalized_score[doc[0]][word] = normalizeScore(doc[0], doc[4]) #  (docID, score)
                    #The doc_normalized_score looks like this:
                    #{'query': {'computer': 0.7071067811865475,
                    #           'science': 0.7071067811865475},
                    # '0/27': {'computer': 0.09617813036613498, (adding docs start here...)
                    #           'science': 0.10540192798327865},
                    #...}
            except:
                break

    # print(doc_normalized_score)

    '''RANKING PROCESS'''
    for doc in doc_normalized_score.keys():
        sum = 0
        if doc != 'query':
            for q in userInputList:
                if q in doc_normalized_score[doc]:
                    sum += doc_normalized_score['query'][q] * doc_normalized_score[doc][q]
            rankedDocs[doc] = sum

    # print(rankedDocs)
    #Rank from highest to lowest
    rankedDocs = sorted(rankedDocs, key=lambda x:(-rankedDocs[x],x))
    # print(rankedDocs)
    
    '''MAC version'''
    urlList = []
    for doc in rankedDocs:
        urlList.append( (doc,bookkeeping[doc]) )

    return {userInput: urlList}

    '''Windows Version'''
    # urlList = []
    # for doc in rankedDocs:
    #     splitDocID = doc.split('\\')
    #     docID = splitDocID[0] + '/' + splitDocID[1]
    #     urlList.append( (docID,bookkeeping[docID]) )
    # return {userInput: urlList}