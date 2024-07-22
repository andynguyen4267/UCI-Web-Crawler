from tkinter import *
import os
from milestone2 import Milestone1
from milestone2 import Milestone2
import nltk
import textwrap

root = Tk()
root.title('INF 141 SEARCH PROGRAM')
root.geometry("700x650")

CORPUS_PATH = 'WEBPAGES_RAW'  
SINGLE_INDEX = 'SINGLE-INDEX.PKL'
BIGRAM_INDEX = 'BIGRAM-INDEX.PKL'
# Cache File of Doc Description
META_FILE = 'META.PKL'



# Check for existing INDEX:
if os.path.isfile(SINGLE_INDEX) and os.path.isfile(BIGRAM_INDEX) and os.path.isfile(META_FILE):
    start = Milestone2()
    print ("\n\n\n\n------------LOADING -- STARTING IN A FEW SECONDS ----------\n\n\n")
    start.open_files()
    
else:
    #IF INDEX NOT NOT FOUND, GENERATE INDEX
    index = Milestone1()
    documents = index.retrieval(CORPUS_PATH) 
    index.handle_document(documents)
    bigram_dict, singe_dict = index.return_dict()
    tf_idf_single, single_doc_length = index.calculate_tf_idf(singe_dict)
    tf_idf_bigram, bigram_doc_length = index.calculate_tf_idf(bigram_dict)

    single_index = index.calculate_normalzie(tf_idf_single, single_doc_length)
    bigram_index = index.calculate_normalzie(tf_idf_bigram, bigram_doc_length)
    index.report(single_index, bigram_index) # CREATE UNIGRAM AND BIGRAM INDEX


def clear():
    my_text.delete(0.0, END)

   
def search():
    bigram_check = False
    query_list = []
    result = {}
    cosine_score_dict = []
    query = entry_box.get()

    if ' ' in query:
        bigram_query = start.tokenizer(query)
        query_list = list(nltk.bigrams(bigram_query))
        bigram_check = True
    else:
        query_list = start.tokenizer(query)
        
    tf_idf_query_dict, term_index = start.calculate_tf_idf(query_list, bigram_check)
    if len(tf_idf_query_dict) != 0:
        cosine_score_dict = start.calculate_cosine_score(tf_idf_query_dict, term_index)
        result_ulrs = start.get_urls(cosine_score_dict)
        title = start.get_title(cosine_score_dict)
        desc = start.get_description(cosine_score_dict)
        final_result = ""

        for i,result in enumerate(result_ulrs):
            final_result += "TITLE: "
            final_result += title[i].upper()

            final_result += "\n\n \tURL:"
            final_result += result_ulrs[i]

            final_result += "\n\n \tDESCRIPTION: "
            final_result += textwrap.fill(desc[i],80)
            final_result += "\n\n"

     
    clear()
    if len(result) != 0:
        my_text.insert(0.9, final_result)
    else:
        my_text.insert(0.9, "NO RESULTS")



# GRAPHING INTERFACE

tk_frame = LabelFrame(root, text = "PLEASE ENTER SEARCH QUERY")
tk_frame.pack(pady=20)

entry_box = Entry(tk_frame, font=('Helvetica', 18), width=47)
entry_box.pack(pady=20, padx=20)

frame = Frame(root)
frame.pack(pady=5)

text_scroll = Scrollbar(frame)
text_scroll.pack(side=RIGHT, fill = Y)

hor_scroll = Scrollbar(frame, orient='horizontal')
hor_scroll.pack(side=BOTTOM, fill=X)

my_text = Text(frame, yscrollcommand=text_scroll.set, wrap="none", xscrollcommand=hor_scroll.set)
my_text.pack()

#configure scrollbar
text_scroll.config(command=my_text.yview)
hor_scroll.config(command=my_text.xview)

#button
button_frame = Frame(root)
button_frame.pack(pady=10)

#button
search_button = Button(button_frame, text="SEARCH", font=('Helvetica', 15), fg="#3a3a3a",command=search)
search_button.grid(row=0, column=0,padx=20)

search_button = Button(button_frame, text="CLEAR", font=('Helvetica', 15), fg="#3a3a3a",command=clear)
search_button.grid(row=0, column=1)


root.mainloop()

#Reference on GUI:  Build A Wikipedia Search App - Python Tkinter GUI Tutorial #169 
#               https://youtu.be/iBiAmmqIcyk 