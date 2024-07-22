import sys
from milestone2 import Milestone2
from gui import GUI
import tkinter as tk

if __name__ == "__main__":
    project1 = Milestone2()
    documents = project1.retrieval(sys.argv[1]) 

    # attempt to load existing index, otherwise create a new one
    try:
        print("Attempting to load an existing index...")
        project1.load_index("index4.pickle")
        print("Index loaded sucessfully.")
    except:
        print("Existing index could not be found.\nCreating a new index...")
        project1.handel_document(documents)
        #project1.save_index("index4.pickle")
    # project1.report() 

    # retrieve urls from query
    query = input('Enter Query that you would like to look for: ')
    while(query != '0'):
        
        paths = project1.get_paths(query)
        project1.get_urls(paths)
        query = input('Enter Query that you would like to look for(Enter 0 to exit): ')
        if query == '0':
            break