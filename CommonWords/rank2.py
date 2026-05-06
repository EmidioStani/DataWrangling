import spacy
from collections import Counter
from gensim.parsing.preprocessing import remove_stopwords
import re
import csv
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy
from spacy.matcher import Matcher 
from spacy.tokens import Span 

from pyvis.network import Network

import urllib.request 
from bs4 import BeautifulSoup

import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm

from spacypdfreader import pdf_reader
from IPython.display import display, HTML

pd.set_option('display.max_colwidth', 200)

nlp = spacy.load("en_core_web_md")



def word_rank(file_name: str, top_num: int = 10):
    doc = pdf_reader(file_name, nlp)
    # for i in doc.sents:
    sentences = [[re.sub('\n', '', i)] for i in doc.sents]
    for i in sentences:
      sentences[i] = [re.sub('\n', '', i)
    myheaders = ['sentence']
    myvalues = sentences
    filename = 'article_text2.csv'
    with open(filename, 'w',newline='', encoding='utf-8') as myfile:
        writer = csv.writer(myfile)
        writer.writerow(myheaders)
        writer.writerows(myvalues)
    
    csv_sentences = pd.read_csv("article_text2.csv")
    # print(csv_sentences['sentence'].sample(5))
    entity_pairs = []

    for i in tqdm(csv_sentences["sentence"]):
       entity_pairs.append(get_entities(i))

    # print(entity_pairs[10:20])
    relations = [get_relation(i) for i in tqdm(csv_sentences['sentence'])]

    #print(pd.Series(relations).value_counts()[:50])

    source = [i[0] for i in entity_pairs]

    # extract object
    target = [i[1] for i in entity_pairs]
    kg_df = pd.DataFrame({'source':source, 'target':target, 'label':relations})
    # create a directed-graph from a dataframe
    G=nx.from_pandas_edgelist(kg_df, "source", "target", 
                          edge_attr=True)
    plt.figure(figsize=(12,12))
    pos = nx.spring_layout(G)
    e_labels = {(kg_df.source[i], kg_df.target[i]):kg_df.label[i]
          for i in range(len(kg_df['label']))}
    nx.draw_networkx_edge_labels(G, pos, edge_labels= e_labels, font_color='red')
    #nx.draw(G, pos = pos,with_labels=True)
    #nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
    edge_labels = nx.get_edge_attributes(G, "label")
    print(edge_labels)
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    net = Network(directed=True, select_menu = True, filter_menu = True)
    net.show_buttons()
    net.from_nx(G)

    html = net.generate_html()
    with open("article2.html", mode='w', encoding='utf-8') as fp:
        fp.write(html)
    display(HTML(html))
    
    #net.show("test.html", notebook=False)
    #plt.show()

    '''
    nlp = spacy.load("en_core_web_md")

    nlp.add_pipe("entityLinker", last=True)

    doc = pdf_reader(file_name, nlp)

    pages = doc._.last_page

    #Combine list of words

    all_words = []

    current_page = doc._.linkedEntities
    for sent in doc.sents:
            sent._.linkedEntities.pretty_print()
    
    for i in range(1, pages):
        current_page = doc._.linkedEntities.page(i)
        for sent in doc.sents:
            sent._.linkedEntities.pretty_print()
        for token in current_page:
            text = token.text

            text = text.lower()
            #Remove stopwords
            text_filter = remove_stopwords(text)
            #Split string into list
            text_list = text_filter.split()
            # print(text_list)
            all_words.extend(text_list)

    #Remove symbols
    all_words_alnum = [word for word in all_words if word.isalnum()]
    ignore_list = []
    all_words_filtered = [word for word in all_words_alnum if word not in ignore_list]

	#Use Counter class and the most_common method
    top_words = Counter(all_words_filtered).most_common(top_num)
	  
    return top_words
    '''

def get_entities(sent):
  ## chunk 1
  ent1 = ""
  ent2 = ""

  prv_tok_dep = ""    # dependency tag of previous token in the sentence
  prv_tok_text = ""   # previous token in the sentence

  prefix = ""
  modifier = ""

  #############################################################
  
  for tok in nlp(sent):
    ## chunk 2
    # if token is a punctuation mark then move on to the next token
    if tok.dep_ != "punct":
      # check: token is a compound word or not
      if tok.dep_ == "compound":
        prefix = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          prefix = prv_tok_text + " "+ tok.text
      
      # check: token is a modifier or not
      if tok.dep_.endswith("mod") == True:
        modifier = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          modifier = prv_tok_text + " "+ tok.text
      
      ## chunk 3
      if tok.dep_.find("subj") == True:
        ent1 = modifier +" "+ prefix + " "+ tok.text
        prefix = ""
        modifier = ""
        prv_tok_dep = ""
        prv_tok_text = ""      

      ## chunk 4
      if tok.dep_.find("obj") == True:
        ent2 = modifier +" "+ prefix +" "+ tok.text
        
      ## chunk 5  
      # update variables
      prv_tok_dep = tok.dep_
      prv_tok_text = tok.text
  #############################################################

  return [ent1.strip(), ent2.strip()]

def get_relation(sent):

  doc = nlp(sent)

  # Matcher class object 
  matcher = Matcher(nlp.vocab)

  #define the pattern 
  pattern = [{'DEP':'ROOT'}, 
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},  
            {'POS':'ADJ','OP':"?"}] 

  matcher.add("matching_1",[pattern]) 

  matches = matcher(doc)
  k = len(matches) - 1
  if (k != -1):
    span = doc[matches[k][1]:matches[k][2]].text
  else:
    span = ""
  return(span)