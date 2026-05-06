"""
Common words in PDF file
Author: Gerhard Kling
"""

from rank2 import word_rank

#Arguments
file_name = "STR.pdf"

#Call function
top_words = word_rank(file_name, 50)

#print(top_words)