from model.data import *
from model.train import *
from textclassifer import *
from model.utils import *

# dataframe = fetch_and_save_data()
# print(dataframe.head())

#train()
query ="apple cider veniga"
print("search query: ",query, "\n")
cat = classify_query(query)
print("results\n:", cat)
