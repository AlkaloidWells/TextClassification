from model.data import *
from model.train2 import *
from textclassifer2 import *
from model.utils import *
import model.train  as trl
import  textclassifer as cl1
import model.train2  as trc
import  textclassifer2 as cl2

# dataframe = fetch_and_save_data()
# print(dataframe.head())

#trl.train()
# trc.train()

query ="iphone 15 promax cheap arround buea"
print("search query: ",query, "\n")
cat = cl1.classify_query(query)
print("results\n:", cat)

# result = cl2.classify_query("iphone 15 promax cheap arround buea")
# print(result)
