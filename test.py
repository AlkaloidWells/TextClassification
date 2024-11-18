from model.data import *
from model.train2 import *
from textclassifer2 import *
from model.utils import *
import model.train  as trl
import  textclassifer as cl1
import model.train2  as trc
import  textclassifer2 as cl2
import model.train3  as trs
import  textclassifer3 as cl3

# dataframe = fetch_and_save_data()
# print(dataframe.head())

query ="sofas"
print("search query: ",query, "\n")



#trl.train()
cat = cl1.classify_query(query)
print("\nmodel 1 without smoth results\n\n:", cat)



#trs.train()
cat = cl3.classify_query(query)
print("\n\nmodel 2 with smoth results\n\n:", cat)


# trc.train()
result = cl2.classify_query(query)
print("\n\nmodel 3 cnn results\n:", result)