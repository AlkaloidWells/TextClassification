import model.trainLR  as trl
import  textclassiferLR as cl1
import model.trainCNN  as trc
import  textclassiferCNN as cl2
import model.trainLRS  as trs
import  textclassiferLRS as cl3
import model.trainXB  as trx
import  textclassiferXB as cl4

# dataframe = fetch_and_save_data()
# print(dataframe.head())

query ="android phone"
print("search query: ",query, "\n")



# #trl.train()
# cat = cl1.classify_query(query)
# print("\nmodel 1 without smoth results\n\n:", cat)



# #trs.train()
# cat = cl3.classify_query(query)
# print("\n\nmodel 2 with smoth results\n\n:", cat)


# # trc.train()
# result = cl2.classify_query(query)
# print("\n\nmodel 3 cnn results\n:", result)


trx.train()
result = cl4.classify_query(query)
print("\n\nmodel 4 xgboost results\n:", result)