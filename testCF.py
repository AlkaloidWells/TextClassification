import Classifiers.textclassiferLR as cl1
import Classifiers.textclassiferLRS as cl2
import Classifiers.textclassiferCNN as cl3
import Classifiers.textclassiferXB as cl4
import Classifiers.textclassiferRF as cl5

query = "iphone iphone gb fairly used but works perfectly battery life at"

# Test model 1: Logistic Regression without SMOTE
cat = cl1.classify_query(query)
print("\nModel 1 (Logistic Regression without SMOTE) results:\n", cat)

# Test model 2: Logistic Regression with SMOTE
cat = cl2.classify_query(query)
print("\nModel 2 (Logistic Regression with SMOTE) results:\n", cat)

# Test model 3: CNN
cat = cl3.classify_query(query)
print("\nModel 3 (CNN) results:\n", cat)

# Test model 4: XGBoost
cat = cl4.classify_query(query)
print("\nModel 4 (XGBoost) results:\n", cat)

# Test model 5: Random Forest
cat = cl5.classify_query(query)
print("\nModel 5 (Random Forest) results:\n", cat)