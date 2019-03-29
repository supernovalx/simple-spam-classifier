from scipy.io import loadmat
from processEmail import *
import os
from sklearn.svm import SVC

def read(path):
    return open(path).read()

def testEmail(path,clf):
    print("Test classifier on " + path)
    email = read(path)
    email_features = emailFeatures(preprocessEmail(email))
    print("Classified as %d" % clf.predict(email_features))

print("Loading data...")
train = loadmat('spamTrain.mat')
test = loadmat('spamTest.mat')
X, y = train['X'], train['y'][:,0]
X_test, y_test = test['Xtest'], test['ytest'][:,0]

print("Training...")
clf = SVC(C=0.1,gamma="auto",kernel="linear")
clf.fit(X, y) 

print("Train accuracy: %f" % clf.score(X,y))
print("Test accuracy: %f" % clf.score(X_test,y_test))

print("Top 15 predictors for spam classifirer: ")
vocab = list(loadVocalList())
w = np.argsort(clf.coef_[0])[::-1] 

for i in range(15):
    print(vocab[w[i]])

testEmail("emailSample1.txt",clf)
testEmail("emailSample2.txt",clf)
testEmail("emailSample3.txt",clf)





