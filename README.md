# Simple spam classifier using Support Vector Machine
This is a python implementation of spam classification taught in famous Machine Learning course by Andrew Ng

Train and test set are provided in **spamTrain.mat**, **spamTest.mat**

Full dataset at [SpamAssassin Public Corpus](http://spamassassin.apache.org/old/publiccorpus/)

# Email preprocessing pipeline

  - Lower case email
  - Remove HTML tags
  - Replace URLs with 'httpaddr', emails with 'emailaddr', numbers with 'number', $ with 'dollar'
  - Remove non-alphanumeric characters
  - Tokenize
  - Poster stem

# Email features
Each email is represented by a N-vector. N is the number of word in dictionary (provided in **vocab.txt**). The dictionary is built by collecting most frequency words appear in dataset. Each **i**th-row in feature vector represent whether or not the **i**th word in dictionary appear in preprocessed email.

# Training
Email features are fed to a linear SVM classifier with C parameter set to 0.1. The classifier then have to classify between two class: 0-non spam, 1-spam

# Result
Train accuracy: **0.998250**

Test accuracy: **0.989000**

**Top 15 predictors for spam classifirer**

our click remov guarante visit basenumb dollar will price pleas most nbsp lo ga hour
