### Applications of NLP

- sentiment analysis
- feedback classification
- translation
-  summarization

### NLP: Building Text Cleanup and PreProcessing Pipeline

https://towardsdatascience.com/nlp-building-text-cleanup-and-preprocessing-pipeline-eba4095245a0


### NLP Roadmap 

https://medium.com/pythoneers/nlp-roadmap-of-algorithms-from-bow-to-bert-762527ac1a19

- The automation of this type of method is very hectic sometimes because of the varied data length.

- The methods do not have the capability to have a relationship with the previous words.

#### Tokenization Technique

Every text data have sentences made from words. It is a process to break the text sentences into words called tokens and the process is known as tokenization.

#### Lemmatization Technique

This technique is used to get the root word of the tokens in the data i.e. the token is happily and the root word is happy. It is a very useful technique in text pre-processing. The process of getting the root word is the lemmatization process.

#### Stop Words
This is a process to remove the most common tokens used in the text data that makes the corpus/document heavy for predictive analysis.

#### Concept of Bag of Words, TF-IDF, and N-grams

These techniques are useful to store the meaning of the tokens in relationship with other tokens that will be useful for predictive machine learning models. Whenever we work with text data, we need numeric data so that the machine can understand. These methods are useful because they convert the text tokens to numeric values/vectors so that the machine learning models process this semantic information between the information.


#### Pros BOW:

- This process counts the frequency of tokens.
- The implementation is very easy.
- The classification and feature extraction applications can be based on this technique.

#### Cons of BOW:

- The tokens increases in the bag as the length of the data increases.
- The sparsity in the matrix will also increase as the size of the input data increases. The number of zero’s in the sparsity matrix is more than non-zero numbers.
- There is no relationship/semantic connection with each other because the text is split into independent words.


#### Term Frequency-Inverse Document Frequency (TF-IDF)

- The output of the bag of words is used by TF-IDF Transformer and does the further process.
- This TF-IDF Vectorizer method takes the raw data as input and does further process.

#### Pros of TF-IDF:

- It slightly overcomes the semantic information between tokens.

#### Cons of TF-IDF:

- This method gives chance for the model to overfit.
- Not so much a semantic relationship between the tokens.

#### One Hot representation of Words

The encoding of the text can be done with the help of a one-hot method to map the text into numeric.

The one-hot encoding makes the token into vectors. Each word gets its index position to represent different vectors for different words.

The more big-size vectors are complicated to train in the machine learning model.

#### Word Embedding

These processes are word embedding techniques to overcome the issue of the big size vector representation. The feature representation is more focused on the matrix in these techniques. 



#### Types of Word2Vec
1. CBOW: The working methodology in this type is based on simple neural network architecture. It takes the context word i.e. the big sentences to get the output word.

2. Skip-Gram: In this type, the model takes a word in the input and gives the context of the words as output.


Pros:

- The semantic information can easily get in these techniques.
- The recommendation system applications can also be based on these techniques.

Cons:

- The sequence information is missing in the words but has some semantic information.
- The embedding is fixed in this technique.

### Deep learning algorithms in NLP

#### RNN

It is an advanced algorithm than we discussed above in natural language processing based on deep learning neural networks. This algorithm keeps the sequence information in the word document.


- It helps to keep the sequence information.

Vanishing Gradient and Gradient exploding problem in RNN: vanishing gradient slower the converge problem to the global minima and gradient exploding jumping left and right and not getting the global minima.

#### LSTM

It overcomes the problem if RNN and LSTM are long short-term memory.

It keeps the context word i.e. sentence as an input in LSTM.

In back-propagation, the parameters are very large so the training time increases due to many gates in LSTM.

The prediction of the future words depends on the previous inputs and the LSTM shows the problem in it.

GRU algorithm that comes in the family of RNN solved the problem of many gates in LSTM.

##### Bidirectional RNN

This algorithm predicts the future-based words by adding the backward layer of RNN to keep the sequence information of the previous words.

- It predicts the future based on words.
- In speech recognition applications this algorithm shows less accuracy because it processes all the input data at once.
- The training time is more and slower than the RNN algorithm.

#### Sequence to Sequence Models

This algorithm is from the RNN family algorithms.

This takes the sequence of data at once and processes it that gives the again sequence data output.

- Language translation
- Pop up phrases advice in chats

Encoder and Decoder are the main components of the seq2seq algorithm. LSTM and GRU are good models than RNN to use in encoder and decoder because of gradient descent problems in back-propagation.

Encoder: The gives the context vector in the output.

Decoder: The working of the decoder to convert the context vector to words again.


- If we increase the input data then it gives low accuracy.
- Encoder shows the issue in compressing the data.

#### Attention Models

The increased length issue is overcome by attention models in seq2seq by using bi-directional RNN or LSTM. It also predicts the context based on thoughts as a cognitive process. The layers of the neural network in this architecture are small because it has only three layers and the dot products of the RNN layer are more.

- The works well with lengthy sentences.
- The accuracy and bleu score increase.

- When we use the attention model with seq2seq RNN the optimization problem occurs.







