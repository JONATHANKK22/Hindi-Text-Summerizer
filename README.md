# Hindi-Text-Summarizer
***NLP Project.***


# Final Year Project Ideas
Text Summarization (TS) is a process to generate a summary while preserving the essence, by eliminating irrelevant or redundant content from the text. TS provides vital information in a much shorter version, usually reduced to less than half of the length of the input text. It remedies the challenge of information overload  (Too much data) and helps in information retrieval (IR) tasks.


### Problem Formulation
The amount of data available online is limitless. Think about a normal college student, who has to go through thousands of pages of documents each semester. That is why text summarization is necessary.  
The main purpose of text summarization is to get the most precise and useful information from a large document and eliminate the irrelevant or less important ones. Text summarization can be done either manually, which is time-consuming, or through machine algorithms and AIs, which takes very little time and is a better option.

### Motivation
Text Summarization is an active field of research in both the Information Retrieval and Natural Language Processing communities. Text Summarization is increasingly being used in the commercial sector such as the Telephone communication industry, data mining of text databases, web-based information retrieval, in word Processing tools. Many approaches differ in the behavior of their problem formulations. High-quality summarization requires sophisticated NLP techniques.

### Scope of The Project
The scope of our project  is to quickly and accurately summarize any piece of information in the Hindi language. A trained model is used to output the summary of the text. This can be used to get the context of the information given in Hindi.

### Use-case diagram and description

- **User**:
  Will upload input text documents to the system for Summarization.
- **System**:
  It will take input and Perform text processing where it removes stopwords using the stopwords dataset and finally by using an algorithm it will generate summarized text.

### Activity Diagram

The flow begins by providing an input text document. after providing input the preprocessing of text takes place where tokenization, stopword removal, stemming, and word tagging happen. Furthermore, we calculate word frequency and TF-IDF value of  words. we calculate average values of word count and generate text summary. 

### Data Flow Diagram



### Architectural Design


1. **Tokenization**: Tokenization is the process of breaking up a paragraph into smaller units such as sentences or words. The fundamental principle of Tokenization is to try to understand the meaning of the text by analyzing the smaller units or tokens that constitute the paragraph. To do this, we shall use the Natural Language Toolkit for Indic Languages (iNLTK). library. NLTK is the Natural Language Toolkit library in python that is used for Text Preprocessing.
 2. **Stop word removal**: Stop word removal is one of the most commonly used preprocessing steps across different NLP applications. The idea is simply removing the words that occur commonly across all the documents in the corpus. Typically, articles and pronouns are generally classified as stop words. These words have no significance in some of the NLP tasks like information retrieval and classification, which means these words are not very discriminative. For this, we can remove them easily, by storing a list of words that you consider to stop words. NLTK(Natural Language Toolkit) in python has a list of stopwords stored in 16 different languages. You can find them in the nltk_data directory. 
3. **Frequency calculation of words**: Frequency calculation create a frequency table which is a dictionary having words as keys and their frequency or number of times that word have appeared in the corpus as value.


In pretty much the same way that you were able to train & evaluate a simple neural network above in a few lines,
you can use Keras to quickly develop new training procedures or exotic model architectures.
Here's a low-level training loop example, combining Keras functionality with the TensorFlow `GradientTape`:

```python
import tensorflow as tf

# Prepare an optimizer.
optimizer = tf.keras.optimizers.Adam()
# Prepare a loss function.
loss_fn = tf.keras.losses.kl_divergence

# Iterate over the batches of a dataset.
for inputs, targets in dataset:
    # Open a GradientTape.
    with tf.GradientTape() as tape:
        # Forward pass.
        predictions = model(inputs)
        # Compute the loss value for this batch.
        loss_value = loss_fn(targets, predictions)

    # Get gradients of loss wrt the weights.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    # Update the weights of the model.
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```



---

## Installation

**Algorithm**

Term Frequency * Inverse Document Frequency

This is a technique to quantify words in a set of documents. We generally compute a score for each word to signify its importance in the document and corpus. This method is a widely used technique in Information Retrieval and Text Mining.

A High weight in **TF-IDF** is reached by a high term frequency(in the given document) and a low document frequency of the term in the whole collection of documents.

It is easier for any programming language to understand textual data in the form of numerical value. So, for this reason, we vectorize all of the text so that it is better represented.

TF-IDF algorithm is made of 2 algorithms multiplied together.

Term Frequency
**Term frequency (TF)** is This measures the frequency of a word in a document. divided by how many words there are.

**TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)**

Inverse document frequency
Term frequency is how common a word is, inverse document frequency (IDF) is how unique or rare a word is.

**IDF(t) = log_e(Total number of documents / Number of documents with term t in it)**

For Example,
Consider a document containing 100 words wherein the word apple appears 5 times. The term frequency (i.e., TF) for apple is then (5 / 100) = 0.05.

Now, assume we have 10 million documents and the word apple appears in one thousand of these. Then, the inverse document frequency (i.e., IDF) is calculated as log(10,000,000 / 1,000) = 4.

Thus, the TF-IDF weight is the product of these quantities: 0.05 * 4 = 0.20.
**TF-IDF = TF(t) * IDF(t)**



