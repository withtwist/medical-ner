# Medical-NER
This was a project by us (_Simon Almgren_ and _Sean Pavlov_), trying to solve the task of _Named Entity Recognition_ (NER) for medical entities in Swedish. We investigated two methods of solving the task: a Recurrent Neural Network (RNN) model and a Word-Vector (WV) model. You'll find them in separate directories, [RNN is here](https://github.com/withtwist/medical-ner/tree/master/rnn-ner) and [WV is here](https://github.com/withtwist/medical-ner/tree/master/wv-ner). They have their own _readmes_ on how to get you started.

## Background
There are very good readings on both approaches. [This post by colah](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) gives a very good image of how a Recurrent Neural Network works and [the original paper by Zhang and Elhadad](http://www.ncbi.nlm.nih.gov/pubmed/23954592) that the Word-Vector model is based on describes the method in a good way. We have however provided a short summarization of the methods down below.

### Recurrent Neural Network Model
The method tries to solve the task by using inherent knowledge already in taxonomies to solve the problem. Using knowledge from external sources, it will be able to train the model to recognize named entities from different categories. The Recurrent Neural Network model will be trained to read through text and will predict the category that the different words or phrase in the text-stream falls into with the output. While it is a supervised method, the training data will be generated with the information from the taxonomies which will make it a semi-supervised approach. This model uses Google's [TensorFlow](https://www.tensorflow.org) to implement the neural network. A basic bidirectional Recurrent Neural Network (RNN) implementation was provided by [Olof Mogren](http://mogren.one).

### Word-Vector Model
The model is inspired by a model originally implemented by [Zhang and Elhadad](http://www.ncbi.nlm.nih.gov/pubmed/23954592). They developed a system for semi-supervised NER with clinical and biological texts in English as input documents. We  implemented this medical journals in Swedish in mind. The method consists of a number of steps and a preprocessing stage. The preprocessing needed is tokenization, sentence splitting and part-of-speech tagging. These were handled using a framework called [GATE](https://gate.ac.uk). Followed by noun phrase chunking which is done with a framework called [Swe-SPARK](http://stp.lingfil.uu.se/~bea/resources/spark/). The idea behind the method is that noun phrases are likely to be entities and chunking is a more efficient way of finding candidates of entities than doing a full parsing on the input documents. After preprocessing, there are three steps in the main algorithm: seed-term collection, boundary detection and entity classification. Seed-term collection is to find seed-terms for the predefined categories, terms that we know belongs to that category. Boundary detection is to find the boundaries of the entities in the input document. Entity classification is to classify the entities in the text. 

## Credits
Since we have used external frameworks and ideas, we would like to list them here to give them credits:
* [TensorFlow](https://www.tensorflow.org) - open source framework for machine learning
* [GATE](https://gate.ac.uk) - open source framework for text annotation
* [OpenNLP](http://opennlp.apache.org) - open source library for Natural Language Processing (NLP) tasks
* [Swe-SPARK](http://stp.lingfil.uu.se/~bea/resources/spark/) - a shallow parser for Swedish, identifying noun phrases
* [Olof Mogren](http://mogren.one) - provided a bidirectional Recurrent Neural Network (RNN) implementation in TensorFlow. 
* [Zhang and Elhadad](http://www.ncbi.nlm.nih.gov/pubmed/23954592) - their paper: Unsupervised biomedical named entity recognition: experiments with clinical and biological texts.
