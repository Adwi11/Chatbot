import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import discord
nltk.download('punkt')
#client=discord.Client()
#client.run('NzA1MzAyMzM3OTE2NTAyMDI2.XqpupQ.VVCEsCsaf3bi5L4mrlPB-qXgjwQ')


with open("intents.json") as file:
    data = json.load(file)
try:

    with open("data.pickle","rb") as f:
        words,labels,training,output= pickle.load(f)                   #if the data is already preprocessed and into the respective lists it will skip the except section else it will execute it

except:
    words=[]
    labels=[]
    docs_x=[]
    docs_y=[]

                                             #/loading data/
    for intents in data["intents"]:                              #we are traversing through every word in the json file and stemming it using the nltk function
        for pattern in intents["patterns"]:                       #tokenize is for getting all the words in the ppattern which we will later stem, it is the same as separating everything with spaces and picking out the words
            wrds=nltk.word_tokenize(pattern)                   #basically we are stemming the words and bringing words down to theur root words and feeding it into wrds....example:whats up will be chnaged to what...as our model wont care about other words and only the main meaning of the word.
            words.extend(wrds)
            docs_x.append(wrds)                                   #basically like the training model i have done earlier we hac class label and class data
            docs_y.append(intents["tag"])

        if intents["tag"] not in labels:
            labels.append(intents["tag"])

                                                #/this is to check the number of words in our model/

    words=[stemmer.stem(w.lower()) for w in words if w != "?"]          #we do not want to input the question marks

    words=sorted(list(set(words)))          #set removes any duplicated words, list converts the data type back to list as set is its own data type, and sorted sorts the words

    labels=sorted(labels)           #there should be no dups in labels hence set is not needed like above and i.e nor is list

    training=[]
    output=[]

    out_empty=[0 for _ in range(len(labels))]                #as the input is strings and our neural network understands nnumbers we are creating bag of words(one hot encoded) its a list of the length of words each number in the list points to a word   in the sentence, we do this for both words and the labels

    for x,doc in enumerate(docs_x):
        bag=[]
        wrds=[stemmer.stem(w) for w in doc]     #to stem all the words in docs_x (we could have done this before when we load words in doc_x)

        for w in words:
            if w in wrds:
                bag.append(1)   #1 if the word exits in the input
            else :
                bag.append(0)

        output_row =out_empty[:]        #making copy of list
        output_row[labels.index(docs_y[x])] = 1         #we are going to see the labels list and see which index of the tag present in the sentence exist and put a 1 there

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)                  #we need to work with arrays for tflearn to feed the into the model
    output = numpy.array(output)
    with open("data.pickle","wb") as f:
        pickle.dump=((words,labels,training,output),f)

tensorflow.reset_default_graph()     #to basically reset (not imp)

#model

net= tflearn.input_data(shape=[None,len(training[0])])                       #since we input the words in training hence why the input layer will have the total number of words in our model
net= tflearn.fully_connected(net,8)                                       #this means we are going to add 8 neurons in the next layer of the neural networ"net"
net= tflearn.fully_connected(net,8)
net= tflearn.fully_connected(net,len(output[0]),activation="softmax")               #as out put contains the labels or classes the length of it will be the number of output neurons, softmax is used to predict the probability of each of the output neuron
net= tflearn.regression(net)

model=tflearn.DNN(net)                        #its a type of neural network
#try:
#model.load("model.tflearn")                 #if model.tflearn will sucessfully load then there is no need of training the model again or else we will go to the except statement
#except:
model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True)                   #epoch is number of time ithe model sees the same data, show _metric is same as metrics in tensorflow
model.save("model.tflearn")

def bag_of_words(s,words):                                  #we make this function to make the input of user into a bag of words, which is the way  out model sees and predicts
    bag=[0 for _ in range(len(words))]
    s_words=nltk.word_tokenize(s)
    s_words=[stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w==se:
                bag[i]=1                #i is the index of the word w


    return numpy.array(bag)

def chat():
    print("Start talking to me ! (type quit to stop me)")
    while True:
        inp=input("You:")                                       #it will show You: before every user input
        ("Bot:")
        if inp.lower()=="quit":
            break

        result=model.predict([bag_of_words(inp,words)])[0]               #error occurs without putting that[0]
        result_index=numpy.argmax(result)


        if result[result_index]>0.3 :

            tag=labels[result_index]


            for tg in data["intents"]:
                if tg['tag']== tag:
                    responses=tg['responses']

            print(random.choice(responses))
        else:
            print("Sorry i dint get that, try again!")


chat()
