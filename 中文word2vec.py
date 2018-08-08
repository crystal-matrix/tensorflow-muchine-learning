
#缺少一个中文分词的语料，程序好像是可以运行的

import tensorflow as tf
import numpy as np
from collections import Counter  #用于统计词频
import  jieba
import jieba.analyse

data_path = "yitiantulongji.txt"  #语料路径
#data_path=jieba.load_userdict("yitiantulongji.txt")
embedding_save_path = "embeddings.txt"  #把最后的emdeddings保存的路径
vocabulary_size = 8000  #词典大小
win_len = 1   #窗口长度，即取中心词左右两边各一个词

batch_size = 500  #一个batch中的训练数据的个数
embedding_size = 128  #生成的词向量的长度
num_sampled = 64  #负采样中用到的负样本的数量

#验证数据
valid_size=16 #抽取的验证单词数
valid_window=100 #验证单词只从频数最高的100个单词中抽取
valid_examples=np.random.choice(valid_window,valid_size,replace=False)#不重复在0——10l里取16个
#valid_examples = 16

'''
由分好词的语料生成word list
'''
def MakeDataToWordList():
    file = open(data_path,'r',encoding = 'utf-8')
    words = []
    data = file.readlines()
    for aline in data:
        aline = aline.strip('\n').split(' ')
        for oneword in aline:
            if oneword != '' and oneword != '\t' :
                words.append(oneword)
    return words


'''
生成由(word,count)这样的pair组成的list
'''
def MakeSortedCountlist(allwords):
    #统计词频top49999,count计数器的第一个放unk
    count = [('unk',-1)]
    counter = Counter(allwords).most_common(vocabulary_size-1)  #统计词频 TOP4999
    count.extend(counter)
    print(count[1:10])
    return count


'''
dict用来保存词和词序号 word：num
reversedict用来保存词序号和词  num：word
'''
def MakeDictAndReverseDict(countlist):
    #worddict:{word1:num1,...}，num就是词在词频表（countlist）中对应的序号
    worddict = {}
    for word, _ in countlist:
        worddict[word] = len(worddict)
    #reversedict:{num1:word1,...}
    reversedict = dict(zip(worddict.values(),worddict.keys()))
    #print(reversedict)
    return worddict,reversedict

'''
将data_path中的word替换成word_num，构成训练语料
'''
def MakeTrainData(worddict,reversedict):
    #将原始分词后的语料，转换成句子的list，list中的每一个元素都是一个句子
    #input_sentence: [sub_sent1, sub_sent2, ...]
    #每个sub_sent是一个单词序列，例如['这次','大选','让']
    file = open(data_path,'r',encoding = 'utf-8')
    sentences = file.readlines()
    input_sentence = []  #所有句子列表

    for asentence in sentences:
        asentence = asentence.strip('\n').split(' ')
        for aword in asentence:
            if aword == '' or aword == '\t' :
                asentence.remove(aword)
        input_sentence.append(asentence)

    #sentence_num = len(input_sentence)
    file.close()
    all_inputs = []  #中心词的序号组成的vector
    all_labels = []  #中心词对应的上下文的序号组成的vector

    for sentence in input_sentence:
        for i in range(len(sentence)):
            start = max(0,i - win_len)
            end = min(len(sentence), i + win_len + 1)
            for index in range(start,end):
                #index对应的是上下文词，即label
                #i对应的是中心词
                if(index == i):
                    continue
                else:
                    if sentence[i] in worddict and sentence[index] in worddict:
                        input_id = worddict[sentence[i]]
                        label_id = worddict[sentence[index]]
                        all_inputs.append(input_id)
                        all_labels.append(label_id)
                    else:
                        continue

    all_inputs = np.array(all_inputs,dtype = np.int32)
    all_labels = np.array(all_labels,dtype = np.int32)
    all_labels = np.reshape(all_labels,[len(all_labels),1])
    return all_inputs, all_labels


'''
定义模型的输入输出，损失函数以及优化器
'''
def SkipGram(all_inputs,all_labels,reverse_dict):
    train_inputs = tf.placeholder(tf.int32,shape = [batch_size])
    train_labels = tf.placeholder(tf.int32,shape = [batch_size, 1])
        #所有词的embedding matrix
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1,1))
        #获取batch_size个embedding，用于训练
    embed = tf.nn.embedding_lookup(embeddings,train_inputs)

    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size],
                                                  stddev=1.0/np.math.sqrt(embedding_size)))
    nce_bias = tf.Variable(tf.zeros([vocabulary_size]))

    loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,
                                         biases = nce_bias,
                                         labels = train_labels,
                                         inputs = embed,
                                         num_sampled = num_sampled,
                                         num_classes = vocabulary_size))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        #验证数据
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),axis=1,keep_dims = True))
    normalized_embeddings=embeddings/norm   #除以其L2范数后得到标准化后的normalized_embeddings

    valid_embeddings=tf.nn.embedding_lookup(normalized_embeddings,valid_examples)    #如果输入的是64，那么对应的embedding是normalized_embeddings第64行的vector
    similarity=tf.matmul(valid_embeddings,normalized_embeddings,transpose_b=True)   #计算验证单词的嵌入向量与词汇表中所有单词的相似性
    print('相似性：',similarity)

    #加载数据，进行训练
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        average_loss = 0.0

        step_num = len(all_inputs) // batch_size  #23123

        for i in range(step_num):
            batch_inputs = all_inputs[i*batch_size:(i+1)*batch_size]
            batch_labels = all_labels[i*batch_size:(i+1)*batch_size]
            feed_dict = {train_inputs:batch_inputs,train_labels:batch_labels}

            _,loss_val = sess.run([optimizer,loss],feed_dict = feed_dict)

            average_loss += loss_val

            if i % 1000 == 0: #每1000个batch计算一次平均的loss
                average_loss /= 1000
                print("loss at iter ",i,": ",average_loss)
                average_loss = 0

            #每5000个batch，计算验证word与全部word的相似度
            #并将与每个验证word最相似的8个找出类
            if i % 5000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    #得到验证单词
                    valid_word = reverse_dict[valid_examples[i]]
                    top_n = 8
                    nearest = (-sim[i,:]).argsort()[1:top_n + 1]

                    log_str = "Nearest to %s: " % valid_word
                    for k in range(top_n):
                        close_word = reverse_dict[nearest[k]]
                        log_str = "%s %s," % (log_str,close_word)
                    print(log_str)

        final_embedding = normalized_embeddings.eval()

    file = open(embedding_save_path,'w',encoding = 'utf-8')
    with open(embedding_save_path,'w',encoding = 'utf-8') as file:
        for i in range(len(final_embedding)):
            word = reverse_dict[i]
            vector = []
            for j in range(len(final_embedding[i])):
                vector.append(final_embedding[i][j])
            file.writelines(word + '：' + str(vector) + '\n')


if __name__ == '__main__':

    words = MakeDataToWordList()

    count_list = MakeSortedCountlist(words)

    word_dict, reverse_dict = MakeDictAndReverseDict(count_list)

    all_inputs,all_labels = MakeTrainData(word_dict,reverse_dict)

    SkipGram(all_inputs,all_labels,reverse_dict)