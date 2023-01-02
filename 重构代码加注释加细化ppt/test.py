from collections import Counter
from tkinter import Tk,Label,Text,Button,Canvas  # 使用Tkinter前需要先导入
from paddle import nn,concat,unsqueeze,squeeze,matmul,add,get_default_dtype,to_tensor,tolist,Model
def readfile(src):
    with open(src,'r',encoding="utf-8") as source:
        lines=source.readlines()
    return lines
def cleanSentence(sen_list):#将数据集中文本按字符拆成序列，每一条数据由<start>开始，由<end>结束
    newlist=[]
    for i in range(len(sen_list)):
        newlist.append(["<start>"]+sen_list[i].strip().split()+['<end>'])
    return newlist
def createDict(data_up,data_down, frequency):#建立词空间字典
    crpous = []#将上下联拆分后的语料合并便于建立字典
    for i in data_up:
        crpous.extend(i)
    for i in data_down:
        crpous.extend(i)
    word_freq_dict = {}# 统计不同汉字的使用频率，记录到字典中
    for ch in crpous:
        if ch not in word_freq_dict:
            word_freq_dict[ch] = 0
        word_freq_dict[ch] += 1
    word_freq_dict = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)# 根据频率对字典进行排序
    wi = {}#     wi={'<pad>':0,"<unk>":1}用于完成，词到id的映射，便于向量化
    iw = {}#     iw={0:'<pad>',1:'<unk>'}用于完成由id转化为词，便于结果输出
    for word, freq in word_freq_dict:
        if freq > frequency:#频率小于阈值被设为1，减少影响
            id = len(wi) # 按照频率，从高到低，开始遍历每个单词，依次加入字典，加入次序即为id
            wi[word] = id
            iw[id] = word
        else:
            wi[word] = 1
    return wi, iw
class Encoder(nn.Layer):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)#embedding层
        self.lstm = nn.LSTM(input_size=embedding_dim,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=0.2 if num_layers > 1 else 0)
    # src_length 的形状为[batch_size],作用是控制inputs中的time_step超过[batch_size]的不再更新状态，就是那些填充
    def forward(self, src, src_length):
        inputs = self.embedding(src)  # [batch_size,time_steps,embedding_dim]
        encoder_out, encoder_state = self.lstm(inputs,sequence_length=src_length)
        return encoder_out, encoder_state
class Decoder(nn.Layer):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_attention = nn.RNN(DecoderCell(num_layers, embedding_dim, hidden_size))
        self.fianl = nn.Linear(hidden_size, vocab_size)

    def forward(self, trg, decoder_initial_states, encoder_output, encoder_padding_mask):
        # trg 的形状为 [batch_size,sequence_length]
        # embedding 之后, [batch_size,sequence_length,embedding_dim]
        inputs = self.embedding(trg)

        # decodr_out [batch_szie,hidden_size]
        decoder_out, _ = self.lstm_attention(inputs,
                                             initial_states=decoder_initial_states,
                                             encoder_out=encoder_output,
                                             encoder_padding_mask=encoder_padding_mask)

        # predict [batch_size,sequence_len,word_size]
        predict = self.fianl(decoder_out)


        return predict
class DecoderCell(nn.RNNCellBase):
    def __init__(self, num_layers, embedding_dim, hidden_size):
        super(DecoderCell, self).__init__()

        self.dropout = nn.Dropout(0.2)
        self.lstmcells = nn.LayerList([nn.LSTMCell(
            input_size=embedding_dim + hidden_size if i == 0 else hidden_size,
            hidden_size=hidden_size
        ) for i in range(num_layers)])

        self.attention = AttentionLayer(hidden_size)

    def forward(self, decoder_input, decoder_initial_states, encoder_out, encoder_padding_mask=None):
        # forward 函数会执行squence_len次 ，每次的decoder_input 为[batch_size,embeddding_dim]

        # 状态分解 states [encoder_final_states,decoder_init_states]
        # encoder_final_states [num_layes,batch_size,hiden_size]
        # decoder_init_states []

        encoder_final_states, decoder_init_states = decoder_initial_states

        # num_layers=len(encoder_final_states[0])
        # decoder_init_states=lstm_init_state


        new_lstm_states = []

        # decoder_input: [batch_size,embedding_dim]

        inputs = concat([decoder_input, decoder_init_states], 1)


        for i, lstm_cell in enumerate(self.lstmcells):
            # inputs 的形状为 [batch_size,input_size]  input_size:输入的大小
            state_h, new_lstm_state = lstm_cell(inputs, encoder_final_states[i])

            inputs = self.dropout(state_h)

            new_lstm_states.append(new_lstm_state)

        state_h = self.attention(inputs, encoder_out, encoder_padding_mask)
        return state_h, [new_lstm_states, state_h]
class AttentionLayer(nn.Layer):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attn1 = nn.Linear(hidden_size, hidden_size)
        self.attn2 = nn.Linear(hidden_size + hidden_size, hidden_size)

    def forward(self, decoder_hidden_h, encoder_output, encoder_padding_mask):
        encoder_output = self.attn1(encoder_output)  # [batch_size,time_steps,hidden_size]

        # decodr_hidden_h 的形状 [batch_size,hidden_size],是lstm公式中的ht.
        # unsqueeze之后[batch_size,1,hidden_size]
        # transpose_y=True,后两维转置 [batch_size,hidden_size,time_steps]
        # matmul之后的 形状 [batch_size,1,time_steps]
        a = unsqueeze(decoder_hidden_h, [1])

        attn_scores = matmul(a, encoder_output, transpose_y=True)

        # 注意力机制中增加掩码操作，在padding 位加上个非常小的数：-1e9
        if encoder_padding_mask is not None:
            # encoder_padding_mask的形状为[batch_size,1,time_steps]
            attn_scores = add(attn_scores, encoder_padding_mask)

        # softmax操作，默认是最后一个维度，axis=-1,形状不变
        attn_scores = nn.functional.softmax(attn_scores)

        # [batch_size,1,time_steps]*[batch_size,time_steps,hidden_size]-->[batch_size,1,hidden_size]
        # squeeze之后：[batch_size,hidden_size]
        attn_out = squeeze(matmul(attn_scores, encoder_output), [1])

        # concat之后 [batch_size,hidden_size+hidden_size]
        attn_out = concat([attn_out, decoder_hidden_h], 1)

        # 最终结果[batch_size,hidden_size]
        attn_out = self.attn2(attn_out)

        return attn_out
class Seq2Seq(nn.Layer):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, eos_id):
        """
        :param vocab_size: 词空间大小
        :param embedding_dim: embedding层大小
        :param hidden_size: 隐藏层大小
        :param num_layers: 隐藏层数
        :param eos_id:结束标志
        """
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.eos_id = eos_id
        self.num_layers = num_layers
        self.INF = 1e9
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_size, num_layers)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_size, num_layers)
    def forward(self, src, src_length, trg):
        encoder_output, encoder_final_state = self.encoder(src, src_length)
        encoder_final_states = [(encoder_final_state[0][i], encoder_final_state[1][i]) for i in range(self.num_layers)]

        decoder_initial_states = [encoder_final_states,
                                  self.decoder.lstm_attention.cell.get_initial_states(batch_ref=encoder_output,
                                                                                      shape=[self.hidden_size])]
        src_mask = (src != self.eos_id).astype(get_default_dtype())
        encoder_mask = (src_mask - 1) * self.INF
        encoder_padding_mask = unsqueeze(encoder_mask, [1])
        predict = self.decoder(trg, decoder_initial_states, encoder_output, encoder_padding_mask)
        return predict
class Seq2SeqInfer(Seq2Seq):#进行模型调用的网络结构
    def __init__(self, word_size, embedding_dim, hidden_size, num_layers, bos_id, eos_id, beam_size,
                 max_out_len):
        self.bos_id = bos_id
        self.beam_size = beam_size
        self.max_out_len = max_out_len
        self.num_layers = num_layers
        super(Seq2SeqInfer, self).__init__(word_size, embedding_dim, hidden_size, num_layers, eos_id)
        self.beam_search_decoder = nn.BeamSearchDecoder(
            self.decoder.lstm_attention.cell,
            start_token=bos_id,#句子开始标志的id
            end_token=eos_id,#句子结束标志的id
            beam_size=beam_size,
            embedding_fn=self.decoder.embedding,
            output_fn=self.decoder.fianl)
    def forward(self, src, src_length):
        encoder_output, encoder_states = self.encoder(src, src_length)
        encoder_final_state = [(encoder_states[0][i], encoder_states[1][i]) for i in range(self.num_layers)]
        # 初始化decoder的隐藏层状态
        decoder_initial_states = [encoder_final_state,
                                  self.decoder.lstm_attention.cell.get_initial_states(batch_ref=encoder_output,
                                                                                      shape=[self.hidden_size])]
        src_mask = (src != self.eos_id).astype(get_default_dtype())
        encoder_padding_mask = (src_mask - 1.0) * self.INF
        encoder_padding_mask = unsqueeze(encoder_padding_mask, [1])
        # 扩展tensor的bacth维度
        encoder_out = nn.BeamSearchDecoder.tile_beam_merge_with_batch(encoder_output, self.beam_size)
        encoder_padding_mask = nn.BeamSearchDecoder.tile_beam_merge_with_batch(encoder_padding_mask,
                                                                                      self.beam_size)
        seq_output, _ = nn.dynamic_decode(decoder=self.beam_search_decoder,
                                                 inits=decoder_initial_states,
                                                 max_step_num=self.max_out_len,
                                                 encoder_out=encoder_output,
                                                 encoder_padding_mask=encoder_padding_mask)
        return seq_output

def text2id(insrc, wi, sentenceMaxlen, start):#将输入的文本通过id 词 映射字典转换成id 向量化
    result = [start, ]
    for ch in insrc:
        result.append(wi[ch])
    result.append(eos_id)
    result_len = len(result)
    if len(result) < sentenceMaxlen:
        result += [eos_id] * (sentenceMaxlen - len(result))
    return unsqueeze(to_tensor(result), axis=0), to_tensor(result_len)
def get_second(inputs, iw):#通过训练好的模型对输入进行预测转换。
    finished_seq = infer_model.predict_batch(inputs=list(inputs))[0][0]
    # finished_seq=finished_seq[0][0]
    input_re = inputs[0][0][1:]
    input_re = tolist(input_re)
    in_input = []
    for subre in input_re:
        if subre == eos_id:
            break
        in_input.append(subre)
    result = []
    for subseq in finished_seq:
        resultid = Counter(list(subseq)).most_common(1)[0][0]
        if resultid == eos_id:
            break
        result.append(resultid)
    word_list_f = [iw[id] for id in in_input]
    word_list_s = [iw[id] for id in result]
    sequence = '上联：' + "".join(word_list_f) + '\t \n下联: ' + "".join(word_list_s) + "\n"
    return sequence
    #input方法，用于获取上联输入框内容
def input_up():
    var = up.get('0.0', 'end')
    var = var.strip()

    tensor = text2id(var, wi,sentenceMaxlen,bos_id,eos_id)

    second = get_second(tensor, iw)
    down.delete('1.0', 'end')
    down.insert('end', second)
def clear_text():#用于清除文本框内容
    up.delete('1.0', 'end')
if __name__ == "__main__":
    data_up_path = "data/fixed_couplets_in.txt"
    data_down_path = "data/fixed_couplets_out.txt"
    # 读取文件
    data_up = readfile(data_up_path)#上联
    data_down = readfile(data_down_path)#下联
    handled_dataup = cleanSentence(data_up)#处理过的上联
    handled_datadown = cleanSentence(data_down)#处理过的下联
    # 将处理好的数据放入data_up/data_down
    global sentenceMaxlen
    sentenceMaxlen = max([len(i) for i in handled_dataup])
    word_frequency = 0#低频词阈值，这里设为0，全都不设低频词
    wi, iw = createDict(handled_dataup, handled_datadown, word_frequency)
    word_size = len(wi)
    id_size = len(iw)
    beam_size = 10#用于beamsearch
    global bos_id
    bos_id = wi['<start>']#起始标记，用于beamsearch
    print(bos_id)
    global eos_id
    eos_id = wi['<end>']#结束标记
    max_out_len = sentenceMaxlen
    #加载模型
    infer_model = Model(Seq2SeqInfer(word_size, 256, 128, 2, bos_id, eos_id, beam_size, max_out_len))
    infer_model.prepare()
    infer_model.load('./model/mod')

    window = Tk()#初始化一个窗口
    window.title('Welcome to our group progect')# 窗口标题
    window.geometry('900x600')# 设定窗口的大小
    Label(window, text='《对联生成器》', bg="pink", font=('Arial', 56), relief="flat").pack()
    canvas = Canvas(window, width=900, height=600, bg='pink')#背景
    canvas.pack(side='top')
    # 上下联输入框提示
    Label(window, text='上联:', font=('Arial', 20)).place(x=10, y=170)
    Label(window, text='下联:', font=('Arial',20)).place(x=10, y=350)
    # 上联输入框，下联输出
    up = Text(window, width=70, height=8, font=('楷体', 15), show=None)  # 显示成明文形式
    up.place(x=100, y=170)
    down = Text(window, width=70, height=8, font=('楷体', 15), show=None)  # 显示成明文形式
    down.place(x=100, y=350)
    # 输入按钮
    submit = Button(window, text='input', width=40,
               height=2,command=input_up)
    submit.place(x=100, y=520)
    #清空按钮
    clear = Button(window, text='clear', width=40,
               height=2,command=clear_text)
    clear.place(x=515, y=520)
    # 第10步，主窗口循环显示
    window.mainloop()