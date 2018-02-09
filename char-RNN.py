get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.io import *
from fastai.conv_learner import *

from fastai.column_data import *

PATH = "data/nietzche"

get_data("https://s3.amazonaws.com/text-datasets/nietzsche.txt", f'{PATH}nietzsche.txt')
text = open(f'{PATH}nietzsche.txt').read()
print('corpus length:', len(text))

text[:400]

chars = sorted(list(set(text)))
vocab_size = len(chars)+1
print("vocab size: " ,vocab_size)

chars.insert(0,"\0")
"".join(chars[1:-6])


char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

idx = [char_indices[c] for c in text]


idx[:10]


''.join(indices_char[i] for i in idx[:100])


#  Three char model


cs = 3
c1_dat = [idx[i] for i in range(0,len(idx)-1-cs, cs)]
c2_dat = [idx[i+1] for i in range(0,len(idx)-1-cs,cs)]
c3_dat = [idx[i+2] for i in range(0, len(idx)-1-cs, cs)]
c4_dat = [idx[i+3] for i in range(0, len(idx)-1-cs, cs)]


x1 = np.stack(c1_dat[:-2])
x2 = np.stack(c2_dat[:-2])
x3 = np.stack(c3_dat[:-2])

y = np.stack(c4_dat[:-2])


x1[:4], x2[:4], x3[:4]


y[:4]


x1.shape, y.shape


# Model

n_hidden = 256


n_fac = 42   #number of latent factors (i.e size of embedding matrix)


class Char3Model(nn.Module):
    def __init__(self, n_fac, vocab_size):
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.l_in = nn.Linear(n_fac, n_hidden)
        self.l_hidden = nn.Linear(n_hidden, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
    
    def forward(self, c1,c2,c3):
        in1 = F.relu(self.l_in(self.e(c1)))
        in2 = F.relu(self.l_in(self.e(c2)))
        in3 = F.relu(self.l_in(self.e(c3)))
        
        h = V(torch.zeros(in1.size()))
        h = F.tanh(self.l_hidden(h+in1))
        h = F.tanh(self.l_hidden(h+in2))
        h = F.tanh(self.l_hidden(h+in3))
        
        return F.log_softmax(self.l_out(h))
    

md = ColumnarModelData.from_arrays('-', [-1], np.stack([x1,x2,x3], axis=1),y,bs=512)


m = Char3Model(n_fac, vocab_size)


it = iter(md.trn_dl)
*xs, yt = next(it)
t = m(*V(xs))


opt = optim.Adam(m.parameters(), 1e-2)


fit(m,md, 1, opt, F.nll_loss)


set_lrs(opt,0.001)


fit(m,md,1,opt,F.nll_loss)


# Test model

def get_next(inp):
    idxs = T(np.array([char_indices[c] for c in inp]))
    p = m(*VV(idxs))
    i = np.argmax(to_np(p))
    return chars[i]

get_next('ppl')


get_next(' th')


# RNN


cs = 8  #Size of unrolled RNN

c_in_dat = [[idx[i+j] for i in range(cs)] for j in range(len(idx)-cs-1)]

c_out_dat = [idx[j+cs] for j in range(len(idx)-cs-1)]

xs = np.stack(c_in_dat, axis=0)


xs.shape

y = np.stack(c_out_dat,axis=0)



y.shape


xs[:cs,:cs]


y[:cs]


#Creating the model



val_idx = get_cv_idxs(len(idx)-cs-1)



md = ColumnarModelData.from_arrays('.',val_idx,xs,y,bs=512)


class CharLoopModel(nn.Module):
    def __init__(self,vocab_size,n_fac):
        super().__init__()
        self.e = nn.Embedding(vocab_size,n_fac)
        self.l_in = nn.Linear(n_fac,n_hidden)
        self.l_hidden = nn.Linear(n_hidden,n_hidden)
        self.l_out = nn.Linear(n_hidden,vocab_size)
        
    def forward(self, *cs):
        bs = cs[0].size(0)
        h = V(torch.zeros(bs,n_hidden))
        for c in cs:
            inp = F.relu(self.l_in(self.e(c)))
            h = F.tanh(self.l_hidden(h+inp))
        
        return F.log_softmax(self.l_out(h)) 



m = CharLoopModel(vocab_size,n_fac)
opt = optim.Adam(m.parameters(),1e-2)



fit(m,md,1,opt,F.nll_loss)



set_lrs(opt,0.001)


fit(m,md,1,opt,F.nll_loss)



class CharConcatModel(nn.Module):
    def __init__(self,vocab_size,n_fac):
        super().__init__()
        self.e = nn.Embedding(vocab_size,n_fac)
        self.l_in = nn.Linear(n_fac+n_hidden,n_hidden)
        self.l_hidden = nn.Linear(n_hidden,n_hidden)
        self.l_out = nn.Linear(n_hidden,vocab_size)
        
    def forward(self,*cs):
        bs = cs[0].size(0)
        h = V(torch.zeros(bs,n_hidden))
        for c in cs:
            inp = torch.cat((h,self.e(c)),1)
            inp = F.relu(self.l_in(inp))
            h = F.tanh(self.l_hidden(inp))
        
        return F.log_softmax(self.l_out(h))



m = CharConcatModel(vocab_size,n_fac)
opt = optim.Adam(m.parameters(),1e-3)



it = iter(md.trn_dl)
*xs,yt = next(it)
t = m(*V(xs))



fit(m, md, 1, opt, F.nll_loss)



set_lrs(opt,1e-4)


fit(m,md,1,opt,F.nll_loss)


#  Testing model

def get_next(inp):
    idxs = T(np.array([char_indices[c] for c in inp]))
    p = m(*VV(idxs))
    i = np.argmax(to_np(p))
    return chars[i]



get_next('for thos')



get_next('part of ')

