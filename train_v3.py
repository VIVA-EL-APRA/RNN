"""
Entrenamiento Optimizado V3 - Seq2Seq Simple pero Efectivo
Taller: Traductor Automatico RNN bajo CRISP-ML(Q)
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

print("=" * 60)
print("ENTRENAMIENTO OPTIMIZADO - SEQ2SEQ")
print("=" * 60)

start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[INFO] Dispositivo: {device}")

CORPUS = [
    ("hello", "hola"), ("goodbye", "adios"), ("good morning", "buenos dias"),
    ("good night", "buenas noches"), ("see you later", "hasta luego"),
    ("thank you", "gracias"), ("thank you very much", "muchas gracias"),
    ("please", "por favor"), ("you are welcome", "de nada"),
    ("excuse me", "disculpe"), ("sorry", "lo siento"),
    ("yes", "si"), ("no", "no"), ("maybe", "quizas"),
    ("of course", "por supuesto"),
    ("i", "yo"), ("you", "tu"), ("he", "el"), ("she", "ella"),
    ("we", "nosotros"), ("they", "ellos"),
    ("i am a student", "soy estudiante"), ("you are a teacher", "tu eres maestro"),
    ("he is a professor", "el es profesor"), ("she is a student", "ella es estudiante"),
    ("we are friends", "somos amigos"), ("what is your name", "cual es tu nombre"),
    ("my name is john", "me llamo john"), ("nice to meet you", "mucho gusto"),
    ("father", "padre"), ("mother", "madre"), ("brother", "hermano"),
    ("sister", "hermana"), ("son", "hijo"), ("daughter", "hija"),
    ("university", "universidad"), ("class", "clase"), ("professor", "profesor"),
    ("student", "estudiante"), ("exam", "examen"), ("homework", "tarea"),
    ("i study at the university", "estudio en la universidad"),
    ("the class starts at eight", "la clase empieza a las ocho"),
    ("the exam is difficult", "el examen es dificil"),
    ("i need a book", "necesito un libro"),
    ("where is the library", "donde esta la biblioteca"),
    ("the professor is strict", "el profesor es estricto"),
    ("i have a class at nine", "tengo clase a las nueve"),
    ("the lecture is interesting", "la conferencia es interesante"),
    ("when is the exam", "cuando es el examen"),
    ("i passed the exam", "aprobe el examen"),
    ("i need to study", "necesito estudiar"),
    ("i am late for class", "llegue tarde a clase"),
    ("one", "uno"), ("two", "dos"), ("three", "tres"),
    ("four", "cuatro"), ("five", "cinco"), ("six", "seis"),
    ("seven", "siete"), ("eight", "ocho"), ("nine", "nueve"),
    ("ten", "diez"),
    ("monday", "lunes"), ("tuesday", "martes"), ("wednesday", "miercoles"),
    ("thursday", "jueves"), ("friday", "viernes"), ("saturday", "sabado"),
    ("sunday", "domingo"), ("today", "hoy"), ("tomorrow", "manana"),
    ("time", "tiempo"), ("hour", "hora"), ("minute", "minuto"),
    ("now", "ahora"), ("later", "despues"), ("early", "temprano"),
    ("late", "tarde"), ("always", "siempre"), ("never", "nunca"),
    ("here", "aqui"), ("there", "alli"), ("where", "donde"),
    ("city", "ciudad"), ("country", "pais"), ("home", "casa"),
    ("office", "oficina"), ("library", "biblioteca"), ("cafe", "cafe"),
    ("park", "parque"),
    ("to be", "ser"), ("to have", "tener"), ("to do", "hacer"),
    ("to go", "ir"), ("to come", "venir"),
    ("to see", "ver"), ("to know", "saber"), ("to think", "pensar"),
    ("to want", "querer"), ("to need", "necesitar"), ("to like", "gustar"),
    ("to learn", "aprender"), ("to teach", "enseñar"), ("to study", "estudiar"),
    ("to work", "trabajar"), ("to live", "vivir"), ("to eat", "comer"),
    ("to drink", "beber"), ("to speak", "hablar"), ("to write", "escribir"),
    ("to read", "leer"), ("to understand", "entender"),
    ("to help", "ayudar"), ("to start", "empezar"), ("to finish", "terminar"),
    ("book", "libro"), ("pen", "lapiz"), ("paper", "papel"),
    ("computer", "computadora"), ("phone", "telefono"), ("table", "mesa"),
    ("chair", "silla"), ("door", "puerta"), ("window", "ventana"),
    ("food", "comida"), ("water", "agua"), ("coffee", "cafe"),
    ("good", "bueno"), ("bad", "malo"), ("big", "grande"),
    ("small", "pequeño"), ("new", "nuevo"), ("old", "viejo"),
    ("fast", "rapido"), ("slow", "lento"), ("easy", "facil"),
    ("difficult", "dificil"), ("important", "importante"),
    ("interesting", "interesante"), ("beautiful", "hermoso"),
    ("happy", "feliz"), ("sad", "triste"),
    ("what", "que"), ("who", "quien"), ("when", "cuando"),
    ("why", "por que"), ("how", "como"),
    ("how are you", "como estas"), ("how much", "cuanto"),
    ("what time is it", "que hora es"),
    ("science", "ciencia"), ("math", "matematicas"), ("history", "historia"),
    ("art", "arte"), ("music", "musica"), ("language", "idioma"),
    ("english", "ingles"), ("spanish", "espanol"),
    ("computer science", "ciencias de la computacion"),
    ("information", "informacion"), ("technology", "tecnologia"),
]

for esp, ing in list(CORPUS):
    if (ing, esp) not in CORPUS:
        CORPUS.append((ing, esp))

print(f"[INFO] Corpus: {len(CORPUS)} parejas")

PAD = "<PAD>"
UNK = "<UNK>"
SOS = "<SOS>"
EOS = "<EOS>"

class Vocab:
    def __init__(self):
        self.w2i = {PAD: 0, UNK: 1, SOS: 2, EOS: 3}
        self.i2w = {0: PAD, 1: UNK, 2: SOS, 3: EOS}
        self.n = 4
    
    def add(self, text):
        for w in text.lower().split():
            if w not in self.w2i:
                self.w2i[w] = self.n
                self.i2w[self.n] = w
                self.n += 1
    
    def enc(self, text, max_len, sos=False, eos=False):
        ids = []
        if sos:
            ids.append(self.w2i[SOS])
        for w in text.lower().split():
            ids.append(self.w2i.get(w, self.w2i[UNK]))
        if eos:
            ids.append(self.w2i[EOS])
        while len(ids) < max_len:
            ids.append(self.w2i[PAD])
        return ids[:max_len]
    
    def dec(self, ids):
        ws = []
        for i in ids:
            if torch.is_tensor(i):
                i = i.item()
            w = self.i2w.get(i, UNK)
            if w not in [PAD, SOS, EOS]:
                ws.append(w)
        return ' '.join(ws)

src_v = Vocab()
tgt_v = Vocab()

for s, t in CORPUS:
    src_v.add(s)
    tgt_v.add(t)

print(f"[OK] Vocab src: {src_v.n}, tgt: {tgt_v.n}")

MAX_LEN = 20
BATCH = 32
EMBED = 256
HIDDEN = 512
LAYERS = 2
DROP = 0.3
EPOCHS = 100
LR = 0.001

print(f"\n[INFO] Embed={EMBED}, Hidden={HIDDEN}, Layers={LAYERS}, Epochs={EPOCHS}")

class Encoder(nn.Module):
    def __init__(self, vs, em, hd, ly, dp):
        super().__init__()
        self.emb = nn.Embedding(vs, em, padding_idx=0)
        self.lstm = nn.LSTM(em, hd, ly, batch_first=True, dropout=dp)
        self.dp = nn.Dropout(dp)
    
    def forward(self, x):
        e = self.dp(self.emb(x))
        o, (h, c) = self.lstm(e)
        return o, h, c

class Decoder(nn.Module):
    def __init__(self, vs, em, hd, ly, dp):
        super().__init__()
        self.emb = nn.Embedding(vs, em, padding_idx=0)
        self.lstm = nn.LSTM(em, hd, ly, batch_first=True, dropout=dp)
        self.fc = nn.Linear(hd, vs)
        self.dp = nn.Dropout(dp)
    
    def forward(self, x, h, c):
        e = self.dp(self.emb(x))
        o, (h, c) = self.lstm(e, (h, c))
        return self.fc(o.squeeze(1)), h, c

class Seq2Seq(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec
    
    def forward(self, src, tgt, tf=0.5):
        bs = src.shape[0]
        max_len = tgt.shape[1]
        out = torch.zeros(bs, max_len, self.dec.fc.out_features).to(device)
        
        _, h, c = self.enc(src)
        
        dec_in = tgt[:, 0]
        
        for t in range(1, max_len):
            o, h, c = self.dec(dec_in.unsqueeze(1), h, c)
            out[:, t] = o
            
            tf_now = np.random.random() < tf
            top1 = o.argmax(1)
            dec_in = tgt[:, t] if tf_now else top1
        
        return out

enc = Encoder(src_v.n, EMBED, HIDDEN, LAYERS, DROP)
dec = Decoder(tgt_v.n, EMBED, HIDDEN, LAYERS, DROP)
model = Seq2Seq(enc, dec).to(device)

params = sum(p.numel() for p in model.parameters())
print(f"[OK] Parametros: {params:,}")

class DS(Dataset):
    def __init__(self, data, sv, tv, ml):
        self.d = [(sv.enc(s, ml), tv.enc(t, ml, True, True)) for s, t in data]
    
    def __len__(self):
        return len(self.d)
    
    def __getitem__(self, i):
        return torch.tensor(self.d[i][0]), torch.tensor(self.d[i][1])

ds = DS(CORPUS, src_v, tgt_v, MAX_LEN)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

crit = nn.CrossEntropyLoss(ignore_index=0)
opt = optim.Adam(model.parameters(), lr=LR)
sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)

print(f"\n[INFO] Entrenando {EPOCHS} epocas...")

model.train()
losses = []
best_loss = float('inf')

for ep in range(1, EPOCHS + 1):
    ep_loss = 0
    n = 0
    
    for src, tgt in dl:
        src, tgt = src.to(device), tgt.to(device)
        opt.zero_grad()
        
        tf = max(0.3, 0.5 * (1 - ep / EPOCHS))
        out = model(src, tgt, tf)
        
        out = out.view(-1, out.shape[-1])
        tgt_flat = tgt.view(-1)
        
        loss = crit(out, tgt_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        ep_loss += loss.item()
        n += 1
    
    avg = ep_loss / n
    losses.append(avg)
    sch.step(avg)
    
    if avg < best_loss:
        best_loss = avg
        torch.save({
            'm': model.state_dict(),
            'src_vocab': src_v.w2i,
            'tgt_vocab': tgt_v.w2i,
            'src_idx2word': src_v.i2w,
            'tgt_idx2word': tgt_v.i2w,
        }, 'best.pt')
    
    if ep % 10 == 0 or ep == EPOCHS:
        print(f"  Ep {ep:3d}/{EPOCHS} - Loss: {avg:.4f}")

def bleu(ref, hyp):
    rw = ref.lower().split()
    hw = hyp.lower().split()
    if not hw:
        return 0.0
    m = sum(1 for w in hw if w in rw)
    p = m / len(hw) if hw else 0
    bp = min(1.0, np.exp(1 - len(rw) / max(len(hw), 1)))
    return bp * p

ckpt = torch.load('best.pt')
model.load_state_dict(ckpt['m'])
model.eval()

src_v.w2i = ckpt['src_vocab']
tgt_v.w2i = ckpt['tgt_vocab']
src_v.i2w = ckpt['src_idx2word']
tgt_v.i2w = ckpt['tgt_idx2word']

tests = [
    ("hello", "hola"), ("goodbye", "adios"), ("thank you", "gracias"),
    ("i am a student", "soy estudiante"), ("where is the library", "donde esta la biblioteca"),
    ("the exam is difficult", "el examen es dificil"), ("i need to study", "necesito estudiar"),
    ("good morning", "buenos dias"), ("how are you", "como estas"),
    ("i study at the university", "estudio en la universidad"),
]

print("\nResultados:")
print("-" * 60)

total = 0
with torch.no_grad():
    for st, tt in tests:
        enc_in = torch.tensor([src_v.enc(st, MAX_LEN)]).to(device)
        
        _, h, c = enc(enc_in)
        
        dec_in = torch.tensor([tgt_v.w2i[SOS]]).to(device)
        res = []
        
        for _ in range(MAX_LEN):
            o, h, c = dec(dec_in.unsqueeze(1), h, c)
            top = o.argmax(1).item()
            
            if top == tgt_v.w2i[EOS] or top == tgt_v.w2i[PAD]:
                break
            
            res.append(top)
            dec_in = torch.tensor([top]).to(device)
        
        trans = tgt_v.dec(res)
        b = bleu(tt, trans)
        total += b
        print(f"{st:<30} -> {tt:<25} BLEU: {b:.2f}")

avg_bleu = total / len(tests)
print("-" * 60)
print(f"\nBLEU Score: {avg_bleu:.2f}")

torch.save({
    'm': model.state_dict(),
    'src_vocab': src_v.w2i,
    'tgt_vocab': tgt_v.w2i,
    'src_idx2word': src_v.i2w,
    'tgt_idx2word': tgt_v.i2w,
    'ls': losses,
    'bl': avg_bleu,
}, 'translator.pt')

elapsed = time.time() - start_time

print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)
print(f"[OK] Tiempo: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"[OK] Epocas: {EPOCHS}")
print(f"[OK] Parametros: {params:,}")
print(f"[OK] BLEU: {avg_bleu:.2f}")
print(f"[OK] Loss: {losses[-1]:.4f}")
print("=" * 60)
print("ENTRENAMIENTO COMPLETADO")