import numpy as np, json, wave
SR=48000; TL=json.load(open('p1/tl.json')); T=TL['total']; N=int(SR*T); ev=TL['ev']; C=TL['clips']
rng=np.random.default_rng(9)
mus=np.zeros(N); fx=np.zeros(N); dry=np.zeros(N)
def add(buf,t,sig,g=1.0):
    i=int(t*SR); 
    if i>=N: return
    if i<0: sig=sig[-i:]; i=0
    j=min(N,i+len(sig)); buf[i:j]+=sig[:j-i]*g
def tt(d): return np.arange(int(d*SR))/SR
def lp(x,fc):  # one-pole lowpass, vectorized via FFT filter
    X=np.fft.rfft(x); f=np.fft.rfftfreq(len(x),1/SR); return np.fft.irfft(X/(1+1j*f/fc),len(x))
def bp(x,lo,hi):
    X=np.fft.rfft(x); f=np.fft.rfftfreq(len(x),1/SR); m=((f>lo)&(f<hi)).astype(float); 
    k=np.ones(64)/64; m=np.convolve(m,k,'same'); return np.fft.irfft(X*m,len(x))
def env(n,a,d): x=np.arange(n)/SR; return np.minimum(x/max(a,1e-4),1)*np.exp(-x/d)
noise=lambda d: rng.uniform(-1,1,int(d*SR))
c=lambda i:C[i]['start']
# ---- drone / horror music ----
t=tt(T)
amp=np.interp(t,[0,3,9.5,27,46,57,63.5,76,78.9,79.1,85.4,86.6,90,T],[0,.5,.55,.6,.7,.55,.5,.75,.95,0,0,.6,.5,0])
dr=np.zeros(N)
for f,a in [(36.7,1),(55,.6),(73.4,.45),(87.3,.3),(110,.18),(146.8,.08)]:
    for det in (-.3,.3): dr+=a*np.sin(2*np.pi*(f+det)*t+rng.uniform(0,6))
dr*= (0.7+0.3*np.sin(2*np.pi*.07*t))
mus+=dr/4*amp
# high eerie tone in some clips
for a,b,f in [(c(1),c(2),587.3),(c(5),c(6),554.4),(c(7),ev['knocks'][0]-.1,622.3)]:
    seg=tt(b-a); e=np.minimum(1,np.minimum(seg,(b-a)-seg)/1.5); add(mus,a,np.sin(2*np.pi*f*seg)*(1+.004*np.sin(2*np.pi*5*seg))*e*.035)
# piano (clip 7 & 8): sparse minor notes
def piano(t0,f,v=.25):
    s=tt(4); x=sum(np.sin(2*np.pi*f*k*s)*(.6**k) for k in range(1,6))*np.exp(-s*1.1)*np.minimum(s/.005,1); add(mus,t0,x*v)
notes=[(0,220),(1.3,329.6),(2.6,261.6),(3.9,246.9),(5.2,220),(6.5,196),(7.8,220),(9.6,174.6),(11.0,164.8)]
for o,f in notes: piano(c(6)+.8+o,f,.18)
# end card theme
for o,f in [(0,146.8),(1.6,174.6),(3.2,130.8)]: piano(ev['card']+.5+o,f,.22)
# ---- wind ----
w=lp(noise(T),500); w/=np.abs(w).max()
gust=.55+.45*np.sin(2*np.pi*.11*t)*np.sin(2*np.pi*.037*t+1)
wa=np.interp(t,[0,9.5,18.5,27.5,37,46,57,58,79,85.6,86.6,T],[.5,.45,.25,.15,.1,.4,.35,.08,.06,.06,.25,.2])
fx+=w*gust*wa*.5
# ---- owl ----
def owl(t0):
    for o in (0,.5,.75):
        s=tt(.35); f=420-60*s/.35; add(fx,t0+o,np.sin(2*np.pi*np.cumsum(f)/SR)*np.sin(np.pi*s/.35)**2*.12)
owl(c(0)+4.2); owl(c(0)+8.0)
# ---- clock ticks ----
def tick(t0,v=.2,f=2800):
    s=tt(.04); add(fx,t0,bp(noise(.04),f*.6,f*1.4)*np.exp(-s*160)*v*4)
for k in range(9): tick(c(1)+.3+k,.16,3200 if k%2 else 2400)
k=0; tt0=c(7)+3.6
while tt0<ev['knocks'][0]-.4: tick(tt0,.13,3200 if k%2 else 2400); tt0+=1; k+=1
s=tt(.3); add(fx,ev['flip'],(bp(noise(.3),200,1500)*np.exp(-s*25)*1.2+np.sin(2*np.pi*140*s)*np.exp(-s*20)*.6))  # clunk
# ---- engine ----
def engine(a,b,stop=False):
    s=tt(b-a); rpm=np.where(stop,np.interp(s,[0,(b-a)*.6,b-a],[1,.6,0]),1) if stop else np.ones_like(s)
    ph=np.cumsum(42*rpm*(1+.03*np.sin(2*np.pi*6*s)))/SR; x=(2*(ph%1)-1)
    x=lp(x,260)+lp(rng.uniform(-1,1,len(s)),400)*.4; e=np.minimum(1,np.minimum(s,(b-a)-s)/.8)*(rpm if stop else 1)
    add(fx,a,x*e*.5)
engine(c(2),c(4)-.5); engine(c(5),c(5)+5.2,stop=True)
# ---- phone vibration ----
for a in (c(4)+.05,c(4)+1.5):
    s=tt(.8); add(fx,a,np.sign(np.sin(2*np.pi*170*s))*(np.sin(2*np.pi*12*s)>0)*.05)
# ---- wood creaks (clip 6) and door creak (clip 7) ----
def creak(t0,d,f0,f1,v):
    s=tt(d); f=np.interp(s,[0,d],[f0,f1]); ph=np.cumsum(f)/SR; pulses=(np.sin(2*np.pi*ph)>.97).astype(float)
    x=bp(pulses+noise(d)*.05,300,2500)*np.sin(np.pi*s/d)*v; add(fx,t0,x)
creak(c(5)+6.5,1.2,30,18,1.2); creak(c(5)+9.2,.8,25,35,.8); creak(c(6)+.4,1.6,40,22,2.0)
# ---- heartbeat ----
tb=c(7)+2.5; bpm=60
while tb<ev['knocks'][0]-.3:
    for o,v in ((0,.5),(.22,.35)):
        s=tt(.25); add(fx,tb+o,np.sin(2*np.pi*55*s)*np.exp(-s*18)*v)
    bpm=min(110,bpm+4); tb+=60/bpm
# ---- riser into knocks, then silence ----
a=ev['knocks'][0]-2.2; s=tt(2.1); add(fx,a,bp(noise(2.1),1500,7000)*(s/2.1)**3*.25)
# ---- knocks ----
for kt in ev['knocks']:
    s=tt(.6); th=np.sin(2*np.pi*(95*s-40*s**2))*np.exp(-s*14)*1.0
    wood=bp(noise(.6),180,900)*np.exp(-s*30)*2.2
    add(dry,kt,th+wood)
# ---- final impact ----
s=tt(4); add(dry,ev['fade']-.15,np.sin(2*np.pi*(70*s-12*s**2))*np.exp(-s*1.2)*1.0+lp(noise(4),200)*np.exp(-s*2)*1.5)
# ---- simple reverb on fx+dry (FFT convolution with decaying noise IR) ----
ir_t=tt(2.2); ir=rng.uniform(-1,1,len(ir_t))*np.exp(-ir_t*3.0); ir[0]=0
def reverb(x,mix):
    n=len(x)+len(ir); X=np.fft.rfft(x,n); I=np.fft.rfft(ir,n); y=np.fft.irfft(X*I,n)[:len(x)]; y/=np.abs(y).max()+1e-9; return x+y*mix*np.abs(x).max()
fxall=reverb(fx+dry,.35)
mus=reverb(mus,.25)
for name,buf in (('p1/music.wav',mus),('p1/fx.wav',fxall)):
    b=buf/ (np.abs(buf).max()+1e-9)*.9; st=(np.stack([b,b],1)*32767).astype(np.int16)
    wv=wave.open(name,'wb'); wv.setnchannels(2); wv.setsampwidth(2); wv.setframerate(SR); wv.writeframes(st.tobytes()); wv.close()
print('ok')
