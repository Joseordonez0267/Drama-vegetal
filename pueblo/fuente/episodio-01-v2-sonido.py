import numpy as np, json, wave
SR=48000; TL=json.load(open('p2/tl.json')); T=TL['total']; N=int(SR*T); ev=TL['ev']; C=TL['clips']
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
# ---- timeline helpers ----
CL={x['name']:x for x in C}; cs=lambda n:CL[n]['start']; ce=lambda n:CL[n]['end']
VO={v['k']:v for v in TL['voices']}
t=tt(T); kn=ev['knocks']
# ---- drone / horror music ----
amp=np.interp(t,[0,3,cs('road'),cs('phone'),cs('arrive'),cs('door'),cs('warn'),kn[0]-.3,kn[0]-.1,ev['fade']-.2,ev['card'],ev['card']+3,T],[0,.5,.55,.6,.65,.5,.6,.95,0,0,.6,.5,0])
dr=np.zeros(N)
for f,a in [(36.7,1),(55,.6),(73.4,.45),(87.3,.3),(110,.18),(146.8,.08)]:
    for det in (-.3,.3): dr+=a*np.sin(2*np.pi*(f+det)*t+rng.uniform(0,6))
dr*=(0.7+0.3*np.sin(2*np.pi*.07*t)); mus+=dr/4*amp
for a,b,f in [(cs('street'),ce('street'),587.3),(cs('arrive'),ce('arrive'),554.4),(cs('warn'),kn[0]-.1,622.3)]:
    seg=tt(b-a); e=np.minimum(1,np.minimum(seg,(b-a)-seg)/1.5); add(mus,a,np.sin(2*np.pi*f*seg)*(1+.004*np.sin(2*np.pi*5*seg))*e*.035)
def piano(t0,f,v=.25):
    s_=tt(4); x=sum(np.sin(2*np.pi*f*k*s_)*(.6**k) for k in range(1,6))*np.exp(-s_*1.1)*np.minimum(s_/.005,1); add(mus,t0,x*v)
for o,f in [(0,220),(1.3,329.6),(2.6,261.6),(3.9,246.9),(5.2,220),(6.5,196),(7.8,220),(9.4,174.6),(10.8,164.8),(12.4,196),(13.8,220)]: piano(ev['open']-.3+o,f,.17)
for o,f in [(0,146.8),(1.6,174.6),(3.2,130.8)]: piano(ev['card']+.5+o,f,.22)
# ---- wind ----
w=lp(noise(T),500); w/=np.abs(w).max(); gust=.55+.45*np.sin(2*np.pi*.11*t)*np.sin(2*np.pi*.037*t+1)
wa=np.interp(t,[0,cs('street'),cs('road'),cs('drive'),cs('phone'),cs('arrive'),cs('door'),ev['open']+1,cs('warn'),ev['card'],T],[.5,.45,.35,.12,.1,.35,.4,.2,.06,.25,.2])
fx+=w*gust*wa*.5
def owl(t0):
    for o in (0,.5,.75):
        s_=tt(.35); f=420-60*s_/.35; add(fx,t0+o,np.sin(2*np.pi*np.cumsum(f)/SR)*np.sin(np.pi*s_/.35)**2*.12)
owl(4.0); owl(7.6); owl(cs('door')+1.0)
def tick(t0,v=.2,f=2800):
    s_=tt(.04); add(fx,t0,bp(noise(.04),f*.6,f*1.4)*np.exp(-s_*160)*v*4)
for k in range(int(ce('street')-cs('street'))): tick(cs('street')+.3+k,.16,3200 if k%2 else 2400)
x0=cs('warn'); k=0
while x0<kn[0]-.4: tick(x0,.12,3200 if k%2 else 2400); x0+=1; k+=1
s_=tt(.3); add(fx,ev['flip'],(bp(noise(.3),200,1500)*np.exp(-s_*25)*1.2+np.sin(2*np.pi*140*s_)*np.exp(-s_*20)*.6))
# hurried footsteps + door slam of the villager (street clip)
for i in range(14): s_=tt(.08); add(fx,cs('street')+2.5+i*.23,lp(noise(.08),700)*np.exp(-s_*40)*.5)
s_=tt(.5); add(fx,cs('street')+5.8,(lp(noise(.5),500)*np.exp(-s_*12)*1.2+np.sin(2*np.pi*80*s_)*np.exp(-s_*10)*.6))
def engine(a,b,stop_at=None,lvl=.5):
    s_=tt(b-a); rpm=np.ones_like(s_)
    if stop_at is not None: rpm=np.interp(s_,[0,stop_at*.5,stop_at,stop_at+.6,b-a],[1,.75,.45,0,0])
    ph=np.cumsum(42*np.maximum(rpm,.0)*(1+.03*np.sin(2*np.pi*6*s_)))/SR; x=(2*(ph%1)-1)
    x=lp(x,260)+lp(rng.uniform(-1,1,len(s_)),400)*.4; e=np.minimum(1,np.minimum(s_,(b-a)-s_)/.6)*np.minimum(1,rpm*1.5)
    add(fx,a,x*e*lvl)
engine(cs('road'),ce('drive'),lvl=.5)
n4=VO['n4']; engine(n4['start']+(n4['end']-n4['start'])*.55,ce('phone'),lvl=.55)
engine(cs('arrive'),cs('arrive')+7.2,stop_at=6.4,lvl=.5)
# phone vibration
for a in (cs('phone')+.05,cs('phone')+1.45):
    s_=tt(.8); add(fx,a,np.sign(np.sin(2*np.pi*170*s_))*(np.sin(2*np.pi*12*s_)>0)*.06)
def creak(t0,d,f0,f1,v):
    s_=tt(d); f=np.interp(s_,[0,d],[f0,f1]); ph=np.cumsum(f)/SR; pulses=(np.sin(2*np.pi*ph)>.97).astype(float)
    add(fx,t0,bp(pulses+noise(d)*.05,300,2500)*np.sin(np.pi*s_/d)*v)
creak(cs('arrive')+8.0,1.0,30,18,1.0)
# Daniel's footsteps to the door, soft knocks, door creaks open
for i in range(7): s_=tt(.1); add(fx,cs('door')+.2+i*.4,lp(noise(.1),900)*np.exp(-s_*35)*.35)
for kt in ev['knockSoft']:
    s_=tt(.3); add(dry,kt,np.sin(2*np.pi*(120*s_-30*s_**2))*np.exp(-s_*22)*.45+bp(noise(.3),250,1000)*np.exp(-s_*40)*1.0)
creak(ev['open'],1.5,40,22,2.0)
# heartbeat
tb=cs('warn')+2.0; bpm=62
while tb<kn[0]-.3:
    for o,v in ((0,.5),(.22,.35)):
        s_=tt(.25); add(fx,tb+o,np.sin(2*np.pi*55*s_)*np.exp(-s_*18)*v)
    bpm=min(112,bpm+3); tb+=60/bpm
a=kn[0]-2.2; s_=tt(2.1); add(fx,a,bp(noise(2.1),1500,7000)*(s_/2.1)**3*.25)
for kt in kn:
    s_=tt(.6); add(dry,kt,np.sin(2*np.pi*(95*s_-40*s_**2))*np.exp(-s_*14)*1.0+bp(noise(.6),180,900)*np.exp(-s_*30)*2.2)
s_=tt(4); add(dry,ev['fade']-.15,np.sin(2*np.pi*(70*s_-12*s_**2))*np.exp(-s_*1.2)*1.0+lp(noise(4),200)*np.exp(-s_*2)*1.5)
# ---- simple reverb on fx+dry (FFT convolution with decaying noise IR) ----
ir_t=tt(2.2); ir=rng.uniform(-1,1,len(ir_t))*np.exp(-ir_t*3.0); ir[0]=0
def reverb(x,mix):
    n=len(x)+len(ir); X=np.fft.rfft(x,n); I=np.fft.rfft(ir,n); y=np.fft.irfft(X*I,n)[:len(x)]; y/=np.abs(y).max()+1e-9; return x+y*mix*np.abs(x).max()
fxall=reverb(fx+dry,.35)
mus=reverb(mus,.25)
for name,buf in (('p2/music.wav',mus),('p2/fx.wav',fxall)):
    b=buf/ (np.abs(buf).max()+1e-9)*.9; st=(np.stack([b,b],1)*32767).astype(np.int16)
    wv=wave.open(name,'wb'); wv.setnchannels(2); wv.setsampwidth(2); wv.setframerate(SR); wv.writeframes(st.tobytes()); wv.close()
print('ok')
