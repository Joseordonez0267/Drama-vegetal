import numpy as np, json, wave
SR=48000; TL=json.load(open('p5/tl.json')); T=TL['total']; N=int(SR*T); ev=TL['ev']; C=TL['clips']
rng=np.random.default_rng(21)
mus=np.zeros(N); fx=np.zeros(N); dry=np.zeros(N)
CL={x['name']:x for x in C}; cs=lambda n:CL[n]['start']; ce=lambda n:CL[n]['end']; VO={v['k']:v for v in TL['voices']}; kn=ev['knocks']
def add(buf,t,sig,g=1.0):
    i=int(t*SR)
    if i>=N: return
    if i<0: sig=sig[-i:]; i=0
    j=min(N,i+len(sig)); buf[i:j]+=sig[:j-i]*g
tt=lambda d: np.arange(int(d*SR))/SR
noise=lambda d: rng.uniform(-1,1,int(d*SR))
def bp(x,lo,hi):
    X=np.fft.rfft(x); f=np.fft.rfftfreq(len(x),1/SR); m=((f>lo)&(f<hi)).astype(float); m=np.convolve(m,np.ones(32)/32,'same'); return np.fft.irfft(X*m,len(x))
def lp(x,fc): X=np.fft.rfft(x); f=np.fft.rfftfreq(len(x),1/SR); return np.fft.irfft(X/(1+(f/fc)**4),len(x))
t=tt(T)
# ---------- strings pad (D minor), no noise ----------
def string(f,d,v):
    s=tt(d); x=np.zeros_like(s)
    for det in (-.0025,0,.0025):
        ph=2*np.pi*f*(1+det)*s+rng.uniform(0,6)+.003*np.sin(2*np.pi*5.2*s)*f/50
        for h in range(1,9): x+=np.sin(h*ph)/(h**1.5)
    e=np.minimum(1,np.minimum(s/1.8,(d-s)/1.8)); return x*e*v/9
chords=[[146.8,174.6,220,293.7],[116.5,174.6,233.1,293.7],[98,146.8,196,233.1],[110,138.6,164.8,220]]
lvl=lambda x: np.interp(x,[0,cs('street'),cs('road'),cs('phone'),cs('arrive'),cs('door'),VO['d2']['start'],cs('warn'),kn[0]-1,kn[0]-.05,ev['figure'],ev['fade'],ev['poll']+.6,ev['card']-.4,ev['card'],T],[.5,.55,.6,.65,.6,.5,.55,.65,.9,0,0,0,.35,.4,.7,.6])
x0=.2; i=0
while x0<T-1:
    L=lvl(x0)
    if L>.02:
        for f in chords[i%4]: add(mus,x0,string(f,6.6,L*.5))
    x0+=6; i+=1
# high tremolo tension in the warning scene
a,b=cs('warn'),kn[0]-.05; s=tt(b-a); add(mus,a,np.sin(2*np.pi*587.3*s)*(.6+.4*np.sin(2*np.pi*11*s))*np.minimum(1,s/4)*(s/(b-a))**1.5*.12)
def piano(t0,f,v=.25):
    s=tt(4); x=sum(np.sin(2*np.pi*f*k*s)*(.55**k) for k in range(1,6))*np.exp(-s*1.2)*np.minimum(s/.004,1); add(mus,t0,x*v)
for o,f in [(0,293.7),(1.2,349.2),(2.4,440),(3.6,392),(5.0,349.2),(6.2,293.7),(7.6,329.6),(9.0,277.2),(10.4,293.7)]: piano(ev['open']+o,f,.16)
for o,f in [(0,293.7),(.9,440),(1.8,349.2),(3.0,329.6),(4.2,293.7)]: piano(ev['card']+.4+o,f,.2)
# figure sting + boom
s=tt(2.5); add(mus,ev['figure']+2.0,sum(np.sin(2*np.pi*f*s) for f in (311.1,329.6,466.2,493.9))*np.exp(-s*1.2)*np.minimum(1,s/.05)*.12)
s=tt(4); add(dry,ev['fade']-.1,np.sin(2*np.pi*(65*s-10*s**2))*np.exp(-s*1.1)*.9)
# ---------- night ambience (exterior only) ----------
ext_ranges=[(0,cs('drive')),(0,ev['cold']),(cs('phone')+ (VO['n4']['start']-cs('phone')) ,cs('warn')),(ev['figure'],ev['fade']),(ev['card'],T)]
def cricket(t0,f,v):
    for k in range(3):
        s=tt(.035); add(fx,t0+k*.055,np.sin(2*np.pi*f*s)*np.hanning(len(s))*v)
for a,b in ext_ranges:
    x0=a+.1
    while x0<b: cricket(x0,4300+rng.uniform(-80,80),.022); x0+=rng.uniform(.55,.8)
    x0=a+.4
    while x0<b: cricket(x0,5200,.014); x0+=rng.uniform(.9,1.3)
def owl(t0):
    for o in (0,.45,.7):
        s=tt(.35); f=410-60*s/.35; add(fx,t0+o,np.sin(2*np.pi*np.cumsum(f)/SR)*np.sin(np.pi*s/.35)**2*.07)
owl(3.5); owl(cs('arrive')+7.5); owl(ev['card']+2)
def gust(t0,d,v):
    s=tt(d); add(fx,t0,bp(noise(d),250,1100)*np.sin(np.pi*s/d)**2*v)
for g0 in (1.0,cs('street')+1,cs('road')+.5,cs('door')+.5,ev['figure']): gust(g0,3.2,.07)
# dog barks (synced with the street shot) + whimper
for bt in (1.2,1.55,2.6,2.95):
    s=tt(.22); f=np.interp(s,[0,.05,.22],[380,620,300]); x=np.sin(2*np.pi*np.cumsum(f)/SR); x=np.sign(x)*np.abs(x)**.6
    add(fx,cs('street')+bt,bp(x+noise(.22)*.3,250,2500)*np.sin(np.pi*s/.22)*.35)
s=tt(.9); add(fx,cs('street')+4.4,np.sin(2*np.pi*np.cumsum(np.interp(s,[0,.9],[900,600]))/SR)*np.sin(np.pi*s/.9)*.05)
for i in range(10): s=tt(.05); add(fx,cs('street')+4.4+i*.15,bp(noise(.05),400,2000)*np.exp(-s*60)*.15)
# villager footsteps + door slam
for i in range(14): s=tt(.08); add(fx,cs('street')+2.4+i*.23,lp(noise(.08),600)*np.exp(-s*40)*.6)
s=tt(.6); add(fx,cs('street')+5.6,lp(noise(.6),400)*np.exp(-s*12)*1.4+np.sin(2*np.pi*75*s)*np.exp(-s*10)*.5)
# clock ticks
def tick(t0,v=.18,f=2800): s=tt(.04); add(fx,t0,bp(noise(.04),f*.6,f*1.4)*np.exp(-s*160)*v*4)
for k in range(9): tick(cs('street')+.3+k,.14,3200 if k%2 else 2400)
x0=cs('warn'); k=0
while x0<kn[0]-.4: tick(x0,.12,3200 if k%2 else 2400); x0+=1; k+=1
s=tt(.3); add(fx,ev['flip'],bp(noise(.3),200,1500)*np.exp(-s*25)*1.2+np.sin(2*np.pi*140*s)*np.exp(-s*20)*.6)
# engine: low rumble (firing pulses), no hiss
def engine(a,b,rate=34,lvl=.05,stop=None):
    s=tt(b-a); r=np.full_like(s,rate)
    if stop: r=np.interp(s,[0,stop*.6,stop,stop+.5,b-a],[rate,rate*.75,rate*.55,0.001,0.001])
    ph=np.cumsum(r)/SR; x=np.zeros_like(s)
    for h in range(1,7): x+=np.sin(2*np.pi*h*ph)/(h**1.2)
    x*=1+.3*np.sin(2*np.pi*ph*.5); x=lp(x,220)
    e=np.minimum(1,np.minimum(s/.6,(b-a-s)/.6))*(np.minimum(1,r/rate*1.5) if stop else 1); add(fx,a,x*e*lvl)
engine(cs('road'),ce('drive')); engine(VO['n4']['start']+(VO['n4']['end']-VO['n4']['start'])*.5,ce('phone'),38)
engine(cs("arrive"),cs("arrive")+7.2,34,.05,stop=6.4)
for a in (cs('phone')+.05,cs('phone')+1.45): s=tt(.8); add(fx,a,np.sign(np.sin(2*np.pi*170*s))*(np.sin(2*np.pi*12*s)>0)*.05)
def creak(t0,d,f0,f1,v):
    s=tt(d); f=np.interp(s,[0,d],[f0,f1]); pulses=(np.sin(2*np.pi*np.cumsum(f)/SR)>.97).astype(float); add(fx,t0,bp(pulses,300,2500)*np.sin(np.pi*s/d)*v)
creak(cs('arrive')+8.0,1.0,30,18,.9)
for i in range(7): s=tt(.1); add(fx,cs('door')+.2+i*.4,lp(noise(.1),800)*np.exp(-s*35)*.45)
for kt in ev['knockSoft']: s=tt(.3); add(dry,kt,np.sin(2*np.pi*(120*s-30*s**2))*np.exp(-s*22)*.45+bp(noise(.3),250,1000)*np.exp(-s*40)*1.0)
creak(ev['open'],1.5,40,22,1.8)
# candle crackle indoors
x0=cs('warn')
while x0<ev['figure']: s=tt(.01); add(fx,x0,bp(noise(.01),1500,6000)*.06); x0+=rng.uniform(.3,1.5)
tb=cs('warn')+2.0; bpm=62
while tb<kn[0]-.3:
    for o,v in ((0,.5),(.22,.35)): s=tt(.25); add(fx,tb+o,np.sin(2*np.pi*55*s)*np.exp(-s*18)*v)
    bpm=min(112,bpm+3); tb+=60/bpm
a=kn[0]-2.2; s=tt(2.1); add(fx,a,np.sin(2*np.pi*np.cumsum(np.interp(s,[0,2.1],[200,900]))/SR)*(s/2.1)**3*.12)
for kt in kn: s=tt(.6); add(dry,kt,np.sin(2*np.pi*(95*s-40*s**2))*np.exp(-s*14)*1.0+bp(noise(.6),180,900)*np.exp(-s*30)*2.2)
# voting segment: slow clock + heartbeat under the narrator
x0=ev['poll']+.5; k=0
while x0<ev['card']-.3: tick(x0,.1,3200 if k%2 else 2400); x0+=1; k+=1
tb=ev['poll']+.8
while tb<ev['card']-.5:
    for o,v in ((0,.35),(.22,.25)): s=tt(.25); add(fx,tb+o,np.sin(2*np.pi*55*s)*np.exp(-s*18)*v)
    tb+=60/72
s=tt(1.2); add(mus,ev['poll']+.1,sum(np.sin(2*np.pi*f*s) for f in (146.8,220,293.7))*np.exp(-s*1.5)*np.minimum(1,s/.02)*.1)
# ---- cold open: knocks, riser, impact into the title ----
for kt in ev['coKnocks']: s=tt(.6); add(dry,kt,np.sin(2*np.pi*(95*s-40*s**2))*np.exp(-s*14)*1.0+bp(noise(.6),180,900)*np.exp(-s*30)*2.2)
s=tt(1.6); add(mus,ev['cold']-1.7,np.sin(2*np.pi*np.cumsum(np.interp(s,[0,1.6],[180,880]))/SR)*(s/1.6)**3*.15)
s=tt(3.5); add(dry,ev['cold']-.05,np.sin(2*np.pi*(62*s-9*s**2))*np.exp(-s*1.3)*.9)
s=tt(2.2); add(mus,ev['cold']-.6+2.0,sum(np.sin(2*np.pi*f*s) for f in (311.1,329.6,466.2))*np.exp(-s*1.4)*np.minimum(1,s/.05)*.0)
# ---- part-2 teaser: handle rattles and turns, sting ----
te=ev['teaser']
for i in range(6): s=tt(.08); add(fx,te+.35+i*.22,bp(noise(.08),900,4000)*np.exp(-s*50)*.5)
s=tt(1.4); f=np.interp(s,[0,1.4],[30,12]); pulses=(np.sin(2*np.pi*np.cumsum(f)/SR)>.96).astype(float); add(fx,te+.4,bp(pulses,400,3000)*np.sin(np.pi*s/1.4)*1.4)
s=tt(3); add(dry,te+1.85,np.sin(2*np.pi*(70*s-12*s**2))*np.exp(-s*1.2)*.8); add(mus,te+1.85,sum(np.sin(2*np.pi*f*s) for f in (293.7,311.1,440,466.2))*np.exp(-s*1.3)*np.minimum(1,s/.03)*.12)
ir_t=tt(2.0); ir=rng.uniform(-1,1,len(ir_t))*np.exp(-ir_t*3.2); ir[0]=0
def reverb(x,mix):
    n=len(x)+len(ir); y=np.fft.irfft(np.fft.rfft(x,n)*np.fft.rfft(ir,n),n)[:len(x)]; y/=np.abs(y).max()+1e-9; return x+y*mix*np.abs(x).max()
fxall=reverb(fx+dry,.3); mus=reverb(mus,.3)
for name,buf in (('p5/music.wav',mus),('p5/fx.wav',fxall)):
    b=buf/(np.abs(buf).max()+1e-9)*.9; st=(np.stack([b,b],1)*32767).astype(np.int16)
    wv=wave.open(name,'wb'); wv.setnchannels(2); wv.setsampwidth(2); wv.setframerate(SR); wv.writeframes(st.tobytes()); wv.close()
print('ok')
