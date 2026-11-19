import numpy as np, json
C=343.0
def ring(n,r,off=0.0): a=np.radians(off)+2*np.pi*np.arange(n)/n; return np.c_[r*np.cos(a),r*np.sin(a),0*a]
G_={ "ReSpeaker Core v2: 6 on 93 mm": ring(6,.0463,90),
     "4 on 64 mm": ring(4,.032),
     "6 on 93 mm + centre (7)": np.r_[ring(6,.0463),[[0,0,0]]],
     "8 on 70 mm": ring(8,.035),
     "8 on 100 mm": ring(8,.050),
     "8 on 140 mm": ring(8,.070),
     "4 on 40 mm + 8 on 120 mm (12)": np.r_[ring(4,.020),ring(8,.060)],
     "6 on 60 mm + 10 on 188 mm (16)": np.r_[ring(6,.030),ring(10,.0941)],
     "Whistler: 16 on 188 mm": ring(16,.0941,90)}
EL=20.0
def u(az,el=EL): az,el=np.radians(az),np.radians(el); return np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
def sv(P,f,az,el=EL): return np.exp(2j*np.pi*f*(P@u(az,el))/C)
F=np.geomspace(300,7000,40); SPOT=[250,500,1000,2000,4000,7000]
LOOK=np.arange(0,360,7.5)
def weights(P,f,az,kind,wng_db):
    d=sv(P,f,az); n=len(P)
    if kind=='ds': return d/n
    D=np.linalg.norm(P[:,None]-P[None],axis=2); G=np.sinc(2*f*D/C)
    for eps in 10.0**np.arange(-5,2.01,0.2):
        Ri=np.linalg.inv(G+eps*np.eye(n)); w=Ri@d/(d.conj()@Ri@d)
        if -10*np.log10(np.real(w.conj()@w))>=wng_db: break
    return w
def di(P,f,w):
    D=np.linalg.norm(P[:,None]-P[None],axis=2); G=np.sinc(2*f*D/C)
    return -10*np.log10(np.real(w.conj()@G@w))
def run(P,kind,wng):
    spot=[np.mean([di(P,f,weights(P,f,a,kind,wng)) for a in LOOK[::6]]) for f in SPOT]
    band=[];rej={60:[],90:[],180:[]};side=[]
    for f in F:
        for a in LOOK[::6]:
            w=weights(P,f,a,kind,wng); band.append(di(P,f,w))
            for k in rej: rej[k].append(abs(w.conj()@sv(P,f,a+k))**2)
            resp=np.array([abs(w.conj()@sv(P,f,a+o))**2 for o in np.arange(45,316,5)]); side.append(resp.max())
    return spot,np.mean(band),{k:10*np.log10(np.mean(v)) for k,v in rej.items()},10*np.log10(np.mean(side))
res={}
for name,P in G_.items():
    for kind,wng in [('ds',0),('sd',0),('sd',-10)]:
        s,b,r,sl=run(P,kind,wng); key=f"{kind}{'' if kind=='ds' else wng}"
        res.setdefault(name,{})[key]=dict(spot=[round(float(x),1) for x in s],band=round(float(b),1),rej={k:round(float(v),1) for k,v in r.items()},side=round(float(sl),1),n=len(P),wn=round(10*np.log10(len(P)),1))
        print(f"{name:34s} {key:6s} DI@{SPOT}: {' '.join(f'{x:5.1f}' for x in s)} | band {b:4.1f} | rej60 {r[60]:6.1f} rej90 {r[90]:6.1f} rej180 {r[180]:6.1f} | worst lobe {sl:6.1f}")
json.dump(res,open('res.json','w'),indent=1)
