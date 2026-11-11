"""ПР4 — модель процессов П1 и П2 ЮMoney (Монте-Карло) и сверка базового уровня выручки с историей.
Параметры шагов — из таблиц потерь ПР3 (А28–А44). Ожидания — логнормальные с CV=1 (А56), работа — CV=0,3.
"""
import numpy as np
SEED = 20261111
rng = np.random.default_rng(SEED)
N = 40000
H = 8.0  # рабочих часов в дне

def ln(mean, cv, n):
    if mean <= 0: return np.zeros(n)
    s2 = np.log(1 + cv**2); mu = np.log(mean) - s2/2
    return rng.lognormal(mu, np.sqrt(s2), n)

def wait(days, n, cv=1.0): return ln(days, cv, n)          # в рабочих днях
def work(hours, n, cv=0.3): return ln(hours, cv, n) / H     # в рабочих днях

def stats(x):
    return dict(mean=x.mean(), p50=np.percentile(x,50), p85=np.percentile(x,85), p95=np.percentile(x,95))

# ---------- П1: выдача данных ----------
def p1_before(n=N):
    rew = rng.random(n) < 0.30
    t = (work(.5,n)+wait(1,n) + work(1,n)+wait(1,n) + rew*(work(.5,n)+wait(1,n))
         + work(2,n)+wait(1.5,n) + work(8,n)+wait(1,n) + work(1.5,n)+wait(1,n) + work(.5,n)+wait(.5,n))
    return t
def p1_after(trig, n=N):
    tr = rng.random(n) < trig
    base = work(.25,n)+wait(2/H,n) + work(.25,n) + work(1.1,n)+wait(.5,n) + work(.1,n)
    ib_std = work(.25,n)+wait(2/H,n)
    legal = tr*(work(2,n)+wait(1,n))
    ib_full = tr*(work(1.5,n)+wait(.5,n))
    t = base + np.where(tr, ib_full, ib_std) + legal
    return t, tr

# ---------- П2: поставка изменения API ----------
def p2_before(n=N):
    rew = rng.random(n) < 0.25
    t = (work(2,n)+wait(3,n) + work(32,n) + work(1,n)+wait(1,n) + rew*(work(4,n)+wait(.5,n))
         + work(4,n)+wait(2,n) + work(1.5,n)+wait(2.5,n) + work(1,n)+wait(5,n) + work(2,n)+wait(1,n))
    return t
def p2_after(dev_h=24, review_wait=.5, rew_p=.15, n=N):
    rew = rng.random(n) < rew_p
    normal = rng.random(n) < .30
    t = (work(1,n)+wait(3,n) + work(dev_h,n) + work(.25,n) + work(1.5,n)+wait(review_wait,n)
         + rew*(work(dev_h/8,n)+wait(review_wait,n))
         + work(1,n)+wait(.5,n) + work(.3,n)+normal*wait(1,n) + work(.5,n)+wait(.5,n) + work(.5,n)+wait(.5,n))
    return t

CAP = 2.1*168  # часов в месяц на поток
def throughput(hours_per_change, cfr, cap=CAP): return cap/hours_per_change*(1-cfr)


# ===== Часть 1: сценарии, эффект, окупаемость =====
rng = np.random.default_rng(SEED)
import numpy as np
R = lambda x,n=2: round(float(x),n)
# --- П1 сценарии ---
b = p1_before(); print('P1 base', R(b.mean()), [R(np.percentile(b,q)) for q in (50,85,95)], 'sla', R((b<=2).mean(),3))
res={}
for nm,trig in [('exp',.35),('pess',.45)]:
    t,tr = p1_after(trig); sla = np.where(tr,t<=4,t<=2)
    res[nm]=sla.mean()
    print('P1',nm,R(t.mean()),[R(np.percentile(t,q)) for q in (50,85,95)],'sla',R(sla.mean(),3),'std sla',R((t[~tr]<=2).mean(),3),'trg sla',R((t[tr]<=4).mean(),3))
# чувствительность: ожидание аналитика 0,5 -> 0,25 дня и ИБ 2ч->1ч
def p1_after2(trig, aw=.5, n=N):
    tr = rng.random(n) < trig
    base = work(.25,n)+wait(2/H,n) + work(.25,n) + work(1.1,n)+wait(aw,n) + work(.1,n)
    t = base + np.where(tr, work(1.5,n)+wait(.5,n), work(.25,n)+wait(2/H,n)) + tr*(work(2,n)+wait(1,n))
    return t,tr
for aw in (.5,.25):
    t,tr=p1_after2(.35,aw); print('sens aw',aw,R(np.where(tr,t<=4,t<=2).mean(),3), R(np.percentile(t[~tr],85)))
# Г1: магазин считает обещание нарушенным, если в квартале >=2 из 3 выдач опоздали (А57)
def broken(p): q=1-p; return 3*q*q*p+q**3
for k,p in [('base',0.0),('exp',res['exp']),('pess',res['pess'])]:
    print('G1 broken',k,R(broken(p),3))
# --- П2 ---
b2=p2_before(); e2=p2_after(); s2=p2_after(dev_h=32,review_wait=1.0,rew_p=.20); n2=p2_after(dev_h=32)
for nm,x in [('base',b2),('exp',e2),('pess',s2),('noAI',n2)]:
    print('P2',nm,R(x.mean()),[R(np.percentile(x,q)) for q in (50,85,95)])
promise = np.percentile(e2,85)+2  # дата = P85 ожидаемой модели + резерв 2 дня (И18)
print('promise',R(promise))
def stay(p): return p**3+3*p*p*(1-p)
for nm,x in [('base',b2),('exp',e2),('pess',s2),('noAI',n2)]:
    p=(x<=promise).mean(); print('G3',nm,'hit',R(p,3),'stay',R(stay(p),3))
cap=2.1*168
for nm,h,c in [('base',44.5,.15),('exp',29.5,.15),('pess',37.4,.20),('noAI',37.2,.15)]:
    print('thr',nm,R(cap/h*(1-c)))
# --- экономия ФОТ (валовая, до новых постоянных затрат), млн ₽/год ---
r=1740
p1_exp=(360*13.7/0.9-360*2.4/0.95)*r/1e6; p1_pess=(360*13.7/0.9-360*2.4/0.90)*r/1e6
p2_exp=(44.5/0.85-29.5/0.85)*81*r/1e6; p2_pess=(44.5/0.85-37.4/0.80)*81*r/1e6
ss_exp=260*2.93/0.95*r/1e6; ss_pess=160*3.08/0.90*r/1e6
print('save P1',R(p1_exp),R(p1_pess),'P2',R(p2_exp),R(p2_pess),'self',R(ss_exp),R(ss_pess))
# --- помесячный поток: разовые и постоянные, окупаемость ---
def flow(sc):
    one = {'П1 изменения':.88,'подпись+ЕГРЮЛ':.29,'самообслуживание':.58,'пилот ИИ':.09,'миграция':.15,'замеры и экран':.44,'месяц ретеста':.04}
    months=np.arange(1,25); cum=-sum(one.values()); out=[]
    for m in months:
        if sc=='exp':
            g = (p1_exp/12 if m>=2 else 0) + (p2_exp/12 if m>=4 else 0) + (ss_exp/12 if m>=5 else 0)
            c = (0.09+0.33+0.16)/12*(m>=2) + 0.72/12*(m>=4) + 0.20/12*(m>=5) + 0.35/12
        else:
            g = (p1_pess/12 if m>=3 else 0) + (p2_pess/12 if m>=5 else 0) + (ss_pess/12 if m>=7 else 0)
            c = (0.09+0.42+0.16)/12*(m>=3) + 0.20/12*(m>=7) + 0.35/12
        cum += g-c; out.append(cum)
    pb = next((m for m,v in zip(months,out) if v>=0),None)
    return sum(one.values()), pb, out[11]
for sc in ('exp','pess'): print('flow',sc,[R(x) for x in flow(sc)])
def path(sc):
    one=2.47; cum=-one; out=[]
    for m in range(1,13):
        if sc=='exp':
            g=(p1_exp/12 if m>=2 else 0)+(p2_exp/12 if m>=4 else 0)+(ss_exp/12 if m>=5 else 0)
            c=(0.09+0.33+0.16)/12*(m>=2)+0.72/12*(m>=4)+0.20/12*(m>=5)+0.35/12
        else:
            g=(p1_pess/12 if m>=3 else 0)+(p2_pess/12 if m>=5 else 0)+(ss_pess/12 if m>=7 else 0)
            c=(0.09+0.42+0.16)/12*(m>=3)+0.20/12*(m>=7)+0.35/12
        cum+=g-c; out.append(round(cum,2))
    return out
print('exp',path('exp')); print('pess',path('pess'))

# ===== Часть 2: прогноз на декабрь 2026 (месяц из 30 выдач и ~7 выкаток) =====
rng = np.random.default_rng(SEED)
import numpy as np
R=lambda x:round(float(x),2)
K=20000
# П1: 30 запросов в декабре
t,tr=p1_after(.35,n=K*30); t=t.reshape(K,30); tr=tr.reshape(K,30)
sla=np.where(tr,t<=4,t<=2).mean(1); p85=np.percentile(t,85,axis=1); trs=tr.mean(1)
std85=np.array([np.percentile(a[~b],85) for a,b in zip(t[:3000],tr[:3000])])
for nm,x in [('sla',sla),('p85',p85),('trig',trs),('std85',std85)]:
    print(nm,R(np.median(x)),R(np.percentile(x,10)),R(np.percentile(x,90)))
# П2 в декабре: новый поток без ИИ, ~7 выкаток
x=p2_after(dev_h=32,n=K*7).reshape(K,7)
p85=np.percentile(x,85,axis=1); p50=np.median(x,axis=1)
print('p2 p85',R(np.median(p85)),R(np.percentile(p85,10)),R(np.percentile(p85,90)))
print('p2 p50',R(np.median(p50)),R(np.percentile(p50,10)),R(np.percentile(p50,90)))
# принятые: число выкаток ~ Пуассон(7,4), каждая неудачна с p=0,15
dep=rng.poisson(7.4,K); acc=rng.binomial(dep,0.85)
print('acc',np.median(acc),np.percentile(acc,10),np.percentile(acc,90))
cfr=np.where(dep>0,(dep-acc)/np.maximum(dep,1),0); print('cfr',R(np.median(cfr)),R(np.percentile(cfr,10)),R(np.percentile(cfr,90)))
print('rev2026',R(27.55*1.12),R(27.55*1.12*(1-.033)),R(27.55*1.12*1.033))

# ---------- сверка базового уровня выручки с историей ----------
# настройка: рост 2024->2025 = 15,89% (Smart Ranking); проверка: II кв. 2026 (Коммерсантъ)
pred = 6.53*1.1589; fact = 7.325
print('history: прогноз II кв. 2026', round(pred,2), 'факт', fact, 'ошибка', round((pred/fact-1)*100,1), '%')
