import matplotlib.pyplot as plt
import numpy as np

# ds diff
a = open("bs-64 lr-0.100000 -ds30.txt", "r").readlines()
b = open('bs-64 lr-0.100000 -ds40.txt', 'r').readlines()
c = open('bs-64 lr-0.100000 -ds50.txt', 'r').readlines()
d = open('bs-64 lr-0.100000 -ds60.txt', 'r').readlines()

# lr diff and ds diff
e = open('bs-32 lr-0.100000 -ds40.txt', 'r').readlines()
f = open('bs-32 lr-0.100000 -ds40 with ema.txt', 'r').readlines()
g = open('bs-32 lr-0.100000 -ds50 with ema.txt', 'r').readlines()
h = open('bs-32 lr-0.200000 -ds40 with ema.txt', 'r').readlines()
i = open('bs-32 lr-0.200000 -ds50 with ema.txt', 'r').readlines()
j = open('bs-32 lr-0.200000 -ds60 with ema.txt', 'r').readlines()

# bs diff
k = open('bs-32 lr-0.100000 -ds40.txt', 'r').readlines()
l = open('bs-64 lr-0.100000 -ds40.txt', 'r').readlines()
m = open('bs-128 lr-0.100000 -ds40.txt', 'r').readlines()
n = open('bs-256 lr-0.100000 -ds40.txt', 'r').readlines()

# adam optimizer diff
o = open('bs-64 lr-0.100000 -ds40 adam betas=(0.7, 0.7) -eps=1e-8.txt', 'r').readlines()
p = open('bs-64 lr-0.100000 -ds40 adam betas=(0.8, 0.8) -eps=1e-8.txt', 'r').readlines()
q = open('bs-64 lr-0.100000 -ds40 adam betas=(0.9, 0.999) -eps=1e-5.txt', 'r').readlines()
r = open('bs-64 lr-0.100000 -ds40 adam betas=(0.9, 0.999) -eps=1e-6.txt', 'r').readlines()
s = open('bs-64 lr-0.100000 -ds40 adam betas=(0.9, 0.999) -eps=1e-7.txt', 'r').readlines()
t = open('bs-64 lr-0.100000 -ds40 adam betas=(0.9, 0.999) -eps=1e-8.txt', 'r').readlines()

# sgd with cross entropy
u = open('bs-64 lr-0.100000 -ds40 -sgd -CrossEntropy.txt', 'r').readlines()
v = open('bs-64 lr-0.100000 -ds50 -sgd -CrossEntropy.txt', 'r').readlines()
w = open('bs-64 lr-0.150000 -ds40 -sgd -CrossEntropy.txt', 'r').readlines()
x = open('bs-64 lr-0.150000 -ds50 -sgd -CrossEntropy.txt', 'r').readlines()
y = open('bs-64 lr-0.200000 -ds40 -sgd -CrossEntropy.txt', 'r').readlines()
z = open('bs-64 lr-0.200000 -ds50 -sgd -CrossEntropy.txt', 'r').readlines()


for fa in a:
    fa = fa.strip().strip("[]").split(", ")

for fb in b:
    fb = fb.strip().strip("[]").split(", ")

for fc in c:
    fc = fc.strip().strip("[]").split(", ")

for fd in d:
    fd = fd.strip().strip("[]").split(", ")
for fe in e:
    fe = fe.strip().strip("[]").split(", ")

for ff in f:
    ff = ff.strip().strip("[]").split(", ")

for fg in g:
    fg = fg.strip().strip("[]").split(", ")

for fh in h:
    fh = fh.strip().strip("[]").split(", ")
for fi in i:
    fi = fi.strip().strip("[]").split(", ")

for fj in j:
    fj = fj.strip().strip("[]").split(", ")

for fk in k:
    fk = fk.strip().strip("[]").split(", ")

for fl in l:
    fl = fl.strip().strip("[]").split(", ")
for fm in m:
    fm = fm.strip().strip("[]").split(", ")

for fn in n:
    fn = fn.strip().strip("[]").split(", ")

for fo in o:
    fo = fo.strip().strip("[]").split(", ")

for fp in p:
    fp = fp.strip().strip("[]").split(", ")
for fq in q:
    fq = fq.strip().strip("[]").split(", ")

for fr in r:
    fr = fr.strip().strip("[]").split(", ")

for fs in s:
    fs = fs.strip().strip("[]").split(", ")
for ft in t:
    ft = ft.strip().strip("[]").split(", ")

for fu in u:
    fu = fu.strip().strip("[]").split(", ")

for fv in v:
    fv = fv.strip().strip("[]").split(", ")

for fw in w:
    fw = fw.strip().strip("[]").split(", ")

for fx in x:
    fx = fx.strip().strip("[]").split(", ")

for fy in y:
    fy = fy.strip().strip("[]").split(", ")

for fz in z:
    fz = fz.strip().strip("[]").split(", ")

zzz = open('opti.txt').readlines()

for fzzz in zzz:
    fzzz = fzzz.strip().strip("[]").split(", ")

fzzz = [float(x) for x in fzzz]

fa = [float(x) for x in fa]
fb = [float(x) for x in fb]
fc = [float(x) for x in fc]
fd = [float(x) for x in fd]
fe = [float(x) for x in fe]
ff = [float(x) for x in ff]
fg = [float(x) for x in fg]
fh = [float(x) for x in fh]
fi = [float(x) for x in fi]
fj = [float(x) for x in fj]
fk = [float(x) for x in fk]
fl = [float(x) for x in fl]
fm = [float(x) for x in fm]
fn = [float(x) for x in fn]
fo = [float(x) for x in fo]
fp = [float(x) for x in fp]
fq = [float(x) for x in fq]
fr = [float(x) for x in fr]
fs = [float(x) for x in fs]
ft = [float(x) for x in ft]
fu = [float(x) for x in fu]
fv = [float(x) for x in fv]
fw = [float(x) for x in fw]
fx = [float(x) for x in fx]
fy = [float(x) for x in fy]
fz = [float(x) for x in fz]

x = np.linspace(0, 200, 200)
ya = np.array(fa)
yb = np.array(fb)
yc = np.array(fc)
yd = np.array(fd)
ye = np.array(fe)
yf = np.array(ff)
yg = np.array(fg)
yh = np.array(fh)
yi = np.array(fi)
yj = np.array(fj)
yk = np.array(fk)
yl = np.array(fl)
ym = np.array(fm)
yn = np.array(fn)
yo = np.array(fo)
yp = np.array(fp)
yq = np.array(fq)
yr = np.array(fr)
ys = np.array(fs)
yt = np.array(ft)
yu = np.array(fu)
yv = np.array(fv)
yw = np.array(fw)
yx = np.array(fx)
yy = np.array(fy)
yz = np.array(fz)

yzzz = np.array(fzzz)

plt.xlim(0,210)
plt.ylim(80,100)
plt.plot(x, ya, label = 'bs-64 lr-0.10 -ds30')
plt.plot(x, yb, label = 'bs-64 lr-0.10 -ds40')
plt.plot(x, yc, label = 'bs-64 lr-0.10 -ds50')
plt.plot(x, yd, label = 'bs-64 lr-0.10 -ds60')
plt.title("Accuracy with Decay Step")
plt.ylabel("Test Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()

plt.xlim(0,210)
plt.ylim(75,100)
plt.plot(x, ye, label = 'bs-32 lr-0.10 -ds40')
plt.plot(x, yf, label = 'bs-32 lr-0.10 -ds40 with ema')
plt.plot(x, yg, label = 'bs-32 lr-0.10 -ds50 with ema')
plt.plot(x, yh, label = 'bs-32 lr-0.20 -ds40 with ema')
plt.plot(x, yi, label = 'bs-32 lr-0.20 -ds50 with ema')
plt.plot(x, yj, label = 'bs-32 lr-0.20 -ds60 with ema')
plt.title("Accuracy with Learning Rate")
plt.ylabel("Test Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()

plt.xlim(0,210)
plt.ylim(80,100)
plt.plot(x, yk, label = 'bs-32 lr-0.10 -ds40')
plt.plot(x, yl, label = 'bs-64 lr-0.10 -ds40')
plt.plot(x, ym, label = 'bs-128 lr-0.10 -ds40')
plt.plot(x, yn, label = 'bs-128 lr-0.10 -ds40')
plt.title("Accuracy with Batch Size")
plt.ylabel("Test Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()

plt.xlim(0,210)
plt.ylim(40,100)
plt.plot(x, yo, label = 'bs-64 lr-0.10 -ds40 adam betas=(0.7, 0.7) -eps=1e-8')
plt.plot(x, yp, label = 'bs-64 lr-0.10 -ds40 adam betas=(0.8, 0.8) -eps=1e-8')
plt.plot(x, yq, label = 'bs-64 lr-0.10 -ds40 adam betas=(0.9, 0.999) -eps=1e-5')
plt.plot(x, yr, label = 'bs-64 lr-0.10 -ds40 adam betas=(0.9, 0.999) -eps=1e-6')
plt.plot(x, ys, label = 'bs-64 lr-0.10 -ds40 adam betas=(0.9, 0.999) -eps=1e-7')
plt.plot(x, yt, label = 'bs-64 lr-0.10 -ds40 adam betas=(0.9, 0.999) -eps=1e-8')
plt.title("Accuracy with ADAM")
plt.ylabel("Test Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()

plt.xlim(0,210)
plt.ylim(80,100)
plt.plot(x, yu, label = 'bs-64 lr-0.10 -ds40 -sgd -CrossEntropy')
plt.plot(x, yv, label = 'bs-64 lr-0.10 -ds50 -sgd -CrossEntropy')
plt.plot(x, yw, label = 'bs-64 lr-0.15 -ds40 -sgd -CrossEntropy')
plt.plot(x, yx, label = 'bs-64 lr-0.15 -ds50 -sgd -CrossEntropy')
plt.plot(x, yy, label = 'bs-64 lr-0.20 -ds40 -sgd -CrossEntropy')
plt.plot(x, yz, label = 'bs-64 lr-0.20 -ds50 -sgd -CrossEntropy')
plt.title("Accuracy with SGD and CrossEntropy")
plt.ylabel("Test Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()

plt.xlim(0,210)
plt.ylim(80,100)
plt.plot(x, yzzz, label = 'bs-64 lr-0.10 -ds50')
plt.title("Accuracy of final model")
plt.ylabel("Test Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()
