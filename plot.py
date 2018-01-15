import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mathe import matheplot as mp
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import unumpy

mpl.use('pgf')
mpl.rcParams.update({
    'pgf.preamble': r'\usepackage{siunitx}',
})

t, y = np.genfromtxt('content/Messwerte1.txt', unpack=True)

x= t/26320

plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$U/\si{\volt}$')
plt.yscale('log')
plt.grid(True, which='both')


# Fitvorschrift
def f(x, A, B):
    return np.exp(B)*np.exp(A*x)      #jeweilige Fitfunktion auswaehlen:

params, covar = curve_fit(f, x, y)            #eigene Messwerte hier uebergeben
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
for i in range(0, len(uparams)):
    print(chr(ord('A') + i), "=" , uparams[i])
print()

lin = np.linspace(x[0], x[-1], 1000)
plt.plot(lin, f(lin, *params), "xkcd:orange", label=r'Regression' )
plt.plot(x, y, ".", color="xkcd:blue", label="Messwerte")



plt.tight_layout()
plt.legend()
plt.savefig('build/messung1.pdf')
plt.clf()

plt.figure()
plt.savefig("build/plot.pdf")

data = np.genfromtxt("content/freq.txt", unpack=True)
data[0] /= data[1]
data[2] *= 2*np.pi


def freq(w, L, C, R):
    return 1 / np.sqrt((1-L*C*w**2)**2 + (w*R*C)**2)

params, covar = curve_fit(freq, data[2], data[0], p0=(0.016, 0.000000002, 682))
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))

print(uparams)

plt.plot(data[2], data[0], ".", label="Messwerte")
plt.xlabel(r"$\omega /\si[per-mode=reciprocal]{\second}$")
plt.ylabel(r"$\frac{U_C}{U_0}$")
#lin = np.linspace(np.amin(data[2]), np.amax(data[2]), 1000)
#plt.plot(freq(lin, *params), label="Regression")
plt.xscale("log")
plt.grid(which="both")
plt.legend()
plt.tight_layout()
plt.savefig("build/freq.pdf")

plt.clf()

data = np.genfromtxt("content/phase.txt", unpack=True)

phi = np.array([2*np.pi*data[0] / data[1]])
data = np.concatenate((data, phi))
print(data[3])

def phase(w, L, C, R):
    return np.arctan((-w*R*C)/(1-L*C*w**2))

params, covar = curve_fit(phase, data[2]*2*np.pi, data[3], p0=(0.016, 0.000000002, 682))
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))

print(uparams)

#lin = np.linspace(np.amin(data[2]), np.amax(data[2]), 1000)
#plt.plot(lin, phase(lin*2*np.pi, *params), label="Regression")
plt.plot(data[2], data[3], ".", label="Messdaten")
plt.xlabel(r"$f/\si{\hertz}$")
plt.ylabel(r"$\phi$")
plt.xscale("log")
plt.grid(which="both")
plt.legend()
plt.tight_layout()
plt.savefig("build/phase.pdf")
