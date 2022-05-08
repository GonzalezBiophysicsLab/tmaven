import logging
logger = logging.getLogger(__name__)


def dwell_inversion(gui):
	logger.info('Experimental: Dwell Inversion')

	import numpy as np
	import matplotlib.pyplot as plt
	from scipy.optimize import minimize

	def newton(w,k,t,s,verbose=False):
		delta = k[1]-k[0]
		r = s*t/(1-np.exp(-delta*t))
		gamma = .1
		l0 = -np.inf
		for iter in range(100):
			dlnL = -np.sum(np.exp(-k[:,None]*t[None,:])*(r-np.sum(w[:,None]*np.exp(-k[:,None]*t[None,:]),axis=0)),axis=1)
			ddlnL = +1.*np.sum(np.exp(-(k[:,None,None]+k[None,:,None])*t[None,None,:]),axis=2)
			ddlnLinv = np.linalg.inv(ddlnL)
			w = w - gamma*np.dot(ddlnLinv,dlnL)
			w = np.abs(w)
			# from scipy.ndimage import gaussian_filter1d
			# w = gaussian_filter1d(w,.5)
			w /= w.sum()*delta
			lnL = .5*np.sum((r-np.sum(w*np.exp(-k[None,:]*t[:,None]),axis=1))**2.)
			l1 = lnL
			rel = (l1-l0)/np.abs(l0)
			if verbose: print(iter, rel, lnL)
			l0 = l1
			if rel < 1e-10 and iter > 4:
				break
		ss = (1.-np.exp(-delta*t))/t * np.sum(w[:,None]*np.exp(-k[:,None]*t[None,:]),axis=0)
		return w,ss

	def simplex(w,k,t,s):
		delta = k[1]-k[0]
		def fxn(w,s,delta,k,t):
			if np.any(w < 0):
				return np.inf
			return np.sum((s - (1.-np.exp(-delta*t))/t * np.sum(w[:,None]*np.exp(-k[:,None]*t[None,:]),axis=0))**2.)
		out = minimize(fxn,x0=w,args=(s,delta,k,t),method='Nelder-Mead')
		ww = out.x
		ss = (1.-np.exp(-delta*t))/t * np.sum(ww[:,None]*np.exp(-k[:,None]*t[None,:]),axis=0)
		return ww,ss

	try:
		from ..controllers.modeler.dwells import calculate_dwells
		calculate_dwells(gui.maven.modeler.model)
	except:
		pass
	try:
		t,s = gui.maven.modeler.get_survival_dwells(gui.maven.plots.survival_dwell.prefs['dwell_state']) ## blows up if t = 0
		t = t[1:].astype('double')
		s = s[1:].astype('double')
		t *= gui.maven.prefs['plot.time_dt']
	except:
		logger.info('Failed to get dwells')
		return

	n = np.min((500,t.size//2))
	k = np.linspace(0,(t[1]-t[0])/2.,n+1)[1:]
	w = np.ones(k.size)/((k[1]-k[0])*k.size)
	logger.info('Running Newton optimizer')
	w1,s1 = newton(w,k,t,s,verbose=False)
	logger.info('Running Nelder-Mead optimizer')
	w2,s2 = simplex(w1,k,t,s)

	from PyQt5.QtWidgets import QMainWindow,QApplication,QSizePolicy,QVBoxLayout,QWidget
	from PyQt5.QtCore import QSize
	from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
	from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

	qmw = QMainWindow()
	app = QApplication.instance()
	screen = app.screens()[0]
	dpi = screen.physicalDotsPerInch()
	dpr = screen.devicePixelRatio()
	fig,ax = plt.subplots(3,figsize=(5,6),dpi=dpi)
	qmw.canvas = FigureCanvas(fig)
	toolbar = NavigationToolbar(qmw.canvas,None)
	fig = qmw.canvas.figure
	qmw.canvas.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
	toolbar.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
	fig.set_dpi(dpi)
	qmw.canvas.sizeHint = lambda : QSize(5*dpi, 6*dpi)

	qw = QWidget()
	vbox = QVBoxLayout()
	vbox.addWidget(qmw.canvas)
	vbox.addWidget(toolbar)
	qw.setLayout(vbox)
	qmw.setCentralWidget(qw)

	ax[0].step(t,s,'k')
	ax[0].step(t,s2)

	ax[1].step(k,w2)

	beta = 1./0.593 ## kt @ room temp in kcal/mol
	A = 1./(1.6*10**(-13.)) #kappa kb T / h; kappa = 1
	E = -np.log(k/A)/beta
	p_E = w2 * beta*k
	ax[2].step(E,p_E)

	# ax[2].legend(['GS1->GS2 (?)','GS2->GS1 (?)'])
	ax[2].set_xlabel(r'E (kcal/mol)')
	ax[2].set_ylabel(r'Spectral Density')
	# ax[1].legend(['GS1->GS2 (?)','GS2->GS1 (?)'])
	ax[1].set_xscale('log')
	ax[1].set_xlabel(r'k (frame$^{-1}$)')
	ax[1].set_ylabel(r'Spectral Density')
	# ax[1].set_xlim(k.min(),k.max())
	ax[0].set_xlabel(r'Time (frame)')
	ax[0].set_ylabel(r'Survival ($\tau > t$)')
	fig.tight_layout()
	qmw.show()
