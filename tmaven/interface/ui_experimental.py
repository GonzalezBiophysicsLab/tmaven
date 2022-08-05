import logging
logger = logging.getLogger(__name__)


def dwell_inversion(gui):
	logger.info('Experimental: Dwell Inversion')

	import numpy as np
	import matplotlib.pyplot as plt

	def fxn(x,args):
		### data is N[nbatch],x
		if np.any(x<0):
			return np.inf
		data,taua,taub,tauc,k, = args
		t = data[:,0]
		s = data[:,1]
		x /= np.sum(x*(k[1:]-k[:-1]))

		out = -taua/2.*np.sum((s*t - np.sum(x[None,:]*(np.exp(-k[None,:-1]*t[:,None])-np.exp(-k[None,1:]*t[:,None])),axis=1))**2.)
		out += -taub/8.*np.sum((x[2:]-2.*x[1:-1]+x[:-2])**2.)
		out += -taub/2.*((x[1]-x[0])**2. + (x[-1]-x[-2])**2.)
		# out += -tauc/2.*(1.-np.sum(x*(k[1:]-k[:-1])))**2.
		return -out

	def minimize(w,k,data):
		from scipy.optimize import minimize
		r2 = 1.
		l0 = np.inf
		iter = 0
		args = [data,1.e2,1.e1,1.e1,k]
		# args = [data,1.e5,1.e4,1.e2,k]
		while True:
			iter += 1
			out = minimize(fxn,x0=w,args=args,method='Nelder-Mead', options={'maxiter':1000})
			w = out.x
			w /= np.sum(w*(k[1:]-k[:-1]))
			ss = 1./t * np.sum(w[None,:]*(np.exp(-k[None,:-1]*t[:,None])-np.exp(-k[None,1:]*t[:,None])),axis=1)
			r2 = np.nansum((ss-s)**2.)/np.sum(s**2.)

			l1 = out.fun
			rel = np.abs(l0-l1)/np.abs(l0)
			l0 = l1
			print(iter,r2,out.success,out.fun,rel)
			if rel < 1e-4:
				break
			if out.success:
				break
		return w,ss,r2,out.success,out.fun

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
		# np.save('temp.npy',np.array((t,s)))
	except:
		logger.info('Failed to get dwells')
		return

	cutoff = .01
	ndiv = 1
	k = np.logspace(np.log10(1./t[-1]),np.log10(1./(t[1]-t[0])),np.sum(s>cutoff)//ndiv+1)
	t = t[s>cutoff]
	s = s[s>cutoff]
	data = np.swapaxes(np.array((t,s)),0,1)

	logger.info('Running Optimizer')
	from scipy.ndimage import gaussian_filter1d
	w = np.zeros(k.size-1)
	k0 = 1./((t[1]-t[0])*np.sum(s))
	w[np.searchsorted(k,k0)] = 1.
	# w = uniform_filter(w,100)
	w = gaussian_filter1d(w,5.)
	w += np.random.rand(w.size)*.1
	w /= np.sum(w*(k[1:]-k[:-1]))
	w,ss,r2,success,fun = minimize(w,k,data)
	np.save('temp.npy',np.array((k,w,t,s)))

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

	stepx = np.zeros((w.size+1)*2)
	stepy = np.zeros_like(stepx)
	stepx[0::2] = k
	stepx[1::2] = k
	stepy[1:-1:2] = w
	stepy[2:-1:2] = w

	ax[0].plot(t,s,'k')
	ax[0].plot(t,ss)

	ax[1].semilogx(stepx[1:-1],stepy[1:-1])

	beta = 1./0.593 ## kt @ room temp in kcal/mol
	A = 1./(1.6*10**(-13.)) #kappa kb T / h; kappa = 1
	kavg = .5*(k[1:]+k[:-1])
	E = -np.log(kavg/A)/beta
	p_E = w * beta*kavg
	ax[2].step(E,p_E)

	# ax[2].legend(['GS1->GS2 (?)','GS2->GS1 (?)'])
	# ax[0].set_yscale('log')
	ax[2].set_xlabel(r'E (kcal/mol)')
	ax[2].set_ylabel(r'Spectral Density')
	# ax[1].legend(['GS1->GS2 (?)','GS2->GS1 (?)'])
	ax[1].set_xscale('log')
	ax[1].set_xlabel(r'k (s$^{-1}$)')
	ax[1].set_ylabel(r'Spectral Density')
	# ax[1].set_xlim(k.min(),k.max())
	ax[0].set_xlabel(r'Time (s)')
	ax[0].set_ylabel(r'Survival ($\tau > t$)')
	fig.tight_layout()
	qmw.show()
