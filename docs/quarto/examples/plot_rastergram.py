import numpy as np
import matplotlib.pyplot as plt

inds = [0,]

m = maven.modeler.model
nt = maven.data.nt
nk = m.nstates
kb = 1.

vits = m.chain
nmol,nt = vits.shape

pre = maven.data.pre_list
post = maven.data.post_list

data = np.zeros((nmol,nt)) + np.nan
for i in range(nmol):
	if post[i]-pre[i] > 1:
		data[i,0:post[i]-pre[i]] = np.isin(vits[i][pre[i]:post[i]],inds).astype('float')


first = np.nanargmax(data,axis=1)
order = first.argsort()
data = data[order]

dt = maven.prefs['plot.time_dt']
desc = m.description()
desc = desc.split('] ')[1]
desc = ''.join(desc.split(' -')[:-1])

fig,ax = plt.subplots(1,figsize=(3,3))
ax.pcolormesh(np.arange(nt)*dt,np.arange(nmol),data,cmap='Greys')
ax.set_ylabel('Molecules')
ax.set_yticks(())
ax.set_xlabel('Time (s)')
ax.set_title(desc)
fig.tight_layout()
plt.show()

