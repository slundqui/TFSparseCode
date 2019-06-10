import pdb
import numpy as np
import h5py
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

filename = "/mnt/drive1/large_seismic_hdf5/data.txt"

#filename is a list of filenames
with open(filename, 'r') as f:
  list_of_filenames = f.readlines()

list_of_filenames = [fn[:-1] for fn in list_of_filenames]

srcs = []
mag = []
for filename in list_of_filenames:
  f = h5py.File(filename, 'r')
  #Get latitude/logitude/depth from srcs
  srcs.append(f['srcs'][:, :3])
  mag.append(f['srcs'][:, 5])

srcs = np.concatenate(srcs, axis=0)
mag = np.concatenate(mag, axis=0)

#Make histogram of magnitudes
fig = plt.figure()
plt.hist(mag)
plt.savefig("/home/slundquist/Work/Projects/seismic_vis/mag_hist.png")
plt.close('all')

lat_filter = np.logical_and(srcs[:, 0] > -26, srcs[:, 1] < -24)
lon_filter = np.logical_and(srcs[:, 1] > -71, srcs[:, 1] < -69)
depth_filter = np.logical_and(srcs[:, 2] > -100000, srcs[:, 2] < -25000)
loc_filter = np.logical_and(np.logical_and(lat_filter, lon_filter), depth_filter)
mag_filter = mag >= 1
total_filter = np.logical_and(loc_filter, mag_filter)
filter_idx = np.nonzero(total_filter)

#
#srcs = srcs[filter_idx]
#mag = mag[filter_idx]
pdb.set_trace()



#Make 2d plot of locs
fig = plt.figure()
ax = fig.add_subplot(111)
#sc = ax.scatter(srcs[:, 0], srcs[:, 1], c=srcs[:, 2])
sc = ax.scatter(srcs[:, 0], srcs[:, 1], c=mag)

fig.colorbar(sc)

ax.set_xlabel('latitude')
ax.set_ylabel('logitude')

plt.savefig('/home/slundquist/Work/Projects/seismic_vis/locs_2d.png')
plt.close('all')


#Make 3d plot of locs
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(srcs[:, 0], srcs[:, 1], srcs[:, 2], c=mag)
fig.colorbar(sc)

ax.set_xlabel('latitude')
ax.set_ylabel('logitude')
ax.set_zlabel('depth')

plt.savefig('/home/slundquist/Work/Projects/seismic_vis/locs_3d.png')
plt.close('all')

pdb.set_trace()
