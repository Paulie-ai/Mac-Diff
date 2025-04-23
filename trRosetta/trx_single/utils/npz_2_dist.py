#!/usr/bin/env /usr/bin/python
import sys
import numpy as np


def logo():
    print('*********************************************************************')
    print('\
*           _        ____                _   _                      *\n\
*          | |_ _ __|  _ \ ___  ___  ___| |_| |_ __ _               *\n\
*          | __| \'__| |_) / _ \/ __|/ _ \ __| __/ _` |              *\n\
*          | |_| |  |  _ < (_) \__ \  __/ |_| || (_| |              *\n\
*           \__|_|  |_| \_\___/|___/\___|\__|\__\__,_|              *')
    print('*                                                                   *')
    print("* J Yang et al, Improved protein structure prediction using         *\n* predicted interresidue orientations, PNAS, 117: 1496-1503 (2020)  *")
    print("* Please email your comments to: yangjy@nankai.edu.cn               *")
    print('*********************************************************************')


if (len(sys.argv) < 3):

    logo()
    print('\n This script converts the distance distribtuion in the npz file \n (predicted from trrosetta network: https://github.com/gjoni/trRosetta) \n into distance map based on weighted average, \n and visualizes the distance and contact maps with the matplotlib module.\n')
    print(' Usage:  python npz_2_dist.py npz_file_name prefix_of_output\n')
    print(' Example: python npz_2_dist.py seq.npz seq \n The output files include: seq_dist.png, seq_cont.png, seq_dist_cont.npz\n')
    exit(1)



NPZ = sys.argv[1]
OUT = sys.argv[2]

npz = np.load(NPZ)


import matplotlib, pandas
matplotlib.use('Agg')
import matplotlib.pyplot as plt





matplotlib.rcParams['image.cmap'] = 'gist_stern'

def weighted_avg_and_std(values, weights):
    if(len(values)==1):return values[0],0

    average = np.average(values, weights=weights)
    variance=np.array(values).std()
    return average, variance


def dist(npz):
    dat=npz['dist']
    nres=int(dat.shape[0])

    pcut=0.05
    bin_p=0.01

    mat=np.zeros((nres, nres))
    cont=np.zeros((nres, nres))
    
    for i in range(0, nres):
        for j in range(0, nres):
            
            if(j == i):
                mat[i][i]=4
                cont[i][j]=1
                continue

            if(j<i):
                mat[i][j]=mat[j][i]
                cont[i][j]=cont[j][i]
                continue

                       
            #check probability
            Praw = dat[i][j]
            first_bin=5
            first_d=4.25 #4-->3.75, 5-->4.25
            weight=0

            pcont=0
            for ii in range(first_bin, 13):
                pcont += Praw[ii]
            cont[i][j]=pcont
            
            for P in Praw[first_bin:]:
                if(P>bin_p): weight += P
                
                
            if(weight < pcut):
                mat[i][j]=20
                continue


            Pnorm = [P for P in Praw[first_bin:]]

            probs=[]
            dists=[]
            xs = []
            ys = []
            dmax=0
            pmax=-1

            for k,P in enumerate(Pnorm):
                d = first_d + k*0.5
                if(P>pmax):
                    dmax=d
                    pmax=P
                #endif

                if(P>bin_p):
                    probs.append(P)
                    dists.append(d)
                #endif
                xs.append(d)

            e_dis=8;
            e_std=0;
            if(len(probs)==0):
                e_dis=dmax
                e_std=0;
            else:
                probs = [P/sum(probs) for P in probs]

                e_dis, e_std=weighted_avg_and_std(dists, probs)

            mat[i][j]=e_dis


    return(mat,cont)
#def dist

def tocontact(contacts, out_file):
    w = np.sum(contacts['dist'][:,:,1:13], axis=-1)
    L = w.shape[0]
    idx = np.array([[i+1,j+1,0,8,w[i,j]] for i in range(L) for j in range(i+5,L)])
    out = idx[np.flip(np.argsort(idx[:,4]))]

    data = [out[:,0].astype(int), out[:,1].astype(int), out[:,2].astype(int), out[:,3].astype(int), out[:,4].astype(float)]
    df = pandas.DataFrame(data)
    df = df.transpose()
    df[0] = df[0].astype(int)
    df[1] = df[1].astype(int)
    df.columns = ["i", "j", "d1", "d2", "p"]
    df.to_csv(out_file, sep=' ', index=False)

logo()
print("convert distance...")

#Distance
plt.figure()
img,cont=dist(npz)
color_map = plt.imshow(img.astype(float))
color_map.set_cmap("hot")
plt.colorbar()
name = OUT+'_'+"dist"+'.png'    
plt.savefig(name, bbox_inches='tight', dpi=600, transparent=False)


#contact
plt.figure()
color_map = plt.imshow(cont.astype(float))
color_map.set_cmap("hot_r")
plt.colorbar()
name = OUT+'_'+"cont"+'.png'    
plt.savefig(name, bbox_inches='tight', dpi=600, transparent=False)


for i in range(0,img.shape[0]): img[i,i]=0

outname=OUT + '_dist_cont.npz'
np.savez_compressed(outname, dist=img,contact=cont)


print("The conversion is done.\n")

