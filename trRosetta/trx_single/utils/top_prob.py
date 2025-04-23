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


def main():
    if (len(sys.argv) < 2):

        logo()
        print('\n This script computes the average probability of the top 15L long+medium-range \n (i.e., |i-j|>=12) predicted distance from the npz file.\n')
        print(' Please note that higher probability usually yileds more accurate 3D models.\n')

        print(' Example usage: python3 top_prob.py seq.npz\n')
        exit(1)



    NPZ = sys.argv[1]
    dat = np.load(NPZ)['dist']
    #print(dat)
    logo()
    dist,sepmax=top_dist(dat,12)
    print("\nAverage probability of the top predicted distances: %.2f\n" %(dist))

def top_cont(dat,separation):
    wc = np.sum(dat[:,:,1:13], axis=-1) #pcon
    
    L=wc.shape[0]
    idxc = np.array([[i+1,j+1,wc[i,j]] for j in range(L) for i in range(j+separation,L)])
    precon = idxc[np.flip(np.argsort(idxc[:,2]),axis=0)]
    topn=min(L,len(precon))
    top_cont=round(np.mean(precon[:topn,:][:,2]),2)
    return top_cont


def top_dist(dat,separation):

    w  = np.sum(dat[:,:,1:37],axis=-1) #p
    w1 = np.sum(dat[:,:,1:5], axis=-1) # p1
    w2 = np.sum(dat[:,:,5:9], axis=-1) # p2
    w3 = np.sum(dat[:,:,9:13], axis=-1) # p3
    w4 = np.sum(dat[:,:,13:17], axis=-1) # p4
    w5 = np.sum(dat[:,:,17:21], axis=-1) # p5
    w6 = np.sum(dat[:,:,21:25], axis=-1) # p6
    w7 = np.sum(dat[:,:,25:29], axis=-1) # p7
    w8 = np.sum(dat[:,:,29:33], axis=-1) # p8
    w9 = np.sum(dat[:,:,33:37], axis=-1) # p9


    L = w.shape[0]
    idx = np.array([[i+1,j+1,w1[i,j], w2[i,j], w3[i,j], w4[i,j], w5[i,j], w6[i,j], w7[i,j], w8[i,j], w9[i,j], w[i,j]] for j in range(L) for i in range(j+separation,L)])
    predis = idx[np.flip(np.argsort(idx[:,-1]),axis=0)]  #rewrite idx by last column of idx

    topn=min(15*L,len(predis))
    predis=predis[:topn,:]

    bins=np.argmax(predis[:,2:-1],axis=1)
    probs=predis[:,2:-1][range(len(predis)),bins]

    means=[]
    for i in range(9):
        idx=np.where(bins==i)[0]
        if len(idx) !=0:
            imean=np.mean(probs[idx])
            means.append(imean)
    top_dist=round(np.mean(means),2)
    
    seps=abs(predis[:,0]-predis[:,1])
    sepmax=np.max(seps)/L
    
    return top_dist,sepmax   
    


if __name__ == "__main__":
    main()

