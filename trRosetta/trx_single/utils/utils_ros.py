import numpy as np
import random
from pyrosetta import *
from trx_single.utils.top_prob import *


def gen_rst(npz, tmpdir, params):
    if params['USE_ORIENT'] is True or params['USE_ORIENT'] == "True":
        dist, omega, theta, phi = npz['dist'], npz['omega'], npz['theta'], npz['phi']
        # dictionary to store Rosetta restraints
        rst = {'dist': [], 'omega': [], 'theta': [], 'phi': [], 'rep': []}
    else:
        dist = npz['dist']
        rst = {'dist': []}

    ########################################################
    # assign parameters
    ########################################################
    PCUT = 0.05  # params['PCUT']
    PCUT1 = params['PCUT1'] # 0.5
    EBASE = params['EBASE'] # -0.5
    EREP = params['EREP'] # [10,3,0.5]
    DREP = params['DREP'] # [0,2,3.5]
    PREP = params['PREP'] # 0.1
    SIGD = params['SIGD'] # 10
    SIGM = params['SIGM'] # 1
    MEFF = params['MEFF'] # 0.0001
    DCUT = params['DCUT'] # 19.5
    ALPHA = params['ALPHA'] # 1.57

    DSTEP = params['DSTEP'] # 0.5
    ASTEP = np.deg2rad(params['ASTEP']) # 15

    seq = params['seq']

    ########################################################
    # repultion restraints
    ########################################################
    # cbs = ['CA' if a=='G' else 'CB' for a in params['seq']]
    '''
    prob = np.sum(dist[:,:,5:], axis=-1)
    i,j = np.where(prob<PREP)
    prob = prob[i,j]
    for a,b,p in zip(i,j,prob):
        if b>a:
            name=tmpdir.name+"/%d.%d_rep.txt"%(a+1,b+1)
            rst_line = 'AtomPair %s %d %s %d SCALARWEIGHTEDFUNC %.2f SUMFUNC 2 CONSTANTFUNC 0.5 SIGMOID %.3f %.3f\n'%('CB',a+1,'CB',b+1,-0.5,SIGD,SIGM)
            rst['rep'].append([a,b,p,rst_line])
    print("rep restraints:   %d"%(len(rst['rep'])))
    '''

    ########################################################
    # dist: 0..20A
    ########################################################
    nres = dist.shape[0] # L 
    bins = np.array([4.25 + DSTEP * i for i in range(32)])
    prob = np.sum(dist[:, :, 5:], axis=-1)
    bkgr = np.array((bins / DCUT) ** ALPHA)
    attr = -np.log((dist[:, :, 5:] + MEFF) / (dist[:, :, -1][:, :, None] * bkgr[None, None, :])) + EBASE
    repul = np.maximum(attr[:, :, 0], np.zeros((nres, nres)))[:, :, None] + np.array(EREP)[None, None, :]
    dist = np.concatenate([repul, attr], axis=-1)
    bins = np.concatenate([DREP, bins])
    i, j = np.where(prob > PCUT)
    prob = prob[i, j]
    nbins = 35
    step = 0.5
    for a, b, p in zip(i, j, prob):
        if b > a:
            name = tmpdir.name + "/%d.%d.txt" % (a + 1, b + 1)
            with open(name, "w") as f:
                f.write('x_axis' + '\t%.3f' * nbins % tuple(bins) + '\n')
                f.write('y_axis' + '\t%.3f' * nbins % tuple(dist[a, b]) + '\n')
                f.close()
            rst_line = 'AtomPair %s %d %s %d SPLINE TAG %s 1.0 %.3f %.5f' % ('CB', a + 1, 'CB', b + 1, name, 1.0, step)
            rst['dist'].append([a, b, p, rst_line])
    print("dist restraints:  %d" % (len(rst['dist'])))

    if params['USE_ORIENT'] is True or params['USE_ORIENT'] == "True":
        ########################################################
        # omega: -pi..pi
        ########################################################
        nbins = omega.shape[2] - 1 + 4
        bins = np.linspace(-np.pi - 1.5 * ASTEP, np.pi + 1.5 * ASTEP, nbins)
        prob = np.sum(omega[:, :, 1:], axis=-1)
        i, j = np.where(prob > PCUT)
        prob = prob[i, j]
        omega = -np.log((omega + MEFF) / (omega[:, :, -1] + MEFF)[:, :, None])
        omega = np.concatenate([omega[:, :, -2:], omega[:, :, 1:], omega[:, :, 1:3]], axis=-1)
        for a, b, p in zip(i, j, prob):
            if b > a:
                name = tmpdir.name + "/%d.%d_omega.txt" % (a + 1, b + 1)
                with open(name, "w") as f:
                    f.write('x_axis' + '\t%.5f' * nbins % tuple(bins) + '\n')
                    f.write('y_axis' + '\t%.5f' * nbins % tuple(omega[a, b]) + '\n')
                    f.close()
                rst_line = 'Dihedral CA %d CB %d CB %d CA %d SPLINE TAG %s 1.0 %.3f %.5f' % (a + 1, a + 1, b + 1, b + 1, name, 1.0, ASTEP)
                rst['omega'].append([a, b, p, rst_line])
        print("omega restraints: %d" % (len(rst['omega'])))

        ########################################################
        # theta: -pi..pi
        ########################################################
        prob = np.sum(theta[:, :, 1:], axis=-1)
        i, j = np.where(prob > PCUT)
        prob = prob[i, j]
        theta = -np.log((theta + MEFF) / (theta[:, :, -1] + MEFF)[:, :, None])
        theta = np.concatenate([theta[:, :, -2:], theta[:, :, 1:], theta[:, :, 1:3]], axis=-1)
        for a, b, p in zip(i, j, prob):
            if b != a:
                name = tmpdir.name + "/%d.%d_theta.txt" % (a + 1, b + 1)
                with open(name, "w") as f:
                    f.write('x_axis' + '\t%.3f' * nbins % tuple(bins) + '\n')
                    f.write('y_axis' + '\t%.3f' * nbins % tuple(theta[a, b]) + '\n')
                    f.close()
                rst_line = 'Dihedral N %d CA %d CB %d CB %d SPLINE TAG %s 1.0 %.3f %.5f' % (a + 1, a + 1, a + 1, b + 1, name, 1.0, ASTEP)
                rst['theta'].append([a, b, p, rst_line])
                # if a==0 and b==9:
                #    with open(name,'r') as f:
                #        print(f.read())
        print("theta restraints: %d" % (len(rst['theta'])))

        ########################################################
        # phi: 0..pi
        ########################################################
        nbins = phi.shape[2] - 1 + 4
        bins = np.linspace(-1.5 * ASTEP, np.pi + 1.5 * ASTEP, nbins)
        prob = np.sum(phi[:, :, 1:], axis=-1)
        i, j = np.where(prob > PCUT)
        prob = prob[i, j]
        phi = -np.log((phi + MEFF) / (phi[:, :, -1] + MEFF)[:, :, None])
        phi = np.concatenate([np.flip(phi[:, :, 1:3], axis=-1), phi[:, :, 1:], np.flip(phi[:, :, -2:], axis=-1)], axis=-1)
        for a, b, p in zip(i, j, prob):
            if b != a:
                name = tmpdir.name + "/%d.%d_phi.txt" % (a + 1, b + 1)
                with open(name, "w") as f:
                    f.write('x_axis' + '\t%.3f' * nbins % tuple(bins) + '\n')
                    f.write('y_axis' + '\t%.3f' * nbins % tuple(phi[a, b]) + '\n')
                    f.close()
                rst_line = 'Angle CA %d CB %d CB %d SPLINE TAG %s 1.0 %.3f %.5f' % (a + 1, a + 1, b + 1, name, 1.0, ASTEP)
                rst['phi'].append([a, b, p, rst_line])
                # if a==0 and b==9:
                #    with open(name,'r') as f:
                #        print(f.read())

        print("phi restraints:   %d" % (len(rst['phi'])))

    return rst


def set_random_dihedral(pose):
    nres = pose.total_residue()
    for i in range(1, nres):
        phi, psi = random_dihedral()
        pose.set_phi(i, phi)
        pose.set_psi(i, psi)
        pose.set_omega(i, 180)

    return (pose)


# pick phi/psi randomly from:
# -140  153 180 0.135 B
# -72  145 180 0.155 B
# -122  117 180 0.073 B
# -82  -14 180 0.122 A
# -61  -41 180 0.497 A
#  57   39 180 0.018 L
def random_dihedral():
    phi = 0
    psi = 0
    r = random.random()
    if (r <= 0.135):
        phi = -140
        psi = 153
    elif (r > 0.135 and r <= 0.29):
        phi = -72
        psi = 145
    elif (r > 0.29 and r <= 0.363):
        phi = -122
        psi = 117
    elif (r > 0.363 and r <= 0.485):
        phi = -82
        psi = -14
    elif (r > 0.485 and r <= 0.982):
        phi = -61
        psi = -41
    else:
        phi = 57
        psi = 39
    return (phi, psi)


def remove_clash(scorefxn, mover, pose):
    for _ in range(0, 5):
        if float(scorefxn(pose)) < 10:
            break
        mover.apply(pose)


def add_rst(pose, rst, sep1, sep2, params, nogly=False):
    pcut = params['PCUT']
    seq = params['seq']

    array = []

    if nogly == True:
        array += [line for a, b, p, line in rst['dist'] if abs(a - b) >= sep1 and abs(a - b) < sep2 and seq[a] != 'G' and seq[b] != 'G' and p >= pcut]
        if params['USE_ORIENT'] is True or params['USE_ORIENT'] == "True":
            array += [line for a, b, p, line in rst['omega'] if abs(a - b) >= sep1 and abs(a - b) < sep2 and seq[a] != 'G' and seq[b] != 'G' and p >= pcut + 0.5]  # 0.5
            array += [line for a, b, p, line in rst['theta'] if abs(a - b) >= sep1 and abs(a - b) < sep2 and seq[a] != 'G' and seq[b] != 'G' and p >= pcut + 0.5]  # 0.5
            array += [line for a, b, p, line in rst['phi'] if abs(a - b) >= sep1 and abs(a - b) < sep2 and seq[a] != 'G' and seq[b] != 'G' and p >= pcut + 0.6]  # 0.6
    else:
        array += [line for a, b, p, line in rst['dist'] if abs(a - b) >= sep1 and abs(a - b) < sep2 and p >= pcut]
        if params['USE_ORIENT'] is True or params['USE_ORIENT'] == "True":
            array += [line for a, b, p, line in rst['omega'] if abs(a - b) >= sep1 and abs(a - b) < sep2 and p >= pcut + 0.5]
            array += [line for a, b, p, line in rst['theta'] if abs(a - b) >= sep1 and abs(a - b) < sep2 and p >= pcut + 0.5]
            array += [line for a, b, p, line in rst['phi'] if abs(a - b) >= sep1 and abs(a - b) < sep2 and p >= pcut + 0.6]  # 0.6

    if len(array) < 1:
        return

    random.shuffle(array)

    # save to file
    tmpname = params['TDIR'] + '/minimize.cst'
    with open(tmpname, 'w') as f:
        for line in array:
            f.write(line + '\n')
        f.close()

    # add to pose
    constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
    constraints.constraint_file(tmpname)
    constraints.add_constraints(True)
    constraints.apply(pose)

    os.remove(tmpname)


def cal_cscore(npz):
    prob, sep = top_dist(npz['dist'], 12)
    if os.path.exists("good_temp.txt"):
        cscore = 0.9342 * prob + 0.2333 * sep + 0.0957
    else:
        cscore = 1.158 * prob + 0.1878 * sep + 0.0318
    if cscore > 1:
        cscore = 1.0
    elif cscore < 0.1:
        cscore = 0.1
    with open("cscore.txt", 'w+') as f1:
        print("Estimated TM-score of the top-predicted model: ", round(cscore, 2), file=f1)
    return cscore



def get_conda_pth(name='tRS'):
    for line in os.popen(f'conda info --envs|grep {name}'):
        if line.split()[0] == name:
            return line.split()[1]