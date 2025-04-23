from rosetta_min.utils import *
from pyrosetta import *
from pyrosetta.rosetta.protocols.minimization_packing import MinMover

vdw_weight = {0: 3.0, 1: 5.0, 2: 10.0}
rsr_dist_weight = {0: 3.0, 1: 2.0, 3: 1.0}
rsr_orient_weight = {0: 1.0, 1: 1.0, 3: 0.5}

def run_minimization(
        npz,
        seq,
        scriptdir,
        outPath,
        pose=None,
        angle_std=10,
        dist_std=2,
        use_fastdesign=False,
        use_fastrelax=True,
):
    use_fastdesign = False
    L = len(seq)
    rst = load_constraints(npz,angle_std,dist_std)
    scorefxn = create_score_function("ref2015")
    e = 999999
    # make output directory
    outPath.mkdir(exist_ok=True,parents=True)

    sf = ScoreFunction()
    sf.add_weights_from_file(str(scriptdir.joinpath('data/scorefxn.wts')))

    sf1 = ScoreFunction()
    sf1.add_weights_from_file(str(scriptdir.joinpath('data/scorefxn1.wts')))

    sf_vdw = ScoreFunction()
    sf_vdw.add_weights_from_file(str(scriptdir.joinpath('data/scorefxn_vdw.wts')))

    sf_cart = ScoreFunction()
    sf_cart.add_weights_from_file(str(scriptdir.joinpath('data/scorefxn_cart.wts')))

    mmap = MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(False)
    mmap.set_jump(True)

    min_mover1 = MinMover(mmap, sf1, 'lbfgs_armijo_nonmonotone', 0.001, True)
    min_mover1.max_iter(1000)

    min_mover_vdw = MinMover(mmap, sf_vdw, 'lbfgs_armijo_nonmonotone', 0.001, True)
    min_mover_vdw.max_iter(500)

    min_mover_cart = MinMover(mmap, sf_cart, 'lbfgs_armijo_nonmonotone', 0.000001, True)
    min_mover_cart.max_iter(300)
    min_mover_cart.cartesian(True)

    ########################################################
    # backbone minimization
    ########################################################
    if pose is None:
        pose0 = pose_from_sequence(seq, 'centroid')
        set_random_dihedral(pose0)
        remove_clash(sf_vdw, min_mover_vdw, pose0)
        
        # tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
        # tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
        # tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
        
        indices_to_design = None
        
    else:
        pose0 = pose
        to_centroid = SwitchResidueTypeSetMover('centroid')
        to_centroid.apply(pose0)
        # indices_to_design = [str(i+1) for i,c in enumerate(seq) if c == "_"]
        # prevent_repack = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT() # No repack, no design
        # masked_residues = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(",".join(indices_to_design))
        # unmasked_residues = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(masked_residues)

        # tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
        # tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
        # tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
        # tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(prevent_repack, unmasked_residues))
        # MoveMap
        mm = pyrosetta.rosetta.core.kinematics.MoveMap()
        # mm.set_bb_true_range(int(indices_to_design[0]),int(indices_to_design[-1]))
        min_mover1.set_movemap(mm)
        min_mover_vdw.set_movemap(mm)
        min_mover_cart.set_movemap(mm)
        
    Emin = 999999
    indices_to_design = None
    # 5次最小化，每次重复3次
    for run in range(5):
        # define repeat_mover here!! (update vdw weights: weak (1.0) -> strong (10.0)
        sf.set_weight(rosetta.core.scoring.vdw, vdw_weight.setdefault(run, 10.0))
        sf.set_weight(rosetta.core.scoring.atom_pair_constraint, rsr_dist_weight.setdefault(run, 1.0))
        sf.set_weight(rosetta.core.scoring.dihedral_constraint, rsr_orient_weight.setdefault(run, 0.5))
        sf.set_weight(rosetta.core.scoring.angle_constraint, rsr_orient_weight.setdefault(run, 0.5))
        
        
        min_mover = MinMover(mmap, sf, 'lbfgs_armijo_nonmonotone', 0.001, True)
        min_mover.max_iter(1000)
        
        # if indices_to_design:
        #     min_mover.set_movemap(mm)
           
        repeat_mover = RepeatMover(min_mover, 3)

        pose = Pose()
        pose.assign(pose0)
        pose.remove_constraints()

        if run > 0:
            
            # diversify backbone
            dphi = np.random.uniform(-10,10,L)
            dpsi = np.random.uniform(-10,10,L)
            
            if indices_to_design:
                for i in indices_to_design:
                    i = int(i)
                    pose.set_phi(i,pose.phi(i)+dphi[i-1])
                    pose.set_psi(i,pose.psi(i)+dpsi[i-1])
            else:
                for i in range(1,L+1):
                    pose.set_phi(i,pose.phi(i)+dphi[i-1])
                    pose.set_psi(i,pose.psi(i)+dpsi[i-1])

            # remove clashes
            remove_clash(sf_vdw, min_mover_vdw, pose)

        # short
        add_rst(pose, rst, 3, 12)
        repeat_mover.apply(pose)
        remove_clash(sf_vdw, min_mover1, pose)
        min_mover_cart.apply(pose)

        # medium
        add_rst(pose, rst, 12, 24)
        repeat_mover.apply(pose)
        remove_clash(sf_vdw, min_mover1, pose)
        min_mover_cart.apply(pose)

        # long
        add_rst(pose, rst, 24, len(seq))
        repeat_mover.apply(pose)
        remove_clash(sf_vdw, min_mover1, pose)
        min_mover_cart.apply(pose)

        # check whether energy has decreased
        E = sf_cart(pose)
        if E < Emin:
            Emin = E
            pose0.assign(pose)

    pose0.remove_constraints()
    pose0.dump_pdb(str(outPath.joinpath("structure_before_design.pdb")))

    if use_fastdesign:
        ############################
        ## sidechain minimization ##
        ############################
        # Convert to all atom representation
        switch = SwitchResidueTypeSetMover("fa_standard")
        switch.apply(pose0)

        # MoveMap
        mm = pyrosetta.rosetta.core.kinematics.MoveMap()
        mm.set_bb(False)
        mm.set_chi(True)

        scorefxn = pyrosetta.create_score_function("ref2015_cart")
        rel_design = pyrosetta.rosetta.protocols.relax.FastRelax()
        rel_design.set_scorefxn(scorefxn)
        rel_design.set_task_factory(tf)
        rel_design.set_movemap(mm)
        rel_design.cartesian(True)
        rel_design.apply(pose0)

        pose0.dump_pdb(str(outPath.joinpath("structure_after_design.pdb")))

    if use_fastrelax:
        ########################################################
        # full-atom refinement
        ########################################################
        mmap = MoveMap()
        mmap.set_bb(True)
        mmap.set_chi(True)
        mmap.set_jump(True)

        # First round: Repeat 2 torsion space relax w/ strong disto/anglogram constraints
        sf_fa_round1 = create_score_function('ref2015_cart')
        sf_fa_round1.set_weight(rosetta.core.scoring.atom_pair_constraint, 3.0)
        sf_fa_round1.set_weight(rosetta.core.scoring.dihedral_constraint, 1.0)
        sf_fa_round1.set_weight(rosetta.core.scoring.angle_constraint, 1.0)
        sf_fa_round1.set_weight(rosetta.core.scoring.pro_close, 0.0)

        relax_round1 = rosetta.protocols.relax.FastRelax(sf_fa_round1, "%s/data/relax_round1.txt"%scriptdir)
        relax_round1.set_movemap(mmap)
        # relax_round1.set_task_factory(tf)
        relax_round1.dualspace(True)
        relax_round1.minimize_bond_angles(True)

        pose0.remove_constraints()
        add_rst(pose0, rst, 3, len(seq))
        try:
            relax_round1.apply(pose0)
        except:
            print("Failed full-atom refinement")

        # Set options for disulfide tolerance -> 0.5A
        rosetta.basic.options.set_real_option('in:detect_disulf_tolerance', 0.5)

        sf_fa = create_score_function('ref2015_cart')
        sf_fa.set_weight(rosetta.core.scoring.atom_pair_constraint, 0.1)
        sf_fa.set_weight(rosetta.core.scoring.dihedral_constraint, 0.0)
        sf_fa.set_weight(rosetta.core.scoring.angle_constraint, 0.0)

        relax_round2 = rosetta.protocols.relax.FastRelax(sf_fa, "%s/data/relax_round2.txt"%scriptdir)
        relax_round2.set_movemap(mmap)
        # relax_round2.set_task_factory(tf)
        relax_round2.minimize_bond_angles(True)
        relax_round2.cartesian(True)
        relax_round2.dualspace(True)

        pose0.remove_constraints()
        pose0.conformation().detect_disulfides() # detect disulfide bond again w/ stricter cutoffs
        # To reduce the number of constraints, only pair distances are considered w/ higher prob cutoffs
        add_rst(pose0, rst, 3, len(seq))
        # Instead, apply CA coordinate constraints to prevent drifting away too much (focus on local refinement?)
        add_crd_rst(pose0, L, std=1.0, tol=2.0)
        relax_round2.apply(pose0)

        pose0.dump_pdb(str(outPath.joinpath("final_structure.pdb")))