'''
Parsers
'''
from rdkit import Chem
import numpy as np
import torch
import re
import os


def get_affinities(affinity_directory):
    affinity_dict = {}
    with open(affinity_directory, 'r', encoding='utf-8') as f:
        for line in f:
            aff_not_uncertain = True
            if line[0] != '#':
                fields = line.split()
                pdb_id = fields[0]
                log_aff = float(fields[3])
                aff_str = fields[4]
                if '<' in aff_str or '>' in aff_str or '~' in aff_str:
                    aff_not_uncertain = False
                aff_tokens = re.split('[=<>~]+', aff_str)
                assert len(aff_tokens) == 2
                label, aff_and_unity = aff_tokens
                assert label in ['Kd', 'Ki', 'IC50']
                affinity_value = float(aff_and_unity[:-2])
                aff_unity = aff_and_unity[-2:]
                aff = float(affinity_value)
                affinity_dict[pdb_id] = [
                    label, log_aff, aff, aff_unity, aff_not_uncertain
                ]
    return affinity_dict


def read_dataset(directory, ligand_file_extention, protein_file_extention,
                 aff_dict):
    '''
    from directory returns a list of pdb_id, path to protein, path to ligand
    The directory contains compound folders (each has an ID with 4 letters,
    ex. abcd) with 4 files:
    - abcd_protein.pdb
    - abcd_pocket.pdb
    - abcd_ligand.sdf
    - abcd_ligand.mol2
    '''
    assert ligand_file_extention in ('sdf', 'mol2')
    assert protein_file_extention in ('protein', 'pocket', 'processed')
    molecules_files = []
    for folder_name in os.listdir(directory):
        if len(folder_name) == 4:
            folder_dir = os.path.join(directory, folder_name)
            files = os.listdir(folder_dir)
            compound_id = folder_name

            for file in files:
                if file.endswith(protein_file_extention + '.pdb'):
                    file_protein = file
                elif file.endswith('ligand.' + ligand_file_extention):
                    file_ligand = file
            if aff_dict[compound_id][4]:
                # only add molecule if affinity is not uncertain
                molecules_files += [
                    (compound_id, os.path.join(folder_dir, file_protein),
                     os.path.join(folder_dir,
                                  file_ligand), aff_dict[compound_id][1])
                ]
    return molecules_files


ele2num = {
    'H': 0,
    'O': 1,
    'N': 2,
    'C': 3,
    'S': 4,
    'Se': 5,
    'P': 6,
    'F': 7,
    'Cl': 7,
    'Br': 7,
    'I': 7,
    'Mg': 8,
    'Ca': 8,
    'Sr': 8,
    'Na': 8,
    'K': 8,
    'Cs': 8,
    'Mn': 8,
    'Fe': 8,
    'Co': 8,
    'Ni': 8,
    'Cu': 8,
    'Zn': 8,
    'Cd': 8,
    'Hg': 8
}

num_atom_types = max(ele2num.values()) + 1

pt = Chem.GetPeriodicTable()


def molecule_info(path, type_mol, num_atoms_ligand):
    '''from path returns the coordinates, atoms and
    bonds of molecule'''

    if type_mol == 'Protein':
        molecule = Chem.MolFromPDBFile(path,
                                       flavor=2,
                                       sanitize=False,
                                       removeHs=False)

    elif type_mol == 'Ligand':

        if path[-4:] == '.sdf':
            suplier = Chem.SDMolSupplier(path, sanitize=False, removeHs=False)
            molecule = next(suplier)
        elif path[-4:] == 'mol2':
            molecule = Chem.MolFromMol2File(path,
                                            sanitize=False,
                                            removeHs=False)

    atom_features = []
    conformer = molecule.GetConformer(0)
    num_atoms = molecule.GetNumAtoms()

    if type_mol == 'Ligand':
        first_elem = [1]
    else:
        first_elem = [0]

    for atom in molecule.GetAtoms():
        atom_symbol = atom.GetSymbol()
        onehot_elem = np.zeros(num_atom_types)
        onehot_elem[ele2num.get(atom_symbol, 8)] = 1

        onehot_total_valence = np.zeros(9)
        onehot_total_valence[atom.GetTotalValence()] = 1

        onehot_explicit_valence = np.zeros(9)
        onehot_explicit_valence[atom.GetExplicitValence()] = 1

        onehot_implicit_valence = np.zeros(5)
        onehot_implicit_valence[atom.GetImplicitValence()] = 1

        van_der_waals_radius = pt.GetRvdw(atom_symbol)
        covalent_radius = pt.GetRcovalent(atom_symbol)

        atom_features += [[
            *first_elem, *onehot_elem, *onehot_total_valence,
            *onehot_explicit_valence, *onehot_implicit_valence, *[
                atom.GetFormalCharge(),
                atom.IsInRing(), van_der_waals_radius, covalent_radius
            ]
        ]]

    atom_features = torch.as_tensor(atom_features)

    coords = conformer.GetPositions()

    rows_l = []
    cols_l = []

    edges_features = []
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        bond_in_ring = bond.IsInRing()
        rows_l += [i + num_atoms_ligand, j + num_atoms_ligand]
        cols_l += [j + num_atoms_ligand, i + num_atoms_ligand]
        length = np.linalg.norm(coords[i] - coords[j])

        vector_ij = list((coords[j] - coords[i]) / length)
        vector_ji = list((coords[i] - coords[j]) / length)

        edges_features += [[*vector_ij, length, bond_type, bond_in_ring],
                           [*vector_ji, length, bond_type, bond_in_ring]]

    edges = torch.as_tensor([rows_l, cols_l])

    edges_features = torch.as_tensor(edges_features)

    return (coords, atom_features, edges, edges_features, num_atoms)


CASF_2016_core_set = {
    '3arq', '4cig', '3ehy', '2vw5', '1gpk', '2x00', '4eor', '4ivb', '4jxs',
    '3oe5', '2wbg', '4bkt', '4e5w', '4x6p', '1r5y', '3e92', '1ydt', '1h23',
    '3ui7', '2fvd', '4w9h', '1vso', '4gr0', '2cet', '3qgy', '2wn9', '3dx2',
    '3ebp', '4de2', '4cr9', '2al5', '1q8u', '3f3e', '4j3l', '3uri', '4kzq',
    '3uew', '4mme', '3gbb', '3zt2', '3nq9', '5a7b', '3udh', '2cbv', '3ozt',
    '4agp', '4w9c', '4owm', '3wtj', '2zb1', '3u8k', '4j21', '3uo4', '4ddh',
    '3rlr', '3zso', '4dld', '4ddk', '2c3i', '3coy', '2vvn', '2xbv', '2brb',
    '2wer', '3gr2', '3twp', '3d6q', '3r88', '4hge', '4ogj', '3n7a', '1p1n',
    '2j78', '3tsk', '2qbq', '4ciw', '4eo8', '2ymd', '4gkm', '1a30', '2zda',
    '3g2z', '1o5b', '3k5v', '2wnc', '1owh', '3arv', '3aru', '3n86', '3g2n',
    '4de3', '1o3f', '2zy1', '4j28', '2p15', '1mq6', '1k1i', '3dx1', '3syr',
    '2fxs', '2qnq', '2y5h', '2xii', '3kr8', '4jia', '3ao4', '1q8t', '3ag9',
    '3bgz', '3coz', '3kgp', '1qkt', '4crc', '4djv', '2xj7', '3g31', '4gid',
    '4u4s', '4kzu', '3arp', '3dxg', '4jfs', '3mss', '4wiv', '2yge', '4agq',
    '4f3c', '4w9i', '3wz8', '3l7b', '3lka', '2yfe', '4k77', '1w4o', '3u5j',
    '3qqs', '3gv9', '1qf1', '2xnb', '1eby', '4qac', '3jya', '3e5a', '2v00',
    '2wvt', '2iwx', '1u1b', '2xb8', '3nx7', '1oyt', '4gfm', '4mgd', '4cra',
    '4qd6', '1z6e', '2v7a', '2pog', '3gy4', '2zcq', '3fv1', '3kwa', '3ozs',
    '4ih7', '5aba', '1uto', '3g0w', '3nw9', '3utu', '1g2k', '3f3a', '3ary',
    '3fur', '3gc5', '1h22', '3p5o', '4f9w', '1o0h', '4ivc', '4rfm', '3uex',
    '1ps3', '3d4z', '4pcs', '1lpg', '4jsz', '3oe4', '3ryj', '1nc1', '4m0z',
    '4e6q', '2qe4', '3jvs', '1z9g', '3cj4', '3pxf', '3b65', '3uuo', '4eky',
    '2qbr', '2xys', '4ty7', '3ejr', '1c5z', '2zcr', '1ydr', '3zsx', '3o9i',
    '2w66', '3zdg', '3u8n', '4w9l', '2xdl', '3b68', '3gnw', '3myg', '5dwr',
    '1bcu', '3up2', '2r9w', '1yc1', '3ivg', '4dli', '1z95', '4f2w', '5tmn',
    '3b27', '3dd0', '4de1', '2wca', '3bv9', '4llx', '4f09', '4abg', '3prs',
    '1e66', '4lzs', '3jvr', '3fv2', '1p1q', '1s38', '3pyy', '3u9q', '1sqa',
    '4k18', '3uev', '4m0y', '2br1', '3rsx', '1gpn', '1y6r', '1nc3', '3b1m',
    '2vkm', '3acw', '1nvq', '3b5r', '3ueu', '4tmn', '2p4y', '4twp', '5c2h',
    '3pww', '1syi', '3rr4', '1bzc', '3f3d', '4ih5', '2w4x', '4kz6', '2hb1',
    '2wtv', '2j7h', '2yki', '4agn', '5c28', '3ge7', '4ivd', '3f3c', '3e93',
    '1pxn', '2qbp', '4ea2', '2weg', '3n76', '3fcq'
}

files_with_error = {
    '3p3h', '3p3j', '3p44', '3p55', '4i60', '4jfv', '4jfw', '4jhq', '4um9',
    '2vnp', '3vjs', '2z3z', '2zjw', '3bwf', '2rib', '2jdk', '3pup', '3dcq',
    '2c5y', '1ppw', '4avt', '3wax', '3zp9', '3egk', '2fov', '2q2n', '4c4n',
    '1qon', '4ie3', '3cst', '5a3o', '4ixv', '3wc5', '4hmq', '1rdn', '4z7n',
    '4rlp', '2eep', '4gql', '4m3b', '2foy', '2jdu', '1h07', '2a5b', '2wfg',
    '2jdp', '2vr0', '2ci9', '4hxq', '3vjt', '4daw', '3lp1', '2a3w', '1epq',
    '3v9b', '4hww', '3eyd', '2cfd', '2w08', '4hze', '3whw', '1r1h', '3e9b',
    '4fxz', '4ayu', '3lil', '1ai6', '4z7s', '1sl3', '1esz', '3kck', '4z7f',
    '2z97', '2fou', '2nwn', '2jsd', '2aoh', '2cfg', '2boj', '2boi', '1biw',
    '3gpe', '3e6k', '4mma', '2jdy', '1a7x', '2g83', '4avs', '4kw6', '1z3j',
    '4kcx', '2ork', '1mue', '2brh', '2bv4', '4ob1', '3q4c', '1ksn', '2pll',
    '3kqr', '4fil', '4x1r', '3lik', '3qlb', '1hyz', '2jdn', '2ggx', '4ob0',
    '4m3f', '3udn', '4ixu', '4ob2', '2wl4', '2jdm', '1bm6', '3w8o', '4kai',
    '2ria', '2jdh', '2yak', '4kb7', '1q54', '4mm4', '2os9', '2fm5', '2ggu',
    '4abd', '3lp2', '4u0x', '4no1', '4aoc', '1cps', '4mm6', '4dcx', '3lir',
    '3zdv', '1k2v', '4kbi', '3rj7', '4wkv', '4lv1', '1rdl', '3h9f', '1nu1',
    '4z7q', '4wku', '1rdi', '3m1s', '2jbl', '1bcj', '4wkt', '3zju', '3l2y',
    '4wk2', '2fuu', '4iu0', '4l6q', '3zjt', '1f92', '3fxz', '3bho', '3fy0',
    '1qpf', '1rdj', '4ie2', '4i06', '1g7v'
}
