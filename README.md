
# lammps_murat

This is a fork of the lammps molecular dynamics code which I modified to my needs. 

I use lammps to simulate a coarse grained DNA model. 

The major modifications lie in 3 files

## bond_harcos.cpp

This is a harmonic-cosine breakable bond

## dihedral_breakable.cpp

This is a dihedral that scales with the middle bond. 

In my dna model this dihedral connects complimentary strands, it dies down as inter-strand bonds are broken. 

## pair_hardcore.cpp

This is a very simple pairwise interaction, it is mainly there to avoid bond crossing. 

It is simply the repulsive half of the harmonic potential. 
