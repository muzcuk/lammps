/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(bond/break,FixBondBreak)

#else

#ifndef LMP_FIX_BOND_BREAK_H
#define LMP_FIX_BOND_BREAK_H

#include "fix.h"

namespace LAMMPS_NS {

class FixBondBreak : public Fix {
 public:
  FixBondBreak(class LAMMPS *, int, char **);
  ~FixBondBreak();
  int setmask();
  void init();
  void post_integrate();
  void post_integrate_respa(int,int);

  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  double compute_vector(int);
  double memory_usage();

 private:
  int me,nprocs;
  int btype,seed;
  double cutoff,cutsq;
  bigint lastcheck;

  int breakcount,breakcounttotal;
  int nmax;
  tagint *partner;

  int nbreak,maxbreak;
  tagint **broken;

  tagint *copy;

  class RanMars *random;
  int nlevels_respa;

  int commflag;
  int nbroken;
  int nangles,ndihedrals,nimpropers;

  void check_ghosts();
  void update_topology();
  void break_dihedrals(int, tagint, tagint);
  void rebuild_special(int);
  int dedup(int, int, tagint *);
  
  // DEBUG

  void print_bb();
  void print_copy(const char *, tagint, int, int, int, int *);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Invalid bond type in fix bond/break command

Self-explanatory.

E: Cannot use fix bond/break with non-molecular systems

Only systems with bonds that can be changed can be used.  Atom_style
template does not qualify.

E: Fix bond/break requires special_bonds = 0,1,1

This is a restriction of the current fix bond/break implementation.

W: Broken bonds will not alter angles, dihedrals, or impropers

See the doc page for fix bond/break for more info on this
restriction.

*/
