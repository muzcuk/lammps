/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "math.h"
#include "mpi.h"
#include "string.h"
#include "stdlib.h"
#include "fix_bond_update.h"
#include "update.h"
#include "respa.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "domain.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define DELTA 16

/* ---------------------------------------------------------------------- */

FixBondUpdate::FixBondUpdate(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg != 6 ) error->all(FLERR,"Illegal fix bond/break command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  nevery = force->inumeric(FLERR,arg[3]);
  if (nevery <= 0) error->all(FLERR,"Illegal fix bond/break command");

  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;

  btype = force->inumeric(FLERR,arg[4]);
  cutoff = force->numeric(FLERR,arg[5]);

  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix bond/break command");
  if (cutoff < 0.0) error->all(FLERR,"Illegal fix bond/break command");

  cutsq = cutoff*cutoff;

  // error check

  if (atom->molecular != 1)
    error->all(FLERR,"Cannot use fix bond/break with non-molecular systems");

  // set comm sizes needed by this fix
  // forward is 4 because nspecial (1-2) + maximum of 3 (1-2) neighbors

  //comm_forward = 4;
  //comm_reverse = 2;

  // allocate arrays local to this fix

  nmax = 0;
  partner = NULL;

  maxbreak = 0;
  broken = NULL;

  // copy = special list for one atom
  // size = ms^2 + ms is sufficient
  // b/c in rebuild_special() neighs of all 1-2s are added,
  //   then a dedup(), then neighs of all 1-3s are added, then final dedup()
  // this means intermediate size cannot exceed ms^2 + ms

  int maxspecial = atom->maxspecial;
  copy = new tagint[maxspecial*maxspecial + maxspecial];

  // zero out stats

  breakcount = 0;
  breakcounttotal = 0;
}

/* ---------------------------------------------------------------------- */

FixBondUpdate::~FixBondUpdate()
{
  delete random;

  // delete locally stored arrays

  memory->destroy(partner);
  memory->destroy(broken);
  delete [] copy;
}

/* ---------------------------------------------------------------------- */

int FixBondUpdate::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBondUpdate::init()
{
  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;

  // we only have dihedrals to break after manipulating bonds
  // nothing to do with angles and impropers

  lastcheck = -1;

  // DEBUG
  //print_bb();
}

/* ---------------------------------------------------------------------- */

void FixBondUpdate::post_integrate()
{
  int i,j,k,m,n,i1,i2,n1,n3,type;
  double delx,dely,delz,rsq;
  tagint *slist;

  if (update->ntimestep % nevery) return;

  // check that all procs have needed ghost atoms within ghost cutoff
  // only if neighbor list has changed since last check

  // necessary b/c are calling this after integrate, but before Verlet comm

  comm->forward_comm();

  // resize bond partner list and initialize it
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(partner);
    nmax = atom->nmax;
    memory->create(partner,nmax,"bond/break:partner");
  }

  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;

  for (i = 0; i < nall; i++) {
    partner[i] = 0;
  }

  // loop over bond list
  // setup list of bonds to break

  double **x = atom->x;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;


 
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  nbreak = 0;
  int ji;

  for (i = 0; i < nlocal; i++) {
	
    // delete bond from atom I if I stores it
    // atom J will also do this

    for (m = 0; m < num_bond[i]; m++) {
	    if (!(mask[i] & groupbit)) continue;
	    
	    if (abs(bond_type[i][m]) != btype) continue;
		
	    // j is the tag of partner atom
	    
	    j = bond_atom[i][m]; 
	    ji = atom->map(j);
	    
	    if (!(mask[ji] & groupbit)) continue;

	    //check if we have j in this domain
	    
	    if (ji == -1) continue; 
	    
	    delx = x[i][0] - x[ji][0];
	    dely = x[i][1] - x[ji][1];
	    delz = x[i][2] - x[ji][2];

	    rsq = delx*delx + dely*dely + delz*delz;

	    if (bond_type[i][m] == btype) {
		    
		    if (rsq <= cutsq)    continue;

		    disable_bond(i,m);
		    continue;
	    }
	    
	    if (bond_type[i][m] == -btype) {
		    
		    if (rsq > cutsq)	continue;

		    enable_bond (i,m);
		    continue;
	    }
        }
  }

  // tally stats

  MPI_Allreduce(&nbreak,&breakcount,1,MPI_INT,MPI_SUM,world);
  breakcounttotal += breakcount;
  atom->nbonds -= breakcount;

  // trigger reneighboring if any bonds were broken
  // this insures neigh lists will immediately reflect the topology changes
  // done if no bonds broken

  if (breakcount) next_reneighbor = update->ntimestep;
  if (!breakcount) return;

  // DEBUG
  // print_bb();
}

/*------------
disable the Bth bond and all dihedrals owned by atom M
*/

void FixBondUpdate::disable_bond(int m, int b) {
	
	int i;
	tagint partner;

	int *bond_type = atom->bond_type[m];
  	tagint *bond_atom = atom->bond_atom[m];
	
	int num_dihedral = atom->num_dihedral[m];
	int *dihedral_type = atom->dihedral_type[m];

	int *nspecial = atom->nspecial[m];
	tagint *special = atom->special[m];

 
	// disable the jth bond
	bond_type[b] = -btype;
	
	// disable all dihedrals
	for (i = 0; i < num_dihedral; i++) 
		dihedral_type[i] = -abs(dihedral_type[i]);


	// remove partner from 1-2 special list
	
	partner = bond_atom[b];
	
	for (i=0; i< nspecial[0]; i++)
		if (special[i] == partner) 
			break;

	for ( ; i<nspecial[2]-1; i++)
		special[i]=special[i+1];

	nspecial[0]--;
	nspecial[1]--;
	nspecial[2]--;
}

/*------------
disable the Bth bond and all dihedrals owned by atom M
*/

void FixBondUpdate::enable_bond(int m, int b) {
	
	int i;
	tagint partner;

	int *bond_type = atom->bond_type[m];
  	tagint *bond_atom = atom->bond_atom[m];
	
	int num_dihedral = atom->num_dihedral[m];
	int *dihedral_type = atom->dihedral_type[m];

	int *nspecial = atom->nspecial[m];
	tagint *special = atom->special[m];

 
	// enable the jth bond
	bond_type[b] = btype;
	
	// enable all dihedrals
	for (i = 0; i < num_dihedral; i++) 
		dihedral_type[i] = abs(dihedral_type[i]);


	// add partner to 1-2 special list
	
	partner = bond_atom[b];

	for ( i = nspecial[0]; i<nspecial[2]; i++)
		special[i+1]=special[i];

	special[nspecial[0]]=partner;

	nspecial[0]++;
	nspecial[1]++;
	nspecial[2]++;
}



//***---------------------------------------------------------------------

double FixBondUpdate::compute_vector(int n)
{
  if (n == 1) return (double) breakcount;
  return (double) breakcounttotal;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixBondUpdate::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = 2*nmax * sizeof(tagint);
  bytes += nmax * sizeof(double);
  return bytes;
}
