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
#include "fix_bond_break.h"
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

FixBondBreak::FixBondBreak(LAMMPS *lmp, int narg, char **arg) :
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

  comm_forward = 4;
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

FixBondBreak::~FixBondBreak()
{
  delete random;

  // delete locally stored arrays

  memory->destroy(partner);
  memory->destroy(broken);
  delete [] copy;
}

/* ---------------------------------------------------------------------- */

int FixBondBreak::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBondBreak::init()
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

void FixBondBreak::post_integrate()
{
  int i,j,k,m,n,i1,i2,n1,n3,type;
  double delx,dely,delz,rsq;
  tagint *slist;

  if (update->ntimestep % nevery) return;

  // check that all procs have needed ghost atoms within ghost cutoff
  // only if neighbor list has changed since last check

  if (lastcheck < neighbor->lastcall) check_ghosts();

  // acquire updated ghost atom positions
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

  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];
    if (!(mask[i1] & groupbit)) continue;
    if (!(mask[i2] & groupbit)) continue;
    if (type != btype) continue;

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];
    rsq = delx*delx + dely*dely + delz*delz;
    
    if (rsq <= cutsq) continue;

    partner[i1] = tag[i2];
    partner[i2] = tag[i1];

  }

  commflag = 1;
  comm->forward_comm_fix(this,2);

  // break bonds

  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  nbreak = 0;
  for (i = 0; i < nlocal; i++) {
    if (partner[i] == 0) continue;
    j = atom->map(partner[i]);

    // this is more of a sanity check now, can be disabled for performance
    if (partner[j] != tag[i]) continue; 

    // delete bond from atom I if I stores it
    // atom J will also do this

    for (m = 0; m < num_bond[i]; m++) {
	
      //if bound harcos
      if (bond_type[i][m] == btype) { 
       
	  // break the bond
	  if (rsq >= cutsq)  
	          bond_type[i][m] *= -1;
        }

      }
      // broken harcos
      else if (bond_type[i][m] == -btype) {

	      //reconnect the bond
	      if ( rsq < cutsq )
		  bond_type[i][m] *=-1;
      }		  

    }

    // remove J from special bond list for atom I
    // atom J will also do this, whatever proc it is on

    slist = special[i];
    n1 = nspecial[i][0];
    for (m = 0; m < n1; m++)
      if (slist[m] == partner[i]) break;
    n3 = nspecial[i][2];
    for (; m < n3-1; m++) slist[m] = slist[m+1];
    nspecial[i][0]--;
    nspecial[i][1]--;
    nspecial[i][2]--;

    // store final broken bond partners and count the broken bond once

    if (tag[i] < tag[j]) nbreak++;
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
flip the Nth bond and all dihedrals owned by atom M
*/

void FiBondBreak::disable_bond(int m, int b) {
	
	int i;

	int *bond_type = atom->bond_type[m];
	
	int num_dihedral = atom->num_dihedral[m];
	int *dihedral_type = atom->dihedral_type[m];
 
	// flip the jth bond
	bond_type[n] *= -btype;
	
	// flip all dihedrals
	for(i = 0; i < num_dihedral; i++) 
		dihedral_type[i] *= -1;
     
}


//***---------------------------------------------------------------------

double FixBondBreak::compute_vector(int n)
{
  if (n == 1) return (double) breakcount;
  return (double) breakcounttotal;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixBondBreak::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = 2*nmax * sizeof(tagint);
  bytes += nmax * sizeof(double);
  return bytes;
}
