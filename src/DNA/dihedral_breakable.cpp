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

/* ----------------------------------------------------------------------
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include "lmptype.h"
#include "mpi.h"
#include "math.h"
#include "stdlib.h"
#include "dihedral_breakable.h"
#include "atom.h"
#include "comm.h"
#include "neighbor.h"
#include "domain.h"
#include "force.h"
#include "update.h"
#include "memory.h"
#include "error.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace MathConst;


#define TOLERANCE 0.05
#define SMALL     0.001

/* ---------------------------------------------------------------------- */

DihedralBreakable::DihedralBreakable(LAMMPS *lmp) : Dihedral(lmp)
{
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

DihedralBreakable::~DihedralBreakable()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(cos_shift);
    memory->destroy(sin_shift);
    
    //new variables for breakable bond
    memory->destroy(shift);
    memory->destroy(rEq);
    memory->destroy(rCut);
    memory->destroy(vShift);
  }
}

/* ---------------------------------------------------------------------- */

void DihedralBreakable::compute(int eflag, int vflag)
{
  int i1,i2,i3,i4,n,type;
  double vb1x,vb1y,vb1z,vb2x,vb2y,vb2z,vb3x,vb3y,vb3z,vb2xm,vb2ym,vb2zm;
  double edihedral,f1[3],f2[3],f3[3],f4[3];
  double ax,ay,az,bx,by,bz,rasq,rbsq,rg,rginv,ra2inv,rb2inv,rabinv;
  double df,df1,fg,hg,fga,hgb,gaa,gbb;
  double dtfx,dtfy,dtfz,dtgx,dtgy,dtgz,dthx,dthy,dthz;
  double c,s,p,sx2,sy2,sz2;
  double kMult, dkMult;

  edihedral = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double **x = atom->x;
  double **f = atom->f;
  int **dihedrallist = neighbor->dihedrallist;
  int ndihedrallist = neighbor->ndihedrallist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < ndihedrallist; n++) {
    i1 = dihedrallist[n][0];
    i2 = dihedrallist[n][1];
    i3 = dihedrallist[n][2];
    i4 = dihedrallist[n][3];
    type = dihedrallist[n][4];

    // 1st bond

    vb1x = x[i1][0] - x[i2][0];
    vb1y = x[i1][1] - x[i2][1];
    vb1z = x[i1][2] - x[i2][2];

    // 2nd bond

    vb2x = x[i3][0] - x[i2][0];
    vb2y = x[i3][1] - x[i2][1];
    vb2z = x[i3][2] - x[i2][2];

    vb2xm = -vb2x;
    vb2ym = -vb2y;
    vb2zm = -vb2z;

    //calculate length of middle bond (vb2)     
    rg = sqrt(vb2x*vb2x + vb2y*vb2y + vb2z*vb2z) ;
    kMult = 1;
    dkMult = 0;
    
    //scale the coefficient if bond is further than equilibrium distance
    if (rg > rEq[type] && rg < rCut[type]) {
	rasq = MY_PI / (rCut[type]-rEq[type]); 	//scaling coefficient (use rasq temporarily for efficiency)
	rbsq = rg - rEq[type]; 			//rdistance from r0   (use rbsq temporarily for efficiency)
   	kMult  = 0.5 * ( 1 + cos(rbsq*rasq)) ;
	dkMult = -0.5*rasq * sin(rbsq*rasq)  ;
	//printf("Scaled dihedral due to bond btw %d and %d with r=%g, kScaled=%g\n" ,i2, i3, rMid,kScaled);
	//printf("Scaled dihedral r=%.2f, kMult=%.2f, dkMult=%.2f\n" ,rg,kMult,dkMult);
    }

    //skip this dihedral if bond is broken
    else if (rg >= rCut[type] )  {
        //printf("Broken bond btw %d and %d with r=  %g\n" ,i2, i3, rg);
        //printf("Broken bond with r=%.2f\n" , rg);
        continue;   
    }

    // 3rd bond

    vb3x = x[i4][0] - x[i3][0];
    vb3y = x[i4][1] - x[i3][1];
    vb3z = x[i4][2] - x[i3][2];

    // c,s calculation

    ax = vb1y*vb2zm - vb1z*vb2ym;
    ay = vb1z*vb2xm - vb1x*vb2zm;
    az = vb1x*vb2ym - vb1y*vb2xm;
    bx = vb3y*vb2zm - vb3z*vb2ym;
    by = vb3z*vb2xm - vb3x*vb2zm;
    bz = vb3x*vb2ym - vb3y*vb2xm;

    rasq = ax*ax + ay*ay + az*az;
    rbsq = bx*bx + by*by + bz*bz;

    rginv = ra2inv = rb2inv = 0.0;
    if (rg > 0) rginv = 1.0/rg;
    if (rasq > 0) ra2inv = 1.0/rasq;
    if (rbsq > 0) rb2inv = 1.0/rbsq;
    rabinv = sqrt(ra2inv*rb2inv);

    c = (ax*bx + ay*by + az*bz)*rabinv;
    s = rg*rabinv*(ax*vb3x + ay*vb3y + az*vb3z);

    // error check

    if (c > 1.0 + TOLERANCE || c < (-1.0 - TOLERANCE)) {
      int me;
      MPI_Comm_rank(world,&me);
      if (screen) {
        char str[128];
        sprintf(str,"Dihedral problem: %d " BIGINT_FORMAT " " 
                TAGINT_FORMAT " " TAGINT_FORMAT " " 
                TAGINT_FORMAT " " TAGINT_FORMAT,
                me,update->ntimestep,
                atom->tag[i1],atom->tag[i2],atom->tag[i3],atom->tag[i4]);
        error->warning(FLERR,str,0);
        fprintf(screen,"  1st atom: %d %g %g %g\n",
                me,x[i1][0],x[i1][1],x[i1][2]);
        fprintf(screen,"  2nd atom: %d %g %g %g\n",
                me,x[i2][0],x[i2][1],x[i2][2]);
        fprintf(screen,"  3rd atom: %d %g %g %g\n",
                me,x[i3][0],x[i3][1],x[i3][2]);
        fprintf(screen,"  4th atom: %d %g %g %g\n",
                me,x[i4][0],x[i4][1],x[i4][2]);
      }
    }

    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;

    // V(phi) = f(rMid) * p(phi)
    // f(r)   = kMult (piecewise)
    // p(phi) = k * [ 1 - cos (phi - shift) ] - vShift 

    p = 1 - c*cos_shift[type] - s*sin_shift[type];
    df1 =   s*cos_shift[type] - c*sin_shift[type];

    p = k[type]*p - vShift[type];

    if (eflag) edihedral = kMult * p;

    // force calculation begins here

    fg = vb1x*vb2xm + vb1y*vb2ym + vb1z*vb2zm;
    hg = vb3x*vb2xm + vb3y*vb2ym + vb3z*vb2zm;
    fga = fg*ra2inv*rginv;
    hgb = hg*rb2inv*rginv;
    gaa = -ra2inv*rg;
    gbb = rb2inv*rg;

    dtfx = gaa*ax;
    dtfy = gaa*ay;
    dtfz = gaa*az;
    dtgx = fga*ax - hgb*bx;
    dtgy = fga*ay - hgb*by;
    dtgz = fga*az - hgb*bz;
    dthx = gbb*bx;
    dthy = gbb*by;
    dthz = gbb*bz;

    df = -kMult * k[type] * df1;

    sx2 = df*dtgx;
    sy2 = df*dtgy;
    sz2 = df*dtgz;

    f1[0] = df*dtfx;
    f1[1] = df*dtfy;
    f1[2] = df*dtfz;

    f2[0] = sx2 - f1[0] - vb2xm*rginv*dkMult*p;
    f2[1] = sy2 - f1[1] - vb2ym*rginv*dkMult*p;
    f2[2] = sz2 - f1[2] - vb2zm*rginv*dkMult*p;

    f4[0] = df*dthx;
    f4[1] = df*dthy;
    f4[2] = df*dthz;

    f3[0] = -sx2 - f4[0] + vb2xm*rginv*dkMult*p;
    f3[1] = -sy2 - f4[1] + vb2ym*rginv*dkMult*p;
    f3[2] = -sz2 - f4[2] + vb2zm*rginv*dkMult*p;

    // apply force to each of 4 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += f1[0];
      f[i1][1] += f1[1];
      f[i1][2] += f1[2];
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] += f2[0];
      f[i2][1] += f2[1];
      f[i2][2] += f2[2];
    }

    if (newton_bond || i3 < nlocal) {
      f[i3][0] += f3[0];
      f[i3][1] += f3[1];
      f[i3][2] += f3[2];
    }

    if (newton_bond || i4 < nlocal) {
      f[i4][0] += f4[0];
      f[i4][1] += f4[1];
      f[i4][2] += f4[2];
    }

    if (evflag)
      ev_tally(i1,i2,i3,i4,nlocal,newton_bond,edihedral,f1,f3,f4,
               vb1x,vb1y,vb1z,vb2x,vb2y,vb2z,vb3x,vb3y,vb3z);
  }
}

/* ---------------------------------------------------------------------- */

void DihedralBreakable::allocate()
{
  allocated = 1;
  int n = atom->ndihedraltypes;

  memory->create(k,n+1,"dihedral:k");
  memory->create(cos_shift,n+1,"dihedral:cos_shift");
  memory->create(sin_shift,n+1,"dihedral:sin_shift");

  memory->create(setflag,n+1,"dihedral:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;

  //new variables for breakable bond
  memory->create(shift,n+1,"dihedral:shift");
  memory->create(rEq,n+1,"dihedral:rEq");
  memory->create(rCut,n+1,"dihedral:rCut");
  memory->create(vShift,n+1,"dihedral:vShift");
}

/* ----------------------------------------------------------------------
   set coeffs for one type
------------------------------------------------------------------------- */

void DihedralBreakable::coeff(int narg, char **arg)
{
  if (narg != 6) error->all(FLERR,"Incorrect args for dihedral coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  force->bounds(arg[0],atom->ndihedraltypes,ilo,ihi);

  double k_one = force->numeric(FLERR,arg[1]);
  
  //original comments : 
  // arbitrary phase angle shift could be allowed, but would break
  // backwards compatibility and is probably not needed
  
  // arbitrary shift is introduced
  double shift_one = force->numeric(FLERR,arg[2]);

  double rEq_one = force->numeric(FLERR,arg[3]);
  double rCut_one = force->numeric(FLERR,arg[4]);
  
  if (rEq_one >= rCut_one) error->all(FLERR,"rCut must be greater than rEq");

  double vShift_one = force->numeric(FLERR,arg[5]); 

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k[i] = k_one;
    
    cos_shift[i] = cos(MY_PI*shift_one/180.0);
    sin_shift[i] = sin(MY_PI*shift_one/180.0); 
    
    setflag[i] = 1;

    shift[i] = shift_one;
    rEq[i] = rEq_one;
    rCut[i] = rCut_one;
    vShift[i] = vShift_one;

    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for dihedral coefficients");
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void DihedralBreakable::write_restart(FILE *fp)
{
  fwrite(&k[1],sizeof(double),atom->ndihedraltypes,fp);
  fwrite(&shift[1],sizeof(double),atom->ndihedraltypes,fp);
  fwrite(&rEq[1],sizeof(double),atom->ndihedraltypes,fp);
  fwrite(&rCut[1],sizeof(double),atom->ndihedraltypes,fp);
  fwrite(&vShift[1],sizeof(double),atom->ndihedraltypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void DihedralBreakable::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&k[1],sizeof(double),atom->ndihedraltypes,fp);
    fread(&shift[1],sizeof(double),atom->ndihedraltypes,fp);
    fread(&rEq[1],sizeof(double),atom->ndihedraltypes,fp);
    fread(&rCut[1],sizeof(double),atom->ndihedraltypes,fp);
    fread(&vShift[1],sizeof(double),atom->ndihedraltypes,fp);
  }
  MPI_Bcast(&k[1],atom->ndihedraltypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&shift[1],atom->ndihedraltypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&rEq[1],atom->ndihedraltypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&rCut[1],atom->ndihedraltypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&vShift[1],atom->ndihedraltypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->ndihedraltypes; i++) {
    setflag[i] = 1;
      
    cos_shift[i] = cos(MY_PI*shift[i]/180.0);
    sin_shift[i] = sin(MY_PI*shift[i]/180.0); 
    
  }
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void DihedralBreakable::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ndihedraltypes; i++)
    fprintf(fp,"%d %g %g %g %g %g\n",i,k[i],shift[i],rEq[i],rCut[i],vShift[i]);
}

