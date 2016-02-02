/* SGCT-Based-Fault-Tolerant-Advection-Solver Code source file.
   Copyright (c) 2015, Md Mohsin Ali. All rights reserved.
   Licensed under the terms of the BSD License as described in the LICENSE_FT_CODE file.
   This comment must be retained in any redistributions of this source file.
*/

// Parallel 2D Advection program
// written by Peter Strazdins, May 13
// modified by Mohsin Ali, July, August, September 2013, and February, June 2014
//
// July 2013      - added "-w" option for separating the group of processes 
//                  for combination (exclusive of sub-grids)
// August 2013    - added "-i i" and "-I I" option for fault tolerance (non-real and real)
//                - featuring single or double real process failures, except process '0', on 
//                  sub-grids during the computation of simulateAdvection
// February 2014  - implemented Isend with Irecv in any order in gatherScatter
// February 2014  - data recovery for lower diagonal grids by re-sampling and for diagonal grids by
//                  copying the data from its duplicating computation (version=RESAMP_AND_COPY
//                  with option -r 1)
//                - an Isend/Irecv() based gatherScatter implementation replacing buffered
//                  send based approach of September 2013           
//   		    -----------------------------
//	            |     3|    10|      |      |
// 	            |  6,7---17,18|      |      |
//  	            |  |   |      |      |      |
//    		    ---v-------------------------
//   		    |     6|     2|      |      |
//    		    |  10  |  4,5 |      |      |
//    	            |      |/ |   |      |      |
//    		    -------/--v------------------
//    		    |     8|     5|     1|     9|
//    		    |13,14 |  9   | 2,3  |15,16 |
//    		    |      |      |/ |   |      |
//    		    --------------/--v-------|---
//    		    |      |     7|     4|     0|
//    		    |      |11,12 |  8   | 0,1  |
//    		    |      |      |      |      |
//    		    -----------------------------
//    		    1. number shown at upper right corner of a grid is `grid id'
//   		    2. number shown at middle of a grid is the rank(s) of process(es)
//                     working on that grid
//    		    3. `|' or `/' between two grids represents that they will be
//                     recovered by copying the tasks of others
//                  4. `|' represents that the lower grids tasks will be recovered
//        	        v
//       	       by re-sampling the tasks from the upper grids
//    		    5. Here level is 4 and minimum -p that required is 2
// February 2014  - data recovery by deriving an alternate combination formula
//                  (version=ALTERNATE_COMB with option -r 2)
//  	            -----------------------------
//    	            |     3|      |      |      |
//  		    |24-31 |      |      |      |
//  		    |      |      |      |      |
//  		    -----------------------------
//  		    |     6|     2|      |      |
//  		    |40-43 |16-23 |      |      |
//  		    |      |      |      |      |
//  		    -----------------------------
//  		    |     8|     5|     1|      |
//  		    |46-47 |36-39 | 8-15 |      |
//  		    |      |      |      |      |
//  		    -----------------------------
//  		    |     9|     7|     4|     0|
//  		    |  48  |44-45 |32-35 | 0-7  |
//  		    |      |      |      |      |
//  		    -----------------------------
//   		    1. number shown at upper right corner of a grid is `grid id'
//   		    2. number shown at middle of a grid is the rank(s) of process(es)
//                     working on that grid
//   		    3. grids with ids 7 and 8 are extra grids
//    		    4. Here level is 4 and minimum -p that required is 8
// February 2014  - integrate three versions (CLASSIC_COMB, RESAMP_AND_COPY, and ALTERNATE_COMB;
//                  with option -r)
// February 2014  - added "-P" option for adding OpenMP
//                  threads
// March 2014     - added Checkpoint/Restart fault tolerance capability for
//                  CHKPOINT_RESTART version (with -r 3 option)
// June 2014      - updated the code to support repeated recovery and
//                - recovery on spare node for node failure
//
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> //getopt(), gethostname()
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>
#include "mpi-ext.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#include "FTAdvect3D.h"
#include "LinGrid3D.h"
#include "FTGridCombine3D.h"
#include "Vec3D.h"

#include "FaultSimulator.h"
#include "FailureRecovery.h"

#define CONSIST_TAG           10000
#define PROCS_LIST_FAILS_TAG  10001

#define CLASSIC_COMB              0
#define RESAMP_AND_COPY           1
#define ALTERNATE_COMB            2
#define CHKPOINT_RESTART          3

#define USAGE "\n \
        ./run3dAdvect [-d] [-h] [-v v] [-V vT] [-r r] [-m method] [-l level] \n \
        \t[-p pD0] [-q pD1] [-P P] [-s sgP] [-T tF] [-S steps] \n \
        \t[-C nC] [-f] [-n] [-o] [-w] [-i i] [-I I] [g.x [g.y]] \n\t "
#define CONSTRAINTS "\n \
        level <= min(g.x,g.y)-2; \n \
        2^sgP <= nprocsUg; \n \
        sgP<=min(g.x,g.y); \n \
        i is valid for r=1, r=2, and r=3 (invalid for r=0); \n"

#define DEFAULTS "\n \
        level=4; \n \
        v=vT=0 m=2 (2nd order); \n \
        g.x=g.y=6; \n \
        2^sgP~=nprocsUg; \n \
        pD0=2 for version CLASSIC_COMB and version RESAMP_AND_COPY, and pD0=8 for version ALTERNATE_COMB; \n \
        pD1=(pD0+1)/2; \n \
        dP0=1; \n \
        P=1; \n \
        tF=0.25; \n \
        nC=1; \n \
        r=0; \n \
        i=-1; \n \
        I=1 if -i but not I is set; \n \
        i=0 if -I but not -i set;\n"

#define NOTES "\n \
        -h: for help; \n \
        -f: fixes dt across grids; \n \
        -d: debug grid combination; \n \
        v: verbosity; \n \
        vT: verbosity of timer output; \n \
        level=n: sparse grid combination for level n; \n \
        pD0=n: n processes for each grid on upper layer; \n \
        pD1=n: n processes for each grid on lower layer; \n \
        P=n: set n OpenMP threads per process; \n \
        r=0 for combination with no extra layer (CLASSIC_COMB), \n \
        \tr=1 (RESAMP_AND_COPY) for combination \n \
        \twith upper layer duplication for exact data recovery for \n \
        \tupper layer grid failure and re-sampling for lower layer \n \
        \tgrid failure, r=2 (ALTERNATE_COMB) for combination with two extra \n \
        \tsmaller layers for calculating alternate combination formula in \n \
        \tthe presence of failures (processes on a layer is half that \n \
        \tof its upper layer), r=3 (CHKPOINT_RESTART) for combination with no \n \
        \textra layer and fault tolerance by checkpoint/restart method; \n \
        -n: print required nprocsUg only; \n \
        tF=n: final time n seconds; \n \
        nC=n: total n combinations; \n \
        steps=n: total n steps; \n \
        sgP=n: n processes on sparse grid; \n \
        g.x, g.y: grid size (g.x) X (g.y); \n \
        -o: print separate required sparse grid processes (nprocsSg) only; \n \
        -w: for separate group of processes for combination (exclusive of \n \
        \tsub-grid process group); \n \
        i=0 for injecting non-real and i=1 for real process failures in \n \
        \tthe system (other than process 0) at time 0.1250; \n \
        I=n: injecting n non-real or real process failures; \n\n"
#define OPTCHARS "v:m:l:s:dp:q:P:T:S:C:V:r:i:I:hfnow"

/*  0: top-level messages from rank 0, 1: 1-off messages from all ranks,
 2: per-iteration messages, 3: dump data: one-off, 4: dump data, per itn.
 */
static int verbosity = 0;          // v above    
static int version = CLASSIC_COMB; // settings of which version of sparse grid combination will be used (r above)
                                   // default is version CLASSIC_COMB
static bool help = false;
static int verbTim = 0;            // vT above. Verbosity (lint maxLevelevel) of timer output
static bool dbgGridComb = false;   // set to test/debug combination alg
static int method = 2;
static int level = 4;        // use combination algorithm for l>=4 
static Vec2D<int> grid;
static int sgProcs = -1;     // not set
static double tF = 0.25,     // final time; must it be a multiple of 0.25?
	      CFL = 0.25;    // CFL condition number
static int rank, nprocs;     // nprocs is the total number of processes (entry after -np)
static int nprocsUg,         // number of processors for sub-grids 
	   nprocsSg;         // number of processors for sparse grid
static int pD[2];            // number of processes for diagonal/lower grids
static int P = 1;            // number of OpenMP threads per process
static int steps = 0;        // #timesteps for sparse grid (dept. on grid & TF)
static int nC = 1;           // apply combination algorithm nC times over TF
static bool fixedDt = false; // if set, use a fixed dt across grids
static bool separateSg = false; // if set, use separate sparse grid processes 
static int failureType = -1; // 0 means non-real and 1 means real process failure in the communicator
static int totFails = 0; //a total of totFails number of processes (non-real or real)  is failed
static bool fixedProcs;      // signifies if #processes isfixed across diagonal
// Otherwise pD refers to the central points and
// the number doubles as you go outwards.
// If set, only gets good load balance if fixedDt

// GLOBAL variable declarations and initializations
static MPI_Errhandler newEh;
static Timer * timer = NULL;
static int grank, tempRank, nPtsL = 0, isNewlySpawned = 0, //determine whether newly spawned or not ('1' for newly spawned, '0' for not)
	 	                       isParent = 1;       //used to determine parent ('1' for parent, '0' for child)
static GridCombine2D * gc = NULL;
static MPI_Comm myGridComm, mySubSparseGridComm, myCommWorld = MPI_COMM_WORLD;
static ProcGrid2D *g = NULL, *sg = NULL;
static Vec2D<int> myGrid;
static HaloArray2D *u = NULL, *usg = NULL;
static Advect2D *adv = NULL, *advsg = NULL;
static LinGrid *ulg = NULL, *slg = NULL;
static double t = 0.0, advT = 0.0, ckptAdvT = 0.0, maxAdvT = 0.0, tC = 0.0,
        dtC = 0.0, advConsis[8], bogus_t = 0.0, sTimePerStep = 0.0, eTimePerStep = 0.0;
static int sumPrevNumNodeFails = 0, numNodeFails = 0;

static int i = 0, j = 0, advS = 0, T = 0, ckptS = 0, numFails = 0, *listFails = NULL, totCkpts = 0;
static MPI_Status advConsisStatus, procsListFailsStatus;
static char ** argvChild = NULL;
static int keyVal = 0, keyCol = 0;
static double sSimAdvTime = 0.0, eSimAdvTime = 0.0, ckpntSimAdvTime = 0.0, totalExpectedTime = 0.0, 
	MTTI = 1.0;//0.0; // Mean Time To Interrup on each node
static int injectCounter = 0, totRealFails = 0, faultSimulated = 0;
static double sCkptWrtTime = 0.0, eCkptWrtTime = 0.0, singleCkptWriteTime = 0.0, singleCkptReadTime = 0.0;

// print a usage message for this program and exit with a status of 1
void usage(std::string msg) {
    if (rank == 0) {
        printf("\ntwoDimAdvect: %s\n", msg.c_str());
        printf("\nUSAGE:%s\nCONSTRAINTS:%s\nDEFAULT VALUES:%s\nNOTES:%s",
        USAGE, CONSTRAINTS, DEFAULTS, NOTES);
    }
    exit(1);
}

#define POWER_OF_TWO(x) (((x)&(x-1)) == 0)

void getArgs(int argc, char *argv[], int rank, int nprocs) {
	extern char *optarg; // points to option argument (for -p option)
	extern int optind;   // index of last option parsed by getopt()
	extern int opterr;
	char optchar;        // option character returned my getopt()
	opterr = 0;          // suppress getopt() error message for invalid option
	pD[0] = pD[1] = 0;
	bool optNgiven = false; //records  if -n was given
	bool optOgiven = false; //records  if -o was given

	while ((optchar = getopt(argc, argv, OPTCHARS)) != -1) {
	   // extract next option from the command line
	   switch (optchar) {
	   case 'v':
	      if (sscanf(optarg, "%d", &verbosity) != 1)   // invalid integer
		  usage("bad value for verbose");
	      break;
	   case 'r':
	      if (sscanf(optarg, "%d", &version) != 1)     // invalid integer
		  usage("bad value for version");
	      break;
           case 'i':
              if (sscanf(optarg, "%d", &failureType) != 1)     // invalid integer
                  usage("bad value for failureType");
              break;
           case 'I':
              if (sscanf(optarg, "%d", &totFails) != 1)     // invalid integer
                  usage("bad value for totFails");
              break;
	   case 'm':
	      if (sscanf(optarg, "%d", &method) != 1)      // invalid integer
		 usage("bad value for method");
	      break;
	   case 'l':
	      if (sscanf(optarg, "%d", &level) != 1)       // invalid integer
		 usage("bad value for level");
	      break;
	   case 's':
	      if (sscanf(optarg, "%d", &sgProcs) != 1)
		 usage("bad value for sgProcs");
	      break;
	   case 'p':
	      if (sscanf(optarg, "%d", &pD[0]) != 1)
		 usage("bad value for pD0");
	      break;
           case 'q':
	      if (sscanf(optarg, "%d", &pD[1]) != 1)
		 usage("bad value for pD1");
	      break;
	   case 'P':
	      if (sscanf(optarg, "%d", &P) != 1)
		 usage("bad value for P");
	      break;
	   case 'd':
	      dbgGridComb = true;
	      break;
	   case 'S':
	      if (sscanf(optarg, "%d", &steps) != 1)
		 usage("bad value for steps");
	      break;
	   case 'C':
	      if (sscanf(optarg, "%d", &nC) != 1)
		 usage("bad value for nC");
	      break;
	   case 'T':
	      if (sscanf(optarg, "%lf", &tF) != 1)
		 usage("bad value for tF");
	      break;
	   case 'V':
	      if (sscanf(optarg, "%d", &verbTim) != 1)
		 usage("bad value for vT");
	      break;
	   case 'h':
	      help = true;
	      break;
	   case 'f':
	      fixedDt = true;
	      break;
	   case 'n':
	      optNgiven = true;
	      break;
	   default:
              usage("unknown option");              
	      break;
	   } //switch
	} //while

	grid.x = grid.y = 6;

	if (optind < argc)
	   if (sscanf(argv[optind], "%d", &grid.x) != 1)
			usage("bad value g.x");
	if (optind + 1 < argc)
	   if (sscanf(argv[optind + 1], "%d", &grid.y) != 1)
			usage("bad value g.y");

	fixedProcs = fixedDt;

	if (pD[0] == 0 && (version == CLASSIC_COMB || version == RESAMP_AND_COPY || version == CHKPOINT_RESTART)) {
	   pD[0] = 2;
	   // pD[0] = GridCombine2D::getPD0(nprocs, level, fixedProcs);
	}
	if (pD[0] == 0 && version == ALTERNATE_COMB) {
	   pD[0] = 8;
	   // pD[0] = GridCombine2D::getPD0(nprocs, level, fixedProcs);
	}
	if (pD[1] == 0){
	   pD[1] = GridCombine2D::getPD1(pD[0], fixedProcs);
        }
	if (sgProcs == -1) { // use as many process as possible for the sparse grid
	   if (version == CLASSIC_COMB || version == CHKPOINT_RESTART) {
	      sgProcs = (int) log2((double) GridCombine2D::nProcs(version, level, pD, fixedProcs));
	   } 
           else if (version == RESAMP_AND_COPY) {
	      sgProcs = (int) log2((double) GridCombine2D::nProcsTwoDiags(level, pD, fixedProcs));
           } 
           else if (version == ALTERNATE_COMB) {
	      sgProcs = (int) log2((double) GridCombine2D::nProcs(version, level, pD, fixedProcs));
	   }
	}

	nprocsUg = GridCombine2D::nProcs(version, level, pD, fixedProcs);
	nprocsSg = separateSg ? 1 << sgProcs : 0;

	int dx = 1 << std::max(grid.x, grid.y);
	if (steps == 0)
	   steps = (int) (tF / CFL * dx);
	else{
	   tF = CFL * steps / (1.0 * dx);
        }
	if (optNgiven && !optOgiven) {
	   printf("%d MPI processes are needed for given command line parameters for both sub-grids and sparse "
                  "grid if -w is not set. Otherwise, for sub-grids only\n",
	        GridCombine2D::nProcs(version, level, pD, fixedProcs));
	   exit(0);
	}

	if (!optNgiven && optOgiven) {
	   printf("%d MPI processes are needed for given command line parameters for sparse grid only\n",
				1 << sgProcs);
	   exit(0);
	}

	//set some default values
	if ((failureType != -1) && (version == CLASSIC_COMB)) {
	   usage("Fault Tolerance (option -i) is not available for version 0 (CLASSIC_COMB) "
                 "with option -r 0. Choose version 1 (RESAMP_AND_COPY) or "
                 "2 (ALTERNATE_COMB) or 3 (CHKPOINT_RESTART) for Fault Tolerance.");
	}

	if ((failureType == -1) && (totFails != 0)) {
	   failureType = 0;
	}

	if ((failureType != -1) && (totFails == 0)) {
	   totFails = 1;
	}

	if (failureType > 1) {
	   failureType = 1; //Currently, it takes only 0 and 1
	}

	if (help) {
	   usage("Help");
	}

#ifdef _OPENMP
	omp_set_num_threads(P);
#endif
	if (rank == 0) {
	   printf("Level %d combination alg to be applied %d times with %d,%d procs/grid%s "
           "combine on 2^%d %sprocs (%d MPI procs)\n", level, nC, pD[0], pD[1], fixedProcs ? "," :
           " (doubling outwards from center),", sgProcs, separateSg ? "separate " : 
           "in-sub-grids ", nprocs);
#ifdef _OPENMP
	printf("[%d] ===== %d threads per process =====\n", rank, omp_get_max_threads());

#pragma omp parallel
	{
	   printf("%d: hello from thread\n", omp_get_thread_num());
	}
#endif
	}

	if (level > (std::min(grid.x, grid.y) - 2)) {
	   usage("level must be <= min(g.x,g.y)-2");
	}

	if ((1 << sgProcs) > nprocsUg || sgProcs > std::min(grid.x, grid.y)) {
	   usage("sgProcs fails its constraints");
	}

	if (nprocsUg != GridCombine2D::nProcs(version, level, pD, fixedProcs)) {
	   usage("nprocsUg fails its constraint");
	}

} //getArgs()

static void printLocGlobAvgs(std::string name, double total, int nlVals,
		Vec2D<int> gix, MPI_Comm comm) {
	int ngVals = GridCombine2D::gridSz(gix).prod();

	double v[1];
	if (verbosity > 0)
		printf("%d: grid (%d,%d): local avg %s is %.2e\n", rank, gix.x, gix.y,
				name.c_str(), nlVals == 0 ? 0.0 : total / nlVals);
	MPI_Reduce(&total, v, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
	int grank;
	MPI_Comm_rank(comm, &grank);
	if (grank == 0){
	   printf("%d: grid (%d,%d): avg %s %.2e\n", rank, gix.x, gix.y,
				(char *) name.c_str(), v[0] / ngVals);
	}
} //printLocGlobAvgs()

void initializeVariablesPartOne(void) {
	timer = new Timer;
	tempRank = separateSg ? ((rank >= nprocsUg) ? -1 : rank) : rank;
	gc = new GridCombine2D(level, pD, fixedProcs, sgProcs, grid, timer,
			verbosity, dbgGridComb, separateSg, version, myCommWorld);

	if (!MPI_SUCCESS == MPI_Comm_split(myCommWorld, gc->getGid(rank), gc->getGrank(rank),
					&myGridComm)) {
	   printf("Error in MPI_Comm_split. Abort\n");
	   exit(1);
	}
	MPI_Comm_rank(myGridComm, &grank);


	// separate sparse grid: (1) grid Id = 3*level-1 for version RESAMP_AND_COPY
        //                           and 4*level-6 for version ALTERNATE_COMB, local 
        //                           rank = 0 , 1, 2, ... (numprocs in sparse grid -1)
	//                       (2) mySubSparseGridComm separates communicator for sub-grid and sparse grid

	if (rank < gc->nProcs(version, level, pD, fixedProcs)) {
	   keyVal = 0;
	   keyCol = rank;
	} 
        else if (nprocsSg != 0 && rank >= (gc->nProcs(version, level, pD, fixedProcs))) { // alternative condition is "tempRank == -1"
	   keyVal = 1;
	   keyCol = gc->getGrank(rank);
	}

	if (!MPI_SUCCESS == MPI_Comm_split(myCommWorld, keyVal, keyCol, &mySubSparseGridComm)) {
	   printf("Error in MPI_Comm_split. Abort\n");
	   exit(1);
	}
} //initializeVariablesPartOne()

void initializeVariablesPartTwo(int isUpdateBoundary) {
	g = gc->pgU;
	sg = gc->pgS;
	myGrid = gc->gxU;
	u = new HaloArray2D(g->G2L(gc->gridSz(myGrid)), 1);
	usg = new HaloArray2D(sg->G2L(gc->gridSz(grid)), 0);
	adv = new Advect2D(gc->gridSz(myGrid), Vec2D<double>(1.0, 1.0), tF, CFL,
			method, verbosity, timer, mySubSparseGridComm, myGridComm,
			gc->getGid(rank), gc->getGrank(rank), myCommWorld);
	advsg = new Advect2D(gc->gridSz(grid), Vec2D<double>(1.0, 1.0), tF, CFL,
			method, verbosity, timer, mySubSparseGridComm, myGridComm,
			gc->getGid(rank), gc->getGrank(rank), myCommWorld);
	int my_size, my_rank;
	MPI_Comm_size(myGridComm, & my_size);
	MPI_Comm_rank(myGridComm, & my_rank);

	if (fixedDt) {
		// sparse gid will also same dt as that needed on grids
		// (level,1) and (1, level), the finest dt
		adv->dt = advsg->dt;
	}

	ulg = new LinGrid(gc->gridSz(myGrid),
			gc->gridSz1(grid) / gc->gridSz1(myGrid));
	slg = new LinGrid(gc->gridSz(grid), Vec2D<int>(1, 1));

	if (rank == 0) {
	   if (not dbgGridComb){
	      printf("2d advection via method %d on grid %d,%d for time %e = max %d "
                      "steps (dt=%e -%sfixed)\n", method, grid.x, grid.y, tF, steps, adv->dt,
					fixedDt ? "" : "not ");
           }
	   else{
	      printf("testing grid combination on grid %d,%d for %d reps\n",
					grid.x, grid.y, nC);
           }
	}

	if (verbosity > 0) {
	   char hostName[128];
	   gethostname(hostName, sizeof(hostName));

	   if (tempRank != -1) {
	      // only for sub-grids
	      printf("%d: process (%d,%d) on grid (%d,%d) with %dx%d processes",
			rank, g->id.x, g->id.y, myGrid.x, myGrid.y, g->P.x, g->P.y);
	      printf(" computes %dx%d points from (%d,%d) on host %s\n", u->l.x,
			u->l.y, g->L2G0(0, adv->gridSize.x),
			g->L2G0(1, adv->gridSize.y), hostName);
	   }
	}

	if (rank == 0) {
	   printf("Compute %ssparse grid (%d,%d) with %dx%d procs\n",
		separateSg ? "separate " : "in-sub-grids ", grid.x, grid.y,
		sg->P.x, sg->P.y);
	}

	nPtsL = 0;
	if (tempRank != -1) {
	   //Only for sub-grids
	   nPtsL = u->l.prod();
	   if (not dbgGridComb) {
	      adv->initGrid(u, g);
		 if (isUpdateBoundary) {
		    adv->updateBoundary(u, g, 0);
		 }
		 //printLocGlobAvgs("norm of initial field", u->norm1(), nPtsL, myGrid,
		 //		     myGridComm);
	   } 
           else
	      ulg->initGrid(u, g);
	}

	if (verbosity > 2)
	   if (verbosity > 2 && tempRank != -1)
	      u->print(rank, "initial field");

	usg->zero(); // in case gatherRecv() is not called to zero it
} //initializeVariablesPartTwo()

/**************************************************
 **                Main Function                 **
 **************************************************/
int main(int argc, char** argv) {

	MPI_Init(&argc, &argv);

	double startTime, endTime;
	startTime = MPI_Wtime();

	// error handler
	MPI_Comm_create_errhandler(mpiErrorHandler, &newEh);
	MPI_Comm_set_errhandler(myCommWorld, newEh);

	// getting parent
	MPI_Comm parent;
	MPI_Comm_get_parent(&parent);

	// control starts from here
	if (MPI_COMM_NULL != parent) { // this is child (newly spawned)
            isNewlySpawned = 1; // this is newly spawned
            totFails = 0; nC = 1; tempRank = 0; dbgGridComb = false;
	} // end of child (newly spawned)

	if (MPI_COMM_NULL == parent) { // this is parent
            isNewlySpawned = 0; // this is NOT newly spawned
            totFails = 0;

            MPI_Comm_rank(myCommWorld, &rank);
            MPI_Comm_size(myCommWorld, &nprocs);

            getArgs(argc, argv, rank, nprocs);

            //==============================
            initializeVariablesPartOne();
            initializeVariablesPartTwo(1);
            //==============================

            t = 0.0; tC = 0.0; // time at last combine operation
	} // end of parent

	for (int s = 0; s < nC; s++) {
            if (MPI_COMM_NULL == parent) { //parent
                dtC = tF / nC; advT = 0.0; advS = 0; injectCounter = 0;
                // this will enable check for process failure after completing the whole time steps under each nC
                while (advT < dtC) {
                    advT += adv->dt; advS++;
                }
                T = advS; ckptS = 1; maxAdvT = advT; advT = 0.0; advS = 0;
            } // end of if
            else { //child
                myCommWorld = MPI_COMM_WORLD;
                advT = 0.0; dtC = 1.0; advS = 1; T = 1; ckptS = 1; faultSimulated = 1; injectCounter = 0;
            }

            if (tempRank != -1) { // only on sub-grids
                if (not dbgGridComb) {
                    if (MPI_COMM_NULL == parent) { //parent
                            MPI_Pcontrol(1, "simulateAdvection");
                            MPI_Barrier(mySubSparseGridComm);
                    }
                } // end of "if (not dbgGridComb)"
            } // end of "if(tempRank != -1)"

            while (advT < dtC) {
                if (isNewlySpawned != 1) { // not newly spawned
                    if (tempRank != -1) { // only on sub-grids
                        if (not dbgGridComb) {
                            timer->start("simulateAdvection", u->l.prod(), 0);
                            
                            sTimePerStep = MPI_Wtime();
                            // call simulateAdvection
                            t += adv->simulateAdvection(u, g);
                            eTimePerStep = MPI_Wtime();
                            
                            timer->stop("simulateAdvection");

                            // update time and steps
                            advT += adv->dt; advS++;
                        } // end of "if (not dbgGridComb)"
                    } // end of "if(tempRank != -1)"
                } // end of "if(isNewlySpawned != 1)"

                // saving checkpoint on disk for version CHKPOINT_RESTART and for sub-grid parent processes
                if (version == CHKPOINT_RESTART && tempRank != -1 && parent == MPI_COMM_NULL
                                && s == 0 && advS == 1 && injectCounter == 0) {
                    injectCounter = 1;  

                    sCkptWrtTime = MPI_Wtime();                                
                    checkpointWrite2D(u, rank);
                    eCkptWrtTime = MPI_Wtime();
       
                    singleCkptWriteTime = eCkptWrtTime-sCkptWrtTime;
                    totalExpectedTime = steps*(eTimePerStep-sTimePerStep);
                    //MTTI = totalExpectedTime / 2;  //assuming MTTI is 1/2 of total expected time
                    if(rank == 0){
                       //totCkpts = (int)sqrt((double)(MTTI/singleCkptWriteTime));
                       totCkpts = (int)(totalExpectedTime / sqrt((double)(2 * singleCkptWriteTime * MTTI/ceil((double)(nprocs/SLOTS)))));
                       if(totCkpts == 0){
                          totCkpts = 1;
                       }

                       MPI_Bcast(&totCkpts, 1, MPI_INT, 0, myCommWorld);
                    }
                    else{
                       MPI_Bcast(&totCkpts, 1, MPI_INT, 0, myCommWorld);                                
                    }

                    ckptS = (int)(T/totCkpts);
                    
                    if(rank == 0){
                     printf("[%d] ===== Called checkpoint WRITE: step = %d, advS = %d, "
                        "T = %d, ckptS = %d, totCkpts = %d, totalExpectedTime = %0.2lf, "
                        "MTTI = %0.2lf, singleCkptWriteTime = %0.2lf =====\n\n", rank, s, advS, T, ckptS, totCkpts, 
                        totalExpectedTime, MTTI, singleCkptWriteTime);
                    }
                                
                }

                MPI_Comm_rank(myCommWorld, &rank);
                MPI_Comm_size(myCommWorld, &nprocs);
                isParent = 1;

                // set value to recognize child
                if (parent != MPI_COMM_NULL) {
                    isParent = 0;
                }

                // injecting faults in the system by failing processes.
                // injecting faults only on sub-grid processes.

                if ((failureType != -1) && tempRank != -1 && s >= 0 && parent == MPI_COMM_NULL
                                && advS == (T/2)) { //non-real or real process failure after 50% advancement of time loop

                    faultSimulated = 1;
                    listFails = (int *) malloc(totFails * sizeof(int)); //released later
                    if (version == RESAMP_AND_COPY) {
                        // failure only on lower diagonal sub-grids
                        // Ranks are ordered as diag, lower diag, and duplicate
                        //faultSimulate(failureType, listFails, totFails, rank, 
                        //          GridCombine2D::nProcsTwoDiags(level, pD, fixedProcs), level*pD[0], myCommWorld);

                        // failure only on duplicate of diagonal sub-grids
                        // Ranks are ordered as diag, lower diag, and duplicate
                        faultSimulate(failureType, listFails, totFails, rank, 
                                  GridCombine2D::nProcs(version, level, pD, fixedProcs),
                                                  level*pD[0]+(level-1)*pD[1], myCommWorld);

                        // failure on lower diagonal sub-grids and duplicate of diagonal sub-grids
                        // Ranks are ordered as diag, lower diag, and duplicate
                        //faultSimulate(failureType, listFails, totFails, rank,
                        //                GridCombine2D::nProcs(version, level, pD,
                        //                                fixedProcs), level * pD[0], myCommWorld);
                    }
                    else if (version == ALTERNATE_COMB) {
                        // failure on diagonal, lower diagonal, and extra sub-grids
                        // Ranks are ordered as diag, lower diag, and extra diags
                        faultSimulate(failureType, listFails, totFails, rank,
                                        GridCombine2D::nProcs(version, level, pD,
                                                        fixedProcs), 0, myCommWorld); //last parameter 0 is converted to 1 inside
                                                                                      // the function as process 0 is not allowed to fail
                    }
                    else if (version == CHKPOINT_RESTART) {
                        // failure on diagonal and lower diagonal sub-grids
                        // Ranks are ordered as diag and lower diag diags
                        faultSimulate(failureType, listFails, totFails, rank,
                                        GridCombine2D::nProcs(version, level, pD,
                                                        fixedProcs), 0, myCommWorld); //last parameter 0 is converted to 1 inside
                                                                                      // the function as process 0 is not allowed to fail
                    }
                }

                // check for process failure after completing the whole time steps under each nC for 
                // version RESAMP_AND_COPY and ALTERNATE_COMB
                // check for process failure before writing to checkpoint file, write if no failure, 
                // otherwise no write for version CHKPOINT_RESTART
                if((advS > 0 && advS <= T && (((advS % T) == 0 && (version == RESAMP_AND_COPY || version == ALTERNATE_COMB)) || 
                    ((advS % ckptS) == 0 && version == CHKPOINT_RESTART))) || MPI_COMM_NULL != parent){
                    /*
                    // call communicator reconstruct function
                    // make child to parent which become child automatically in communicatorReconstruct
                    if (parent != MPI_COMM_NULL) {
                            parent = MPI_COMM_NULL;
                    }
                    */

                    if (isNewlySpawned == 1 || isParent == 0) {// newly spawned
                        // NULL should be replaced by listFails, but totFails is not available on spawned.
                        // totFails can be passed as command-line argument and then allocate memory for listFails
                        // later
                        myCommWorld = communicatorReconstruct(myCommWorld, isNewlySpawned, NULL, &numFails, 
                                      &numNodeFails, sumPrevNumNodeFails, argc, argv, verbosity);
                    }
                    // call communicator reconstruct function
                    // make child to parent which become child automatically in communicatorReconstruct
                    if (parent != MPI_COMM_NULL) {
                        parent = MPI_COMM_NULL;
                    }

                    if (isNewlySpawned != 1) {// not newly spawned
                        if((faultSimulated == 1 && (version == RESAMP_AND_COPY || version == ALTERNATE_COMB)) || version == CHKPOINT_RESTART){
                           totRealFails = numProcsFails(myCommWorld);

                           if (totRealFails > 0) {
                               //listFails = (int *)malloc(totFails*sizeof(int)); //released later
                               myCommWorld = communicatorReconstruct(myCommWorld, isNewlySpawned, listFails, &numFails, 
                                             &numNodeFails, sumPrevNumNodeFails, argc, argv, verbosity);
                           }
                        }
                    }

                    // bring child to its original position
                    if (isParent == 0) {
                        parent = myCommWorld;
                    }

                    // set error handler for new communicator
                    MPI_Comm_set_errhandler(myCommWorld, newEh);

                    // make child to parent
                    // make it obvious that child is generated from failureType 1 and version 
                    // is RESAMP_AND_COPY (or ALTERNATE_COMB or CHKPOINT_RESTART)
                    if (MPI_COMM_NULL != parent) {// child
                        isParent = 0;
                        parent = MPI_COMM_NULL;
                        failureType = 1;
                        version = RESAMP_AND_COPY; // ALTERNATE_COMB or CHKPOINT_RESTART is also OK
                    } else { // parent
                        isParent = 1;
                    }
                    isNewlySpawned = 0; // not newly spawned at the moment. maybe spawned before

                    //adjust parameters for non-real process failure
                    if (s == 0 && faultSimulated == 1 && failureType == 0 && (version == RESAMP_AND_COPY || 
                        version == ALTERNATE_COMB || version == CHKPOINT_RESTART)) {
                        myCommWorld = MPI_COMM_WORLD;
                        numFails = totFails;
                        for (i = 0; i < numFails; i++) {
                            if (listFails[i] == rank) {
                                isParent = 0;
                            }
                        }
                    }

                    if (numFails > 0 && faultSimulated == 1 && (version == RESAMP_AND_COPY || version == ALTERNATE_COMB || 
                        version == CHKPOINT_RESTART)) { 
                        //numFails is available to all processes, including recovered
                        if (failureType > 0) { //real process failure
                            if (isParent == 0) { // this is child
                                // passing command-line arguments to re-constructed process(es)
                                MPI_Comm_rank(myCommWorld, &rank);
                                MPI_Comm_size(myCommWorld, &nprocs);
                                argvChild = (char **) malloc((argc - 1) * sizeof(char *)); // released later
                                for (i = 0; i < argc - 1; i++) {
                                        argvChild[i] = strdup(argv[i + 1]);
                                }
                                getArgs(argc - 1, argvChild, rank, nprocs);
                            } // end of child

                        } //end of if(failureType > 0)

                        if (failureType > 0) { //real process failure
                            //===========================
                            initializeVariablesPartOne();
                            //===========================
                        }

                        if (isParent == 0|| (isParent != 0 && ranksOnSameGrid(rank, numFails, listFails, gc))) { 
                           // this is child or child's same-grid process
                           // listFails is NOT available to recovered process(es)
                           // initialize re-constructed process and re-initialize processes with the same grid as failed process
                            if (failureType > 0) {//real process failure
                                    //============================
                                    initializeVariablesPartTwo(0);
                                    //============================
                            }

                            MPI_Comm_rank(myCommWorld, &rank);
                            MPI_Comm_size(myCommWorld, &nprocs);
                            printf("[%d] (currently %s): re-initialization, recovered or processes with same gridId\n",
                                            rank, (MPI_COMM_NULL == parent) ? "parent" : "child");

                        } // end of child or child's same-grid process

                        // Accumulate number of node failures to track the spare node's index in hostfile
                        sumPrevNumNodeFails += numNodeFails;

                        // make advT, dt, dtC, t, T, and sumPrevNumNodeFails consistent across all processes by process 0.
                        MPI_Comm_rank(myCommWorld, &rank);
                        MPI_Comm_size(myCommWorld, &nprocs);

                        if (rank == 0) {
                            advConsis[0] = advT;
                            advConsis[1] = adv->dt;
                            advConsis[2] = dtC;
                            advConsis[3] = t;
                            advConsis[4] = T;
                            advConsis[5] = advS;
                            advConsis[6] = s;
                            advConsis[7] = sumPrevNumNodeFails;

                            for (i = 0; i < numFails; i++) {
                                MPI_Send(&advConsis, 8, MPI_DOUBLE, listFails[i],
                                CONSIST_TAG, myCommWorld);

                                // sending failed ranks information to the failed ranks
                                MPI_Send(&(listFails[0]), numFails, MPI_INT,
                                                listFails[i], PROCS_LIST_FAILS_TAG,
                                                myCommWorld);
                            }
                        }

                        if (isParent == 0) { // child which was/were failed
                            MPI_Recv(&advConsis, 8, MPI_DOUBLE, 0, CONSIST_TAG,
                                            myCommWorld, &advConsisStatus);
                            advT = advConsis[0];
                            adv->dt = advConsis[1];
                            dtC = advConsis[2];
                            t = advConsis[3];
                            T = (int) advConsis[4];
                            advS = (int) advConsis[5];
                            s = (int) advConsis[6];
                            sumPrevNumNodeFails = (int) advConsis[7];

                            // receiving failed ranks information to the failed ranks from process 0
                            int * tempFails = (int *) malloc(numFails * sizeof(int));
                            listFails = (int *) malloc(numFails * sizeof(int));
                            MPI_Recv(&(tempFails[0]), numFails, MPI_INT, 0,
                            PROCS_LIST_FAILS_TAG, myCommWorld,
                                            &procsListFailsStatus);

                            for (i = 0; i < numFails; i++) {
                                listFails[i] = tempFails[i];
                            } // end of for
                            free(tempFails);
                        }

                        double sTime = 0.0, eTime = 0.0, sCkptReadTime = 0.0, eCkptReadTime = 0.0;

                        if(isParent == 0 || rank == 0){
                            sTime = MPI_Wtime();
                            sCkptReadTime = MPI_Wtime();
                        }

                        // call recovery method for version RESAMP_AND_COPY
                        if(version == RESAMP_AND_COPY) {
                           recoveryByResampling(u, g, myCommWorld, isParent, gc, rank, nprocs, level, numFails, listFails);

                           if(isParent == 0){
                              eTime = MPI_Wtime();
                              printf("[%d]----- Recovering data of the failed grid takes %0.6f Sec (MPI_Wtime) -----\n",
                                                    rank, eTime - sTime);
                           }
                        }

                        // call recovery method for version CHKPOINT_RESTART
                        else if(version == CHKPOINT_RESTART) {
                           if(rank == 0){
                              printf("[%d] ===== Call checkpoint READ: step = %d, advS = %d, T = %d =====\n\n", s, rank, advS, T);
                           }

                           checkpointRead2D(u, rank);

                           /*
                           if(rank == 4){
                           //print contents of read buffer
                              printf("Contents of readBuff (rank %d) is: \n", rank);
                              for (j = 0; j < u->s.y; j++){
                                 for (i = 0; i < u->s.x; i++){
                                    printf("%0.2f ", *(u->ix(i, j)));
                                 }
                                 printf("\n");
                              }
                              printf("\n\n");
                           }
                           */
                           if(rank == 0){
                              eCkptReadTime = MPI_Wtime();
                           }

                           nPtsL = 0;

                           sSimAdvTime = 0.0, eSimAdvTime = 0.0, ckpntSimAdvTime = 0.0;
                           sSimAdvTime = MPI_Wtime();
                           ckptAdvT = 0.0;

                           if (tempRank != -1) {
                              //Only for sub-grids
                              nPtsL = u->l.prod();
                              if (not dbgGridComb) {
                                 //adv->initGrid(u, g); // this has been done before interpolation
                                     adv->updateBoundary(u, g, 1); // this works on sub-grids
                                     //printLocGlobAvgs("norm of initial field", u->norm1(), nPtsL, myGrid,
                                     //myGridComm);
                              } 
                              else
                                 ulg->initGrid(u, g);
                           }

                           while (ckptAdvT < ckptS*(adv->dt)) {
                                if (not dbgGridComb) {
                                    timer->start("simulateAdvection", u->l.prod(), 0);
                                    // call simulateAdvection
                                    bogus_t = adv->simulateAdvection(u, g);
                                    timer->stop("simulateAdvection");
                                    // update time
                                    ckptAdvT += adv->dt;
                                } // end of "if (not dbgGridComb)"
                           } // end of "if(tempRank != -1)"
                           eSimAdvTime = MPI_Wtime();
                           ckpntSimAdvTime = (eSimAdvTime - sSimAdvTime);

                           if(rank == 0){
                              singleCkptWriteTime = eCkptWrtTime - sCkptWrtTime;
                              singleCkptReadTime = eCkptReadTime - sCkptReadTime;
                              printf("[%d]----- Recovering data of the failed grid by Checkpoint/Restart (writing plus "
                                     "reading plus re-computation) takes %0.6f Sec (MPI_Wtime) -----\n",
                                                    rank,  totCkpts*singleCkptWriteTime + singleCkptReadTime + ckpntSimAdvTime);
                           }
                        }// end of "else if(version == CHKPOINT_RESTART)"

                        if ((version == RESAMP_AND_COPY || version == ALTERNATE_COMB) && (isParent == 0 || (isParent != 0 && ranksOnSameGrid(rank, numFails,
                            listFails, gc)))) { 
                            // this is child or child's same-grid process // listFails is NOT available to recovered process(es)
                            nPtsL = 0;

                            if (tempRank != -1) {
                                    //Only for sub-grids
                                    nPtsL = u->l.prod();
                                    if (not dbgGridComb) {
                                            //adv->initGrid(u, g); // this has been done before interpolation
                                            adv->updateBoundary(u, g, 0); // this works on sub-grids
                                            //printLocGlobAvgs("norm of initial field", u->norm1(), nPtsL, myGrid,
                                            //myGridComm);
                                    } else
                                            ulg->initGrid(u, g);
                            }
                        }// end of child or child's same-grid process

                        // reset numFails and totRealFails
                        numFails = 0; totRealFails = 0; faultSimulated = 0;

                        // free allocated memory
                        if (isParent == 0) {// this is child
                                free(argvChild);
                        }
                    }// end of "if (numFails > 0 && (version == RESAMP_AND_COPY || version == ALTERNATE_COMB || version == CHKPOINT_RESTART))"

                    if(version == CHKPOINT_RESTART && isParent != 0){// writing to checkpoint file for parent only
                        if(rank == 0){
                                printf("[%d] ===== Call checkpoint WRITE: step = %d, advS = %d, T = %d =====\n\n", rank, s, advS, T);
                        }
                        checkpointWrite2D(u, rank);
                    }
                }// end of "if((advS > 0 && advS <= T && (((advS % T) == 0 && (version == RESAMP_AND_COPY || version == ALTERNATE_COMB)) || 
                 // ((advS % ckptS) == 0 && version == CHKPOINT_RESTART))) || MPI_COMM_NULL != parent)"

                if (verbosity > 3) {
                    char s[64];
                    sprintf(s, "after time %.4e, field is:\n", t);
                    u->print(g->myrank, s);
                }

            }//end of "while(advT < dtC)"

            if (tempRank != -1) {// only on sub-grids
                if (not dbgGridComb) {
                    MPI_Barrier(mySubSparseGridComm);
                    MPI_Pcontrol(-1, "simulateAdvection");
                    if (MPI_COMM_NULL == parent) {//parent
                        if (nC == 1)
                            printLocGlobAvgs("error of updated field",
                                            adv->checkError(t, u, g), nPtsL, myGrid,
                                            myGridComm);
                    }
                }// end of "if (not dbgGridComb)"
            }// end of "if(tempRank != -1)"

            tC = t;
            MPI_Pcontrol(1, "gatherScatter");
            gc->gatherScatter(u, usg, listFails, totFails, mySubSparseGridComm);
            if (version == RESAMP_AND_COPY) {
               duplicateToSpareGrids(u, g, myCommWorld, gc, rank, nprocs, level); 
               //otherwise scattered value on upper diagonal grids will not be
               //the same as that of the duplicate grids
            }
            MPI_Pcontrol(-1, "gatherScatter");
	} // for (s...)
        
        if(rank == 0 && version == CHKPOINT_RESTART){
           singleCkptWriteTime = eCkptWrtTime - sCkptWrtTime;
           printf("[%d]----- Writing all checkpoints take %0.6f Sec (MPI_Wtime) -----\n",
                                rank,  totCkpts*singleCkptWriteTime);
        }        

	if (tempRank != -1) {
            if (not dbgGridComb) {
                advT = 0.0;
                advS = 0;
                dtC = tF - t;
                MPI_Barrier(mySubSparseGridComm);
                timer->start("simulateAdvection", u->l.prod(), 0);

                while (advT < dtC) {
                    if (isNewlySpawned != 1) { // not newly spawned
                        // call simulateAdvection
                        t += adv->simulateAdvection(u, g);
                        // update time and steps
                        advT += adv->dt;
                        advS++;
                    }
                    if (verbosity > 3) {
                        char s[64];
                        sprintf(s, "after time %.4e, field is:\n", t);
                        u->print(g->myrank, s);
                    }
                }
                timer->stop("simulateAdvection");
                MPI_Barrier(mySubSparseGridComm);
            } // end of "if (not dbgGridComb)"
	} // end of "if(tempRank != -1)"

	// in debug mode, final grid should be scaleFactor * its initial value
	double scaleFactor = std::pow((double) gc->nGrids(), (int) nC);
	if (nC > 0) {
            double sgError = 0.0;
            if (sg->myrank >= 0) // this process holds part of the sparse grid
                    sgError = dbgGridComb ? slg->checkError(scaleFactor, usg, sg) :
                              advsg->checkError(tC, usg, sg);
            printLocGlobAvgs("error of combined field", sgError, usg->l.prod(),
                            grid, myCommWorld);

            if (verbosity > 2 && sg->myrank >= 0)
               usg->print(rank, "combined field");
	}

	if (grank == 0 and not dbgGridComb)
           printf("%d: grid (%d,%d): final time: %e (target %e)\n", rank, myGrid.x,
				myGrid.y, t, tF);

	double uError = dbgGridComb ? ulg->checkError(scaleFactor, u, g) :
					adv->checkError(t, u, g);
	printLocGlobAvgs("error of final field", uError, nPtsL, myGrid, myGridComm);

	if (verbosity > 2)
	   u->print(rank, "final field");

	timer->dump(myCommWorld, verbTim); 
        // Enable this for time measurement and speedup calculation

	// memory release
	delete gc;
	delete adv;
	delete advsg;
	delete u;
	delete usg;
	delete ulg;
	delete slg;
	delete timer;

	// free allocated memory
	if (listFails != NULL) {
		free(listFails); // listFails available to both spawned and non-spawned processes
	}

	MPI_Comm_free(&myGridComm);
	MPI_Comm_free(&mySubSparseGridComm);

	endTime = MPI_Wtime();
	double diffTime = endTime - startTime, maxDiffTime;
	MPI_Reduce(&diffTime, &maxDiffTime, 1, MPI_DOUBLE, MPI_MAX, 0, myCommWorld);
	if (rank == 0) {
	   printf("\nMaximum execution time with MPI_Wtime (including application "
                  "initialization, memory allocation, etc.) is: %0.2f sec\n\n",
				maxDiffTime);
	}

	MPI_Barrier(myCommWorld);
	MPI_Finalize();
	return 0;
} //main()
