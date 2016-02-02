.SUFFIXES:
.PRECIOUS: %.o
	
HDRS=FTAdvect3D.h FTHaloArray3D.h ProcGrid3D.h Vec3D.h FTGridCombine3D.h \
	LinGrid3D.h Timer.h FailureRecovery.h
OBJS=FTAdvect3D.o FTGridCombine3D.o
PROG=FTthreeDimAdvect
CCFLAGS=-g -O2 -fopenmp

all: clean $(PROG) XHELP
%: %.o $(OBJS)
	mpic++ -o $* $*.o $(OBJS) -lgomp
%.o: %.cpp $(HDRS)
	mpic++ -w $(CCFLAGS) -c $*.cpp	
XHELP: 
	@echo " "
	@echo "Make completed successfully!"
	@echo " "
	@echo "Now adjust settings in \"run3dAdvectRaijin\" file and"
	@echo "run \"./FTthreeDimAdvect -h\" to get help how to execute"	
	@echo "(make sure that \"RUN_ON_COMPUTE_NODES\" is defined "
	@echo "and undefine or define \"RECOV_ON_SPARE_NODES\""
	@echo "in \"FailureRecovery.h\" file for cluster run)"
	@echo "Also check and edit \"~/.bashrc\" file if not working"
	@echo " "
	
clean:
	rm -f *.o $(PROG)

run2d:
	./run3dAdvectRaijin -v 0 -f -p 8 -q 4 -r 2 -l 4 8 8 0 #only 2D

