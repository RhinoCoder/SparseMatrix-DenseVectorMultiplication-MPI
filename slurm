

Debug: Parsing with PDT Parser.
Executing> /okyanus/users/abilgin/hbm513e/pdtoolkit-3.25.2/x86_64/bin/cparse redspmvcsr.c -I/okyanus/users/abilgin/hbm513e/tau-2.34/include -DPROFILING_ON -DTAU_PAPI -I/okyanus/users/abilgin/hbm513e/papi-install/include -DTAU_GNU -DTAU_DOT_H_LESS_HEADERS -DTAU_MPI -DTAU_UNIFY -DTAU_MPI_THREADED -DTAU_LINUX_TIMERS -DTAU_MPIGREQUEST -DTAU_MPIDATAREP -DTAU_MPIERRHANDLER -DTAU_MPICONSTCHAR -DTAU_MPIATTRFUNCTION -DTAU_MPITYPEEX -DTAU_MPIADDERROR -DTAU_LARGEFILE -D_LARGEFILE64_SOURCE -DTAU_BFD -DTAU_MPIFILE -DHAVE_GNU_DEMANGLE -DHAVE_TR1_HASH_MAP -DTAU_SS_ALLOC_SUPPORT -DEBS_CLOCK_RES=1 -DTAU_STRSIGNAL_OK -DTAU_TRACK_LD_LOADER -DTAU_MPICH3 -DTAU_OPENMPI3 -DTAU_MPI_EXTENSIONS -DTAU_ELF_BFD -I/okyanus/progs/openmpi/openmpi-3.1.6-gcc-9.5.0/include -I/okyanus/progs/openmpi/openmpi-3.1.6-gcc-9.5.0/include/openmpi -I/okyanus/progs/openmpi/openmpi-3.1.6-gcc-9.5.0/include/openmpi/ompi -I/okyanus/progs/openmpi/openmpi-3.1.6-gcc-9.5.0/lib -I/home/abilgin/hbm513e/papi-install/include -I/okyanus/users/abilgin/hbm513e/tau-2.34/include -I/okyanus/progs/openmpi/openmpi-3.1.6-gcc-9.5.0/include -I/okyanus/progs/openmpi/openmpi-3.1.6-gcc-9.5.0/include/openmpi -I/okyanus/progs/openmpi/openmpi-3.1.6-gcc-9.5.0/include/openmpi/ompi -I/okyanus/progs/openmpi/openmpi-3.1.6-gcc-9.5.0/lib


Debug: Instrumenting with TAU
Executing> /okyanus/users/abilgin/hbm513e/tau-2.34/x86_64/bin/tau_instrumentor redspmvcsr.pdb redspmvcsr.c -o redspmvcsr.inst.c -c


Debug: Compiling (Individually) with Instrumented Code
Executing> /okyanus/progs/openmpi/openmpi-3.1.6-gcc-9.5.0/bin/mpicc -lm -L/home/abilgin/hbm513e/papi-install/lib -lpapi -I. -c redspmvcsr.inst.c -I/home/abilgin/hbm513e/papi-install/include -DPROFILING_ON -DTAU_PAPI -I/okyanus/users/abilgin/hbm513e/papi-install/include -DTAU_GNU -DTAU_DOT_H_LESS_HEADERS -DTAU_MPI -DTAU_UNIFY -DTAU_MPI_THREADED -DTAU_LINUX_TIMERS -DTAU_MPIGREQUEST -DTAU_MPIDATAREP -DTAU_MPIERRHANDLER -DTAU_MPICONSTCHAR -DTAU_MPIATTRFUNCTION -DTAU_MPITYPEEX -DTAU_MPIADDERROR -DTAU_LARGEFILE -D_LARGEFILE64_SOURCE -DTAU_BFD -DTAU_MPIFILE -DHAVE_GNU_DEMANGLE -DHAVE_TR1_HASH_MAP -DTAU_SS_ALLOC_SUPPORT -DEBS_CLOCK_RES=1 -DTAU_STRSIGNAL_OK -DTAU_TRACK_LD_LOADER -DTAU_MPICH3 -DTAU_OPENMPI3 -DTAU_MPI_EXTENSIONS -DTAU_ELF_BFD -I/okyanus/users/abilgin/hbm513e/tau-2.34/include -I/okyanus/progs/openmpi/openmpi-3.1.6-gcc-9.5.0/include -I/okyanus/progs/openmpi/openmpi-3.1.6-gcc-9.5.0/include/openmpi -I/okyanus/progs/openmpi/openmpi-3.1.6-gcc-9.5.0/include/openmpi/ompi -I/okyanus/progs/openmpi/openmpi-3.1.6-gcc-9.5.0/lib -o redspmvcsr.o


Debug: Linking (Together) object files
Executing> /okyanus/progs/openmpi/openmpi-3.1.6-gcc-9.5.0/bin/mpicc redspmvcsr.o -lm -L/home/abilgin/hbm513e/papi-install/lib -lpapi -L/okyanus/users/abilgin/hbm513e/tau-2.34/x86_64/lib -lTauMpi-papi-mpi-pdt -L/okyanus/progs/openmpi/openmpi-3.1.6-gcc-9.5.0/lib -lmpi -Wl,-rpath,/okyanus/progs/openmpi/openmpi-3.1.6-gcc-9.5.0/lib -L/okyanus/users/abilgin/hbm513e/tau-2.34/x86_64/lib -ltau-papi-mpi-pdt -Wl,-rpath,/okyanus/users/abilgin/hbm513e/papi-install/lib -L/okyanus/users/abilgin/hbm513e/papi-install/lib/ -L/okyanus/users/abilgin/hbm513e/papi-install/lib64/ -Wl,-rpath,/okyanus/users/abilgin/hbm513e/papi-install/lib -lpapi -L/okyanus/users/abilgin/hbm513e/tau-2.34/x86_64/binutils-2.40/lib -L/okyanus/users/abilgin/hbm513e/tau-2.34/x86_64/binutils-2.40/lib64 -Wl,-rpath,/okyanus/users/abilgin/hbm513e/tau-2.34/x86_64/binutils-2.40/lib -Wl,-rpath,/okyanus/users/abilgin/hbm513e/tau-2.34/x86_64/binutils-2.40/lib64 -lbfd -Wl,--export-dynamic -lrt -L/okyanus/progs/openmpi/openmpi-3.1.6-gcc-9.5.0/lib -lmpi -Wl,-rpath,/okyanus/progs/openmpi/openmpi-3.1.6-gcc-9.5.0/lib -ldl -lm -L/okyanus/progs/gcc/GCC-9.5.0/lib/gcc/x86_64-pc-linux-gnu/9.5.0/ -lstdc++ -lgcc_s -L/okyanus/users/abilgin/hbm513e/tau-2.34/x86_64/lib/static-papi-mpi-pdt -o redspmvcsr.exe


Debug: cleaning inst file
Executing> /bin/rm -f redspmvcsr.inst.c


Debug: cleaning PDB file
Executing> /bin/rm -f redspmvcsr.pdb

Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5
Python 2.7.5