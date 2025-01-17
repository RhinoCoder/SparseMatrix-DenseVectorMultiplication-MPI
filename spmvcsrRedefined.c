#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <papi.h>
 
void GenerateElevenBandedCsrLocal(int rank,long long N,
                                  long long offset,
                                  long long local_n,
                                  long long **row_ptr,
                                  long long **col_ind,
                                  double **val)
{
    *row_ptr = (long long *)malloc((local_n + 1) * sizeof(long long));
    if (!(*row_ptr)) {
        fprintf(stderr, "Error: failed to allocate row_ptr.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    long long *nnz_per_row = (long long *)calloc(local_n, sizeof(long long));
    if (!nnz_per_row) {
        fprintf(stderr, "Error: failed to allocate nnz_per_row.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Count non-zeros for each local row
    
    long long i_local;
    for (i_local = 0; i_local < local_n; i_local++) {
        long long i_global = offset + i_local;
        long long start_col = (i_global - 5 < 0) ? 0 : (i_global - 5);
        long long end_col   = (i_global + 5 >= N) ? (N - 1) : (i_global + 5);
        nnz_per_row[i_local] = end_col - start_col + 1;
    }

 
    (*row_ptr)[0] = 0;
    long long rwPtrBuilderCount;
    for (rwPtrBuilderCount = 1; rwPtrBuilderCount <= local_n; rwPtrBuilderCount++) {
        (*row_ptr)[rwPtrBuilderCount] = (*row_ptr)[rwPtrBuilderCount - 1] + nnz_per_row[rwPtrBuilderCount - 1];
    }

    long long local_nnz = (*row_ptr)[local_n];
    *col_ind = (long long *)malloc(local_nnz * sizeof(long long));
    *val = (double *)malloc(local_nnz * sizeof(double));
    if (!(*col_ind) || !(*val))
    {
        fprintf(stderr, "Error: failed to allocate col_ind/val.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Fill col_ind and val (dummy random or banded pattern)
    long long i_local2;
    for (i_local2 = 0; i_local2< local_n; i_local2++) 
    {
        long long i_global = offset + i_local2;
        long long start_col = (i_global - 5 < 0) ? 0 : (i_global - 5);
        long long end_col   = (i_global + 5 >= N) ? (N - 1) : (i_global + 5);
        long long idx       = (*row_ptr)[i_local2];

        long long j;
        for (j = start_col; j <= end_col; j++)
        {
	   
    		if (j < 0 || j >= N) {
        	    fprintf(stderr, "Rank %d: col_ind[%lld] = %lld out of range during matrix generation!\n",rank, idx, j);
                printf("Rank %d: col_ind[%lld] = %lld out of range during matrix generation!\n",rank, idx, j);     
        	    MPI_Abort(MPI_COMM_WORLD, 1);
                 
	        }

            (*col_ind)[idx] = j;
    		(*val)[idx] = (double)(rand() % 1000);
    		idx++;

        }
    }

    free(nnz_per_row);
}

/* ---------------------------------------------------------------------
 * Local SpMV: multiplies the local banded block by the global x vector.
 * y_local has size local_n (the partial result).
 * ---------------------------------------------------------------------*/
void spmv_csr_local(long long local_n,
                    long long offset,
                    const long long *row_ptr,
                    const long long *col_ind,
                    const double *val,
                    const double *x,
                    double *y_local)
{
    long long i_local;
    for (i_local = 0; i_local < local_n; i_local++) {
        double sum = 0.0;
        long long start = row_ptr[i_local];
        long long end   = row_ptr[i_local + 1];
        
        long long k;
        for (k = start; k < end; k++) {
            long long j = col_ind[k];  // global column index
            sum += val[k] * x[j];
        }
        y_local[i_local] = sum;
    }
}



int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
   
    

    /*
    int papiVal = PAPI_library_init(PAPI_VER_CURRENT);	
    int events[2] = {PAPI_TOT_CYC,PAPI_L1_TCM};
    int EventSet = PAPI_NULL;
    long long values[2];

    if(papiVal != PAPI_VER_CURRENT)
    {
     	fprintf(stderr, "PAPI library initialization failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if(PAPI_create_eventset(&EventSet) != PAPI_OK)
    {
    	fprintf(stderr, "PAPI create event set failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }    

    if(PAPI_add_event(EventSet,PAPI_TOT_CYC) != PAPI_OK)
    {
       fprintf(stderr, "PAPI Add Event failed\n");
       MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if(PAPI_add_event(EventSet,PAPI_L1_TCM) !=PAPI_OK)
    {
       fprintf(stderr, "PAPI Add Event L1 TCM failed\n");
       MPI_Abort(MPI_COMM_WORLD, 1);
    }
    */

    if (argc < 2) 
    {
        if (rank == 0) 
        {
            printf("Usage: mpirun -np <procs> %s <N>\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    long long N = atoll(argv[1]);
    if (N <= 0) 
    {
        if (rank == 0) 
        {
            fprintf(stderr, "Error: N must be > 0.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    FILE *fp = NULL;

    if(rank == 0){

        fp = fopen("TermProjectResults.txt","w");
        if(!fp){
            fprintf(stderr,"Could not open output .txt file");
            MPI_Abort(MPI_COMM_WORLD,1);
        }

        fprintf(fp,"_____________________\n");
        fprintf(fp,"Term Project:\n");
        fprintf(fp,"_____________________\n");
        fprintf(fp,"Eleven Banded Non Symmetric Sparse Matrix - Dense Vector Multiplication \n\n\n");
        

    }

    // Row-block partition
    long long base = N / size;
    long long rem  = N % size;
    long long local_n = base + ((rank < rem) ? 1 : 0);

    // Offset of first local row for this rank
    long long offset = 0;
    int r;
    for (r = 0; r < rank; r++) 
    {
        offset += base + ((r < rem) ? 1 : 0);
    }

    srand((unsigned)time(NULL) + rank * 5843);
    printf("Rank %d: offset = %lld, local_n = %lld\n", rank, offset, local_n);

    // Generate local portion of the banded matrix
    long long *row_ptr_local = NULL;
    long long *col_ind_local = NULL;
    double    *val_local     = NULL;
    GenerateElevenBandedCsrLocal(rank,N, offset, local_n,&row_ptr_local, &col_ind_local, &val_local);
    //print_memory_usage(rank,fp);	
    // Allocate x (the global vector). Each rank needs the entire x.
    double *x = (double *)malloc(N * sizeof(double));
    if (!x) {
        fprintf(stderr, "[Rank %d] failed to allocate x.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    printf("Rank %d: local_n = %lld\n", rank, local_n);

    //print_memory_usage(rank,fp);
    // Rank 0 initializes x (random or otherwise)
    if (rank == 0) 
    {
        long long total_rows = 0;
	    int r;
	    for (r = 0; r < size; r++) {
  		    long long r_local_n = base + ((r < rem) ? 1 : 0);
 	 	    total_rows += r_local_n;
	    }
	printf("Total of local_n across all ranks = %lld (should be == N)\n", total_rows);

        long long i;
        for (i = 0; i < N; i++) {
            x[i] = (double)(rand() % 1000);
            
        }
    }
   
    MPI_Barrier(MPI_COMM_WORLD);
    // ----------------------------------------
    // Non-blocking Broadcast of x
    // ----------------------------------------
    MPI_Request req_bcast;
    //int bCode = MPI_Bcast(x, (int)N, MPI_DOUBLE, 0, MPI_COMM_WORLD, &req_bcast);
    int bCode = MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(bCode != MPI_SUCCESS)
    {
        char errMsg[MPI_MAX_ERROR_STRING];
        int msgLen;
        MPI_Error_string(bCode,errMsg,&msgLen);
        fprintf(stderr,"Error happened in IGatherv at rank: %d",rank);
        MPI_Abort(MPI_COMM_WORLD,bCode);
        return EXIT_FAILURE;
    }

    //MPI_Wait(&req_bcast, MPI_STATUS_IGNORE); // Ensure the broadcast is completed before proceeding.
     
    
    double *y_local = (double *)calloc(local_n, sizeof(double));
    if (!y_local) {
        fprintf(stderr, "[Rank %d] failed to allocate y_local.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

 
    //MPI_Wait(&req_bcast, MPI_STATUS_IGNORE); 

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    
    /*
    if(PAPI_start(EventSet) != PAPI_OK)
    {   
        fprintf(stderr, "PAPI_start_counters error\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    */

    // Local SpMV
    spmv_csr_local(local_n, offset, row_ptr_local, col_ind_local, val_local, x, y_local);

    /*
    if(PAPI_stop(EventSet,values) != PAPI_OK)
    {
        fprintf(stderr, "PAPI_stop_counters error\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    */
    // End time
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double local_ms = (t1 - t0) * 1000.0;

    // ----------------------------------------
    // Non-blocking Gatherv of local y
    // ----------------------------------------
    double *y_global  = NULL;
    int   *recvcounts = NULL;
    int   *displs     = NULL;

    if (rank == 0) 
    {
        y_global   = (double *)malloc(N * sizeof(double));
        recvcounts = (int *)malloc(size * sizeof(int));
        displs     = (int *)malloc(size * sizeof(int));

        long long disp = 0;
        
        int processors;

        for (processors = 0; processors < size; processors++)
        {
            long long rr_local_n = base + ((processors < rem) ? 1 : 0);
            recvcounts[processors] = (int)rr_local_n;
            displs[processors]     = (int)disp;
            disp += rr_local_n;
        }

          fprintf(fp,"recvcounts and displs:\n");
            int iterator;
            for (iterator = 0; iterator < size; iterator++)
            {
                fprintf(fp,"Rank %d: recvcounts = %d, displs = %d\n", iterator, recvcounts[iterator], displs[iterator]);
            }    
    }

    MPI_Request req_gath;
    MPI_Barrier(MPI_COMM_WORLD);
    //int gatherCode = MPI_Gatherv(y_local, (int)local_n, MPI_DOUBLE,y_global, recvcounts, displs, MPI_DOUBLE,0, MPI_COMM_WORLD, &req_gath);
    int gatherCode = MPI_Gatherv(y_local, (int)local_n, MPI_DOUBLE,y_global, recvcounts, displs, MPI_DOUBLE,0, MPI_COMM_WORLD);
   
    if(gatherCode != MPI_SUCCESS)
    {
        char errMsg[MPI_MAX_ERROR_STRING];
        int msgLen;
        MPI_Error_string(gatherCode,errMsg,&msgLen);
        fprintf(stderr,"Error happened in IGatherv at rank: %d",rank);
        MPI_Abort(MPI_COMM_WORLD,gatherCode);
        return EXIT_FAILURE;
    }

    //MPI_Wait(&req_gath, MPI_STATUS_IGNORE);

    if (rank == 0) 
    {
        fprintf(fp,"Succesfully Completed....\n");
        fprintf(fp,"Showing results.....\n\n");
        fprintf(fp,"...............................................\n");   
        fprintf(fp,".....SpMV (Row-block) with Size N=%lld, Proc Count=%d .....\n", N, size);
        fprintf(fp,"...............................................\n\n");        
        fprintf(fp,"Elapsed time @ [rank 0]: %.3f ms\n\n\n", local_ms);
        fprintf(fp,"...............................................\n");        
        fprintf(fp,"PAPI Profiling Results\n");        
        fprintf(fp,"...............................................\n\n");         
        //fprintf(fp, "Total Cycles: %lld\n", values[0]);
        //fprintf(fp, "L1 Cache Misses: %lld\n", values[1]);
        //fprintf(fp, "Floating Point Operations: %lld\n", values[2]);

        int i;
        for (i = 0; i < 5 && i < N; i++)
        {
            fprintf(fp,"y[%d] = %f\n", i, y_global[i]);
        }
        fclose(fp);
        free(y_global);
        free(recvcounts);
        free(displs);
    }

    free(y_local);
    free(x);
    free(row_ptr_local);
    free(col_ind_local);
    free(val_local);

    MPI_Finalize();
    return 0;
}
