#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <papi.h>


void GenerateElevenBandedCsrLocal(long long N,long long offset,long long local_n,long long **row_ptr,long long **col_ind,double **val)
{
    *row_ptr = (long long *)malloc((local_n + 1) * sizeof(long long));
    if (!(*row_ptr))
    {
        fprintf(stderr, "Failed to allocate row_ptr\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    long long *nnz_per_row = (long long *)calloc(local_n, sizeof(long long));
    if (!nnz_per_row)
    {
        fprintf(stderr, "Failed to allocate nnz_per_row.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    long long i_local;    
    for (i_local = 0; i_local < local_n; i_local++) 
    {
        long long i_global = offset + i_local;
        long long start_col;
        long long end_col;
        
        if(i_global - 5 < 0)
        {
            start_col = 0;
        }
        
        else
        {
            start_col = i_global - 5;
        }
        
        if(i_global + 5 >= N)
        {
            end_col = N-1;
        }
        else
        {
            end_col = i_global + 5;
        }
        nnz_per_row[i_local] = end_col - start_col + 1;
    }


    (*row_ptr)[0] = 0;
    long long i_local2;
    for (i_local2 = 1; i_local2 <= local_n; i_local2++)
    {
        (*row_ptr)[i_local2] = (*row_ptr)[i_local2 - 1] + nnz_per_row[i_local2 - 1];
    }

    long long local_nnz = (*row_ptr)[local_n];
    *col_ind = (long long *)malloc(local_nnz * sizeof(long long));
    *val= (double *)malloc(local_nnz * sizeof(double));
    
    if (!(*col_ind) || !(*val)) 
    {
        fprintf(stderr, "Failed to allocate col_ind/val on rank.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    long long i_local3;
    for (i_local3 = 0; i_local3 < local_n; i_local3++)
    {
        long long i_global = offset + i_local3;
        long long start_col = (i_global - 5 < 0) ? 0 : (i_global - 5);
        long long end_col   = (i_global + 5 >= N) ? (N - 1) : (i_global + 5);

        long long idx = (*row_ptr)[i_local3];
        long long j;
        for (j = start_col; j <= end_col; j++)
        {
            (*col_ind)[idx] = j;
            if (j < i_global)
            {
                (*val)[idx] = (double)(rand() % 1000 + i_global + j + 1);
            }
            else 
            {
                (*val)[idx] = (double)(rand() % 1000 - i_global + j + 1);
            }
            idx++;
        }
    }

    free(nnz_per_row);
}

void spmv_csr_local(long long local_n,long long offset,const long long *row_ptr,const long long *col_ind,const double *val,const double *x,double *y_local) 
{

    long long i_local;
    for (i_local = 0; i_local < local_n; i_local++) 
    {
        double sum = 0.0;
        long long start = row_ptr[i_local];
        long long end   = row_ptr[i_local + 1];

        long long k;
        for (k = start; k < end; k++) 
        {
            long long j = col_ind[k];
            sum += val[k] * x[j];
        }
        y_local[i_local] = sum;
    }
}

int main(int argc, char *argv[])
{

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


   /*  int papiVal = PAPI_library_init(PAPI_VER_CURRENT);	
    int events[2] = {PAPI_TOT_CYC,PAPI_L1_TCM};
    int EventSet = PAPI_NULL;
    long long values[2];

    if(papiVal != PAPI_VER_CURRENT)
    {
     	fprintf(stderr, "PAPI library initialization failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }


    if(PAPI_create_eventset(&EventSet) != PAPI_OK) {
    	fprintf(stderr, "PAPI create event set failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }    

    if(PAPI_add_event(EventSet,PAPI_TOT_CYC) != PAPI_OK){
       fprintf(stderr, "PAPI Add Event failed\n");
       MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if(PAPI_add_event(EventSet,PAPI_L1_TCM) !=PAPI_OK){
       fprintf(stderr, "PAPI Add Event L1 TCM failed\n");
       MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /*
    if(PAPI_add_event(EventSet,PAPI_FP_OPS) !=PAPI_OK){
       fprintf(stderr, "PAPI Add Event PAPI_FP_OPS failed\n");
       MPI_Abort(MPI_COMM_WORLD, 1);
    }
    */ 
    


    if (argc < 2) 
    {
        if (rank == 0) 
        {
            printf("Usage: mpirun -np <num_procs> %s <N1> [N2 N3 ...]\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    FILE *fp = NULL;
    if (rank == 0) 
    {
        fp = fopen("TermProjectResults.txt", "w");
        if (!fp) 
        {
            fprintf(stderr, "Could not open TermProjectResults.txt for writing.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(fp, "Term Project Results\n");
        fprintf(fp, "====================\n\n");
    }

    int arg_i;
    for (arg_i = 1; arg_i < argc; arg_i++) 
    {
        long long N = atoll(argv[arg_i]);
        if (N <= 0) 
        {
            if (rank == 0) 
            {
                fprintf(stderr, "Error: N must be positive (got %lld). Skipping...\n", N);
            }
            continue;
        }

        srand((unsigned)time(NULL) + rank * 5843  * arg_i);
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        long long base = N / size;      
        long long rem  = N % size;      
        long long local_n = base + ((rank < rem) ? 1 : 0);

        
        long long offset = 0;
        int r;
        for (r = 0; r < rank; r++) 
        {
            offset += base + ((r < rem) ? 1 : 0);
        }

        long long *row_ptr_local = NULL;
        long long *col_ind_local = NULL;
        double *val_local  = NULL;

        GenerateElevenBandedCsrLocal(N, offset, local_n,&row_ptr_local, &col_ind_local, &val_local);
    
        double *x = (double *)malloc(N * sizeof(double));
        if (x == NULL)
        {
            fprintf(stderr, "Memory allocation failed for x.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (rank == 0)
        {
            long long i;
            for (i = 0; i < N; i++)
            {
                x[i] = (double)(rand() % 1000);
            }            
            
        }
        
        int bCode = MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if(bCode != MPI_SUCCESS)
        {
            char errMsg[MPI_MAX_ERROR_STRING];
            int msgLen;
            MPI_Error_string(bCode,errMsg,&msgLen);
            fprintf(stderr, "[Rank %d] MPI_Bcast Error: %s\n", rank, errMsg);
            MPI_Abort(MPI_COMM_WORLD,bCode);
            return EXIT_FAILURE;
        }

   
        
        /* 
        if(PAPI_start(EventSet) != PAPI_OK)
        {
            fprintf(stderr, "PAPI_start_counters error\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        } */

        double *y_local = (double *)calloc(local_n, sizeof(double));
        spmv_csr_local(local_n, offset, row_ptr_local, col_ind_local, val_local, x, y_local);

       /*  if(PAPI_stop(EventSet,values) != PAPI_OK)
        {
            fprintf(stderr, "PAPI_stop_counters error\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        } */

        double *y_global = NULL;
        

        if (rank == 0) 
        {
            y_global = (double *)malloc(N * sizeof(double));
        }

        int *recvcounts= NULL;
        int *displs= NULL;
        
        if (rank == 0)
        {
            recvcounts = (int *)malloc(size * sizeof(int));
            displs     = (int *)malloc(size * sizeof(int));
            int disp = 0;
            int nm;
            for (nm = 0; nm < size; nm++)
            {
                long long r_local_n = base + ((nm < rem) ? 1 : 0);
                recvcounts[nm]= r_local_n;
                displs[nm]= disp;
                disp+= r_local_n;
            }
        }

        

        
        int gCode = MPI_Gatherv(y_local, local_n, MPI_DOUBLE,y_global, recvcounts, displs, MPI_DOUBLE,0, MPI_COMM_WORLD);
        if(gCode != MPI_SUCCESS)
        {
            char errMsg[MPI_MAX_ERROR_STRING];
            int msgLen;
            MPI_Error_string(gCode,errMsg,&msgLen);
            fprintf(stderr, "[Rank %d] MPI_Bcast Error: %s\n", rank, errMsg);
            MPI_Abort(MPI_COMM_WORLD,gCode);
            return EXIT_FAILURE;
        }

        
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        double elapsed = (t1 - t0) * 1000;

        if (rank == 0)
        {
        
            int x;
            fprintf(fp, "Test for N = %lld\n", N);
            fprintf(fp, "----------------\n");
            //fprintf(fp, "Total Cycles: %lld\n", values[0]);
            //fprintf(fp, "L1 Cache Misses: %lld\n", values[1]);
            //fprintf(fp, "Floating Point Operations: %lld\n", values[2]);
            fprintf(fp, "----------------\n");
            fprintf(fp, "Elapsed time: %.3f ms\n", elapsed);
            fprintf(fp, "First %d entries of result vector:\n",x);
            for (x = 0; x < 15; x++){fprintf(fp, "y[%d] = %f\n", x, y_global[x]);} 
            fprintf(fp, "\n");
        }

        free(x);
        free(y_local);
        free(row_ptr_local);
        free(col_ind_local);
        free(val_local);

        if (rank == 0) 
        {
            free(y_global);
            free(recvcounts);
            free(displs);
        }
    }
    
 

    if (rank == 0 && fp){fclose(fp);}
    //PAPI_shutdown();
    MPI_Finalize();
    return 0;
}
