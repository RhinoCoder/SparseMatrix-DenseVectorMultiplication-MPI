#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <papi.h>

void GenerateElevenBandedCsrLocal(long long rank, long long N,
                                  long long offset,
                                  long long local_n,
                                  long long **row_ptr,
                                  long long **col_ind,
                                  double **val)
{
    /* row_ptr has size (local_n + 1) */
    *row_ptr = (long long *)malloc((local_n + 1) * sizeof(long long));
    if (!(*row_ptr))
    {
        fprintf(stderr, "[Rank %lld] Error: failed to allocate row_ptr.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Temporary array to store # of non-zeros per row. */
    long long *nnz_per_row = (long long *)calloc(local_n, sizeof(long long));
    if (!nnz_per_row)
    {
        fprintf(stderr, "[Rank %lld] Error: failed to allocate nnz_per_row.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Count non-zeros for each local row. */
    long long i_local;
    for (i_local = 0; i_local < local_n; i_local++)
    {
        long long i_global = offset + i_local;
        long long start_col = (i_global - 5 < 0) ? 0 : (i_global - 5);
        long long end_col = (i_global + 5 >= N) ? (N - 1) : (i_global + 5);
        nnz_per_row[i_local] = end_col - start_col + 1;
    }

    /* Exclusive prefix sum to build row_ptr. */
    (*row_ptr)[0] = 0;
    long long i;
    for (i = 1; i <= local_n; i++)
    {
        (*row_ptr)[i] = (*row_ptr)[i - 1] + nnz_per_row[i - 1];
    }

    long long local_nnz = (*row_ptr)[local_n];
    *col_ind = (long long *)malloc(local_nnz * sizeof(long long));
    *val = (double *)malloc(local_nnz * sizeof(double));
    if (!(*col_ind) || !(*val))
    {
        fprintf(stderr, "[Rank %lld] Error: failed to allocate col_ind/val.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Fill col_ind and val with a (dummy) random banded pattern. */
    long long i_local2;
    for (i_local2 = 0; i_local2 < local_n; i_local2++)
    {
        long long i_global = offset + i_local2;
        long long start_col = (i_global - 5 < 0) ? 0 : (i_global - 5);
        long long end_col = (i_global + 5 >= N) ? (N - 1) : (i_global + 5);
        long long idx = (*row_ptr)[i_local2];

        long long j;
        for (j = start_col; j <= end_col; j++)
        {
            if (j < 0 || j >= N)
            {
                fprintf(stderr, "[Rank %lld] col_ind[%lld] = %lld out of range!\n", rank, idx, j);
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
 * Local SpMV: multiplies the local banded block by the local slice of x.
 * We need to map the global column index j to local sub-vector index j_sub.
 * ---------------------------------------------------------------------*/
void spmv_csr_local(long long local_n,
                    long long offset,
                    long long halo_start,
		    long long local_x_size, 
                    const long long *row_ptr,
                    const long long *col_ind,
                    const double *val,
                    const double *x_sub,
                    double *y_local)
{
    long long i_local;
    for (i_local = 0; i_local < local_n; i_local++)
    {
        double sum = 0.0;
        long long start = row_ptr[i_local];
        long long end = row_ptr[i_local + 1];

        long long k;
        for (k = start; k < end; k++)
        {
            long long j_global = col_ind[k];
            /* Convert to local index for x_sub. */
            long long j_sub = j_global - halo_start;
	    if(j_sub >= 0 && j_sub < local_x_size)
	    {
            	sum += val[k] * x_sub[j_sub];
            }
	
 	}
        y_local[i_local] = sum;
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank_int, size_int;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_int);
    MPI_Comm_size(MPI_COMM_WORLD, &size_int);

    /* Use 64-bit rank if needed, but typically int is fine for rank. */
    long long rank = (long long)rank_int;
    long long size = (long long)size_int;

    if (argc < 2)
    {
        if (rank == 0)
        {
            printf("Usage: srun -n <procs> %s <N>\n", argv[0]);
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

    fp = fopen("TermProjectResults.txt", "w");
    if (!fp)
    {
        fprintf(stderr, "Could not open output .txt file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fprintf(fp, "_____________________\n");
    fprintf(fp, "Term Project:\n");
    fprintf(fp, "_____________________\n\n");
    fprintf(fp, "> Eleven Banded Non Symmetric Sparse Matrix - Dense Vector Multiplication \n\n");

    /* Partition rows in a round-robin manner. */
    long long base = N / size;
    long long rem = N % size;
    long long local_n = base + ((rank < rem) ? 1 : 0);

    /* Offset of the first local row for this rank. */
    long long offset = 0;
    long long r;
    for (r = 0; r < rank; r++)
    {
        offset += base + ((r < rem) ? 1 : 0);
    }

    srand((unsigned)time(NULL) + (unsigned)rank * 5843);

    /* Generate local portion of the banded matrix. */
    long long *row_ptr_local = NULL;
    long long *col_ind_local = NULL;
    double *val_local = NULL;
    GenerateElevenBandedCsrLocal(rank, N, offset, local_n, &row_ptr_local, &col_ind_local, &val_local);

    // Debug purposes, open it.
    if (rank == 0)
    {
        fprintf(fp, "N = %lld, size = %lld\n", N, size);
    }

    /* Synchronize all ranks before writing */
    MPI_Barrier(MPI_COMM_WORLD);



    long long halo_start = (offset - 5) < 0 ? 0 : (offset - 5);
    long long halo_end = (offset + local_n - 1 + 5) >= (N - 1) ? (N - 1) : (offset + local_n - 1 + 5);
    long long local_x_size = halo_end - halo_start + 1;

    double *x_sub = (double *)malloc(local_x_size * sizeof(double));
    if (!x_sub)
    {
        fprintf(stderr, "[Rank %lld] Error: cannot allocate x_sub of size %lld\n", rank, local_x_size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    long long xSubLen;
    for (xSubLen = 0; xSubLen < local_x_size; xSubLen++)
    {
        x_sub[xSubLen] = (double)(rand() % 1000);
    }

    if (rank == 0)
    {

        /* For each rank, compute its halo_start/halo_end and store in sendcounts. */
        long long offset_r = 0;
        long long rr;
        for (rr = 0; rr < size; rr++)
        {
            long long local_n_r = base + ((rr < rem) ? 1 : 0);
            long long halo_start_r = (offset_r - 5) < 0 ? 0 : (offset_r - 5);
            long long halo_end_r = (offset_r + local_n_r - 1 + 5) >= (N - 1)
                                       ? (N - 1)
                                       : (offset_r + local_n_r - 1 + 5);
            long long local_x_size_r = halo_end_r - halo_start_r + 1;

            offset_r += local_n_r;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* Now each rank has x_sub = x[halo_start .. halo_end]. */
    double *y_local = (double *)calloc(local_n, sizeof(double));
    if (!y_local)
    {
        fprintf(stderr, "[Rank %lld] failed to allocate y_local.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Time the local SpMV. */
    double t0 = MPI_Wtime();
    spmv_csr_local(local_n, offset, halo_start,local_x_size, row_ptr_local, col_ind_local, val_local, x_sub, y_local);
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double local_ms = (t1 - t0) * 1000.0;

    /* ----------------------------------------
     * Gatherv local y back to rank 0.
     * If you REALLY want all of y on rank 0,
     * you'll need an array of length N for it.
     * This can be huge (8GB if N=1e9).
     * ----------------------------------------*/
    double *y_global = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == 0)
    {
        y_global = (double *)malloc(N * sizeof(double));
        recvcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));

        /* Recompute row counts for each rank. */
        long long offset_r = 0;
        long long rr;
        for (rr = 0; rr < size; rr++)
        {
            long long local_n_r = base + ((rr < rem) ? 1 : 0);
            recvcounts[rr] = (int)local_n_r;
            displs[rr] = (int)offset_r;
            offset_r += local_n_r;
        }
    }

    int gatherCode = MPI_Gatherv(y_local, (int)local_n, MPI_DOUBLE, y_global, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (gatherCode != MPI_SUCCESS)
    {
        char errMsg[MPI_MAX_ERROR_STRING];
        int msgLen;
        MPI_Error_string(gatherCode, errMsg, &msgLen);
        fprintf(stderr, "Error happened in Gatherv at rank: %lld", rank);
        MPI_Abort(MPI_COMM_WORLD, gatherCode);
        return EXIT_FAILURE;
    }

    // Debug Purposes.
    /*
    long long resultIterator;
    for (resultIterator = 0; resultIterator < local_x_size; resultIterator++)
    {
        printf("[Rank %lld] x_sub[%lld] = %.2f\n", rank, resultIterator + halo_start, x_sub[resultIterator]);
    }
    */

    /* Rank 0 can do final reporting. */
    if (rank == 0)
    {

        fprintf(fp, "> Showing results.....\n\n");
        fprintf(fp, "y_global: ");
        long long i;
        for (i = 0; i < 5; i++)
        {
            fprintf(fp, "%.2f ", y_global[i]);
        }
        fprintf(fp, "\n");
        fprintf(fp, "...............................................\n");
        fprintf(fp, ".....SpMV (Row-block) with Size N=%lld, Proc Count=%lld .....\n", N, size);
        fprintf(fp, "...............................................\n\n");
        fprintf(fp, "Elapsed time @ [rank 0]: %.3f ms\n\n\n", local_ms);
        fprintf(fp, "...............................................\n");
        fprintf(fp, "PAPI Profiling Results\n");
        fprintf(fp, "...............................................\n\n");
        // fprintf(fp, "Total Cycles: %lld\n", values[0]);
        // fprintf(fp, "L1 Cache Misses: %lld\n", values[1]);
        // fprintf(fp, "Floating Point Operations: %lld\n", values[2]);

        free(y_global);
        free(recvcounts);
        free(displs);
    }

    /* Free local data. */
    free(y_local);
    free(x_sub);
    free(row_ptr_local);
    free(col_ind_local);
    free(val_local);
    fclose(fp);
    MPI_Finalize();
    return 0;
}
