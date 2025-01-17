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
                    long long halo_start, /* global index of x_sub[0] */
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
            sum += val[k] * x_sub[j_sub];
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
    fprintf(fp, "_____________________\n");
    fprintf(fp, "Eleven Banded Non Symmetric Sparse Matrix - Dense Vector Multiplication \n\n\n");

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

    fprintf(fp, "N = %lld, size = %lld\n", N, size);
    printf("N = %lld, size = %lld\n", N, size);

    /* Generate local portion of the banded matrix. */
    long long *row_ptr_local = NULL;
    long long *col_ind_local = NULL;
    double *val_local = NULL;
    GenerateElevenBandedCsrLocal(rank, N, offset, local_n,
                                 &row_ptr_local, &col_ind_local, &val_local);

    fprintf(fp, "[Rank %lld/%lld] offset=%lld, local_n=%lld\n", rank, size, offset, local_n);
    printf("[Rank %lld/%lld] offset=%lld, local_n=%lld\n", rank, size, offset, local_n);

    long long halo_start = (offset - 5) < 0 ? 0 : (offset - 5);
    long long halo_end = (offset + local_n - 1 + 5) >= (N - 1)
                             ? (N - 1)
                             : (offset + local_n - 1 + 5);

    long long local_x_size = halo_end - halo_start + 1;

    double *x_sub = (double *)malloc(local_x_size * sizeof(double));
    if (!x_sub)
    {
        fprintf(stderr, "[Rank %lld] Error: cannot allocate x_sub of size %lld\n", rank, local_x_size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double *x_global = NULL;
    int *sendcounts = NULL;
    int *sdispls = NULL;
    if (rank == 0)
    {
        /* Allocate entire x. ~8 bytes * N for doubles. */
        x_global = (double *)malloc(N * sizeof(double));
        if (!x_global)
        {
            fprintf(stderr, "[Rank 0] Error: cannot allocate x_global of size %lld\n", N);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* Initialize x_global with random values. */
        long long xLen;
        for (xLen = 0; xLen < N; xLen++)
        {
            x_global[xLen] = (double)(rand() % 1000);
        }

        /* Prepare arrays for Scatterv. */
        sendcounts = (int *)malloc(size * sizeof(int));
        sdispls = (int *)malloc(size * sizeof(int));

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

            sendcounts[rr] = (int)local_x_size_r; /* watch out for 2^31 limit! */
            if (rr == 0)
            {
                sdispls[rr] = 0;
            }
            else
            {
                sdispls[rr] = sdispls[rr - 1] + sendcounts[rr - 1];
            }
            offset_r += local_n_r;
        }
    }

    /* Scatterv the relevant slices of x to each rank's x_sub. */
    int sCode = MPI_Scatterv(x_global, sendcounts, sdispls, MPI_DOUBLE, x_sub, (int)local_x_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (sCode != MPI_SUCCESS)
    {
        char errMsg[MPI_MAX_ERROR_STRING];
        int msgLen;
        MPI_Error_string(sCode, errMsg, &msgLen);
        fprintf(stderr, "Error happened in IGatherv at rank: %lld", rank);
        MPI_Abort(MPI_COMM_WORLD, sCode);
        return EXIT_FAILURE;
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
    spmv_csr_local(local_n, offset, halo_start, row_ptr_local, col_ind_local, val_local, x_sub, y_local);
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
        int disp_cur = 0;
        long long rr;
        for (rr = 0; rr < size; rr++)
        {
            long long local_n_r = base + ((rr < rem) ? 1 : 0);
            recvcounts[rr] = (int)local_n_r;
            displs[rr] = disp_cur;
            disp_cur += local_n_r;
            offset_r += local_n_r;
        }
    }

    int gatherCode = MPI_Gatherv(y_local, (int)local_n, MPI_DOUBLE, y_global, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (gatherCode != MPI_SUCCESS)
    {
        char errMsg[MPI_MAX_ERROR_STRING];
        int msgLen;
        MPI_Error_string(gatherCode, errMsg, &msgLen);
        fprintf(stderr, "Error happened in IGatherv at rank: %lld", rank);
        MPI_Abort(MPI_COMM_WORLD, gatherCode);
        return EXIT_FAILURE;
    }

    /* Rank 0 can do final reporting. */
    if (rank == 0)
    {
        printf("[Rank 0] SpMV completed in %.3f ms\n", local_ms);
        fprintf(fp, "Showing results.....\n\n");
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

        /* ... Possibly write out y_global[0..some subset]. */
        /* Clean up large arrays. */
        free(y_global);
        free(recvcounts);
        free(displs);

        /* Also free x_global if we allocated it. */
        free(x_global);
        free(sendcounts);
        free(sdispls);
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
