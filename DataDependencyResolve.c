#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <papi.h>

/* ---------------------------------------------------------------------
 * GenerateElevenBandedCsrLocal:
 *    Creates a local portion of an 11-banded matrix in CSR format.
 *    Each MPI rank gets a contiguous set of rows from [offset..offset+local_n-1].
 * ---------------------------------------------------------------------*/
void GenerateElevenBandedCsrLocal(long long rank,
                                  long long N,
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
    for (long long i_local = 0; i_local < local_n; i_local++)
    {
        long long i_global = offset + i_local;
        long long start_col = (i_global - 5 < 0) ? 0 : (i_global - 5);
        long long end_col   = (i_global + 5 >= N) ? (N - 1) : (i_global + 5);
        nnz_per_row[i_local] = end_col - start_col + 1;
    }

    /* Exclusive prefix sum to build row_ptr. */
    (*row_ptr)[0] = 0;
    for (long long i = 1; i <= local_n; i++)
    {
        (*row_ptr)[i] = (*row_ptr)[i - 1] + nnz_per_row[i - 1];
    }

    long long local_nnz = (*row_ptr)[local_n];
    *col_ind = (long long *)malloc(local_nnz * sizeof(long long));
    *val     = (double *)malloc(local_nnz * sizeof(double));

    if (!(*col_ind) || !(*val))
    {
        fprintf(stderr, "[Rank %lld] Error: failed to allocate col_ind/val.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Fill col_ind and val with a (dummy) random banded pattern. */
    for (long long i_local2 = 0; i_local2 < local_n; i_local2++)
    {
        long long i_global  = offset + i_local2;
        long long start_col = (i_global - 5 < 0) ? 0 : (i_global - 5);
        long long end_col   = (i_global + 5 >= N) ? (N - 1) : (i_global + 5);
        long long idx       = (*row_ptr)[i_local2];

        for (long long j = start_col; j <= end_col; j++)
        {
            if (j < 0 || j >= N)
            {
                fprintf(stderr, "[Rank %lld] col_ind[%lld] = %lld out of range!\n",
                        rank, idx, j);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            (*col_ind)[idx] = j;
            (*val)[idx]     = (double)(rand() % 1000);
            idx++;
        }
    }

    free(nnz_per_row);
}

/* ---------------------------------------------------------------------
 * spmv_csr_local:
 *    Performs a local SpMV: y_local = A_local * x_sub,
 *    where A_local is the banded block owned by this rank.
 *    We assume x_sub already contains [halo_start..halo_end] of the global x.
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
    for (long long i_local = 0; i_local < local_n; i_local++)
    {
        double sum = 0.0;
        long long start = row_ptr[i_local];
        long long end   = row_ptr[i_local + 1];

        for (long long k = start; k < end; k++)
        {
            long long j_global = col_ind[k];
            long long j_sub    = j_global - halo_start; /* local index in x_sub */
            if (j_sub >= 0 && j_sub < local_x_size)
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

    FILE *fp = fopen("TermProjectResults.txt", "w");
    if (!fp)
    {
        fprintf(stderr, "Could not open output .txt file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fprintf(fp, "_____________________\n");
    fprintf(fp, "Term Project:\n");
    fprintf(fp, "_____________________\n\n");
    fprintf(fp, "> Eleven Banded Non Symmetric Sparse Matrix - Dense Vector Multiplication \n\n");

    /* Partition rows across ranks in a block fashion */
    long long base = N / size;
    long long rem  = N % size;
    long long local_n = base + ((rank < rem) ? 1 : 0);

    /* Offset is the starting global row index for this rank */
    long long offset = 0;
    for (long long r = 0; r < rank; r++)
    {
        offset += base + ((r < rem) ? 1 : 0);
    }

    /* Seed the random generator differently on each rank */
    srand((unsigned)time(NULL) + (unsigned)rank * 5843);

    /* Generate the local portion of the banded matrix */
    long long *row_ptr_local = NULL;
    long long *col_ind_local = NULL;
    double    *val_local     = NULL;
    GenerateElevenBandedCsrLocal(rank, N, offset, local_n,
                                 &row_ptr_local, &col_ind_local, &val_local);

    if (rank == 0)
    {
        fprintf(fp, "N = %lld, size = %lld\n", N, size);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* Determine the global columns we need for x (the halo region) */
    long long halo_start    = (offset - 5) < 0 ? 0 : (offset - 5);
    long long halo_end      = (offset + local_n - 1 + 5) >= (N - 1)
                                ? (N - 1)
                                : (offset + local_n - 1 + 5);
    long long local_x_size  = halo_end - halo_start + 1;

    /* -----------------------------
     * RANK 0: Create and fill the global x
     * ----------------------------- */
    double *x_global = NULL;
    if (rank == 0)
    {
        x_global = (double *)malloc(N * sizeof(double));
        if (!x_global)
        {
            fprintf(stderr, "Rank 0: failed to allocate x_global.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        /* Fill the entire global x with random values */
        for (long long i = 0; i < N; i++)
        {
            x_global[i] = (double)(rand() % 1000);
        }
    }

    /* -----------------------------
     * Broadcast x_global to all ranks
     * so everyone has the same x data
     * ----------------------------- */
    MPI_Bcast(x_global, (int)N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* -----------------------------
     * Each rank extracts its local portion [halo_start..halo_end]
     * into x_sub
     * ----------------------------- */
    double *x_sub = (double *)malloc(local_x_size * sizeof(double));
    if (!x_sub)
    {
        fprintf(stderr, "[Rank %lld] Error: cannot allocate x_sub of size %lld\n",
                rank, local_x_size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (long long j = 0; j < local_x_size; j++)
    {
        x_sub[j] = x_global[halo_start + j];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* Allocate y_local and compute local SpMV */
    double *y_local = (double *)calloc(local_n, sizeof(double));
    if (!y_local)
    {
        fprintf(stderr, "[Rank %lld] failed to allocate y_local.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double t0 = MPI_Wtime();
    spmv_csr_local(local_n, offset, halo_start, local_x_size,
                   row_ptr_local, col_ind_local, val_local,
                   x_sub, y_local);
    MPI_Barrier(MPI_COMM_WORLD);
    double t1       = MPI_Wtime();
    double local_ms = (t1 - t0) * 1000.0;

    /* ----------------------------------------
     * Gather local y results on rank 0
     * ----------------------------------------*/
    double *y_global = NULL;
    int *recvcounts  = NULL;
    int *displs      = NULL;

    if (rank == 0)
    {
        y_global   = (double *)malloc(N * sizeof(double));
        recvcounts = (int *)malloc(size * sizeof(int));
        displs     = (int *)malloc(size * sizeof(int));

        /* Recompute local_n for each rank to build displacements */
        long long offset_r = 0;
        for (long long rr = 0; rr < size; rr++)
        {
            long long local_n_r = base + ((rr < rem) ? 1 : 0);
            recvcounts[rr] = (int)local_n_r;
            displs[rr]     = (int)offset_r;
            offset_r       += local_n_r;
        }
    }

    int gatherCode = MPI_Gatherv(y_local, (int)local_n, MPI_DOUBLE,
                                 y_global, recvcounts, displs, MPI_DOUBLE,
                                 0, MPI_COMM_WORLD);

    if (gatherCode != MPI_SUCCESS)
    {
        char errMsg[MPI_MAX_ERROR_STRING];
        int msgLen;
        MPI_Error_string(gatherCode, errMsg, &msgLen);
        fprintf(stderr, "Error happened in Gatherv at rank: %lld -> %s\n", rank, errMsg);
        MPI_Abort(MPI_COMM_WORLD, gatherCode);
    }

    /* Rank 0: final reporting */
    if (rank == 0)
    {
        fprintf(fp, "> Showing results.....\n\n");
        fprintf(fp, "y_global: ");
        for (long long i = 0; i < 5 && i < N; i++)
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

        /* Cleanup memory on rank 0 */
        free(y_global);
        free(recvcounts);
        free(displs);
        free(x_global);
    }

    /* Cleanup local data for all ranks */
    free(y_local);
    free(x_sub);
    free(row_ptr_local);
    free(col_ind_local);
    free(val_local);
    fclose(fp);

    MPI_Finalize();
    return 0;
}
